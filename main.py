import copy
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import get_config
from data_utils import get_dataset
from resnet18 import ResNet18
from effnet import EfficientNetB0_CIFAR


def client_update(model, loader, epochs, device, lr, training_flops_per_sample):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total_flops = 0.0
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_flops += training_flops_per_sample * data.size(0)
    return model.state_dict(), total_flops


def evaluate(model, loader, device, forward_flops_per_sample=None):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    total_flops = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            if forward_flops_per_sample is not None:
                total_flops += forward_flops_per_sample * data.size(0)
    return loss / total, correct / total, total_flops
def tensor_dict_bytes(tensor_dict):
    return sum(t.element_size() * t.nelement() for t in tensor_dict.values())


def dict_to_tensor(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])


def estimate_forward_flops(model, sample_input, device):
    flops = 0.0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal flops
        if not isinstance(output, torch.Tensor):
            return
        batch_size = inputs[0].shape[0]
        out_channels = module.out_channels
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels // module.groups
        output_height = output.shape[2]
        output_width = output.shape[3]
        conv_ops = kernel_height * kernel_width * in_channels
        flops += batch_size * output_height * output_width * out_channels * conv_ops * 2
        if module.bias is not None:
            flops += batch_size * output_height * output_width * out_channels

    def linear_hook(module, inputs, output):
        nonlocal flops
        if not isinstance(output, torch.Tensor):
            return
        batch_size = inputs[0].shape[0]
        flops += batch_size * module.in_features * module.out_features * 2
        if module.bias is not None:
            flops += batch_size * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    model_state = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample_input.to(device))
    if model_state:
        model.train()
    else:
        model.eval()

    for handle in handles:
        handle.remove()

    batch_size = sample_input.shape[0] if sample_input.shape[0] > 0 else 1
    return flops / batch_size


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = get_config()
    
    wandb.init(
    project="compression_FL",
    
    config={k: v for k, v in vars(args).items()}
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    client_train_data, client_val_data, test_data, n_classes, _, _ = get_dataset(args)

    model_name = args.model.lower()
    if model_name in {"resnet", "resnet18"}:
        global_model = ResNet18(num_classes=n_classes).to(device)
    elif model_name in {"effnet", "efficientnet", "efficientnetb0"}:
        if args.dataset not in {"cifar10", "cifar100"}:
            raise ValueError("EfficientNetB0_CIFAR is only compatible with CIFAR datasets.")
        global_model = EfficientNetB0_CIFAR(num_classes=n_classes).to(device)
    else:
        raise ValueError(f"Unsupported model selection: {args.model}")

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    n_clients = len(client_train_data)

    sample_input = None
    for subset in client_train_data:
        if len(subset) > 0:
            sample_input = subset[0][0].unsqueeze(0)
            break
    if sample_input is None:
        raise ValueError("Unable to estimate FLOPs because no client data is available.")

    per_sample_forward_flops = estimate_forward_flops(global_model, sample_input, device)
    per_sample_training_flops = per_sample_forward_flops * 2

    total_upload_traffic = 0
    total_download_traffic = 0
    total_flops = 0.0

    val_loaders = []
    for subset in client_val_data:
        if len(subset) > 0:
            val_loaders.append(DataLoader(subset, batch_size=args.batch_size, shuffle=False))
        else:
            val_loaders.append(None)

    # Log metrics for epoch 0 (initialization)
    initial_report = {"epoch": 0}
    initial_round_flops = 0.0

    acc_clients = []
    for idx, subset in enumerate(client_val_data):
        if len(subset) == 0:
            acc_clients.append(0.0)
            continue

        if val_loaders[idx] is None:
            val_loaders[idx] = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
        _, client_acc, val_flops = evaluate(global_model, val_loaders[idx], device, per_sample_forward_flops)
        acc_clients.append(client_acc)
        initial_round_flops += val_flops

    acc_clients_mean = np.mean(acc_clients) if acc_clients else 0.0
    acc_clients_std = np.std(acc_clients) if acc_clients else 0.0

    loss, acc, test_flops = evaluate(global_model, test_loader, device, per_sample_forward_flops)
    initial_round_flops += test_flops

    acc_servers = [acc]
    acc_servers_mean = np.mean(acc_servers)
    acc_servers_std = np.std(acc_servers)

    initial_report["cos_lowest"] = float("nan")
    initial_report["cos_highest"] = float("nan")
    initial_report["training_loss_lowest"] = float("nan")
    initial_report["training_loss_highest"] = float("nan")
    initial_report["acc_clients_lowest"] = acc_clients_mean - acc_clients_std
    initial_report["acc_clients_highest"] = acc_clients_mean + acc_clients_std
    initial_report["acc_servers_lowest"] = acc_servers_mean - acc_servers_std
    initial_report["acc_servers_highest"] = acc_servers_mean + acc_servers_std

    initial_report["upload_traffic"] = 0
    initial_report["download_traffic"] = 0
    initial_report["upload_traffic_per_client"] = 0
    initial_report["overall_traffic"] = total_upload_traffic + total_download_traffic
    initial_report["round_flops"] = initial_round_flops
    total_flops += initial_round_flops
    initial_report["total_flops"] = total_flops

    wandb.log(initial_report)

    for round_idx in range(args.n_epoch):
        m = max(1, int(args.client_fraction * n_clients))
        selected = random.sample(range(n_clients), m)
        active_clients = len(selected)
        report = {"epoch": round_idx + 1}

        cos = []
        training_loss = []

        selected_sizes = [len(client_train_data[idx]) for idx in selected]
        total_size = sum(selected_sizes)

        global_params = dict_to_tensor(global_model.state_dict())
        global_state_reference = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

        weighted_state = None
        upload_traffic_round = 0
        round_flops = 0.0

        for client_order, idx in enumerate(selected):
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=True)
            state_dict, client_flops = client_update(
                local_model, loader, args.n_client_epoch, device, args.lr, per_sample_training_flops
            )
            round_flops += client_flops

            local_params = dict_to_tensor(state_dict)
            cos.append(F.cosine_similarity(local_params, global_params, dim=0).item())

            train_loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=False)
            train_loss, _, eval_train_flops = evaluate(
                local_model, train_loader, device, per_sample_forward_flops
            )
            training_loss.append(train_loss)
            round_flops += eval_train_flops

            state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
            weight = selected_sizes[client_order] / total_size if total_size > 0 else 0.0

            if weighted_state is None:
                weighted_state = {k: tensor * weight for k, tensor in state_dict_cpu.items()}
            else:
                for key in weighted_state.keys():
                    weighted_state[key] += state_dict_cpu[key] * weight

            for key, tensor in state_dict_cpu.items():
                diff = tensor - global_state_reference[key]
                upload_traffic_round += diff.element_size() * diff.nelement()

            del local_params
            del state_dict
            del loader
            del train_loader
            del state_dict_cpu
            del local_model
            cleanup_memory()

        del global_state_reference

        global_state = weighted_state if weighted_state is not None else global_model.state_dict()
        global_model.load_state_dict(global_state)

        loss, acc, test_flops = evaluate(global_model, test_loader, device, per_sample_forward_flops)
        round_flops += test_flops

        cos_mean = np.mean(cos)
        cos_std = np.std(cos)

        training_loss_mean = np.mean(training_loss)
        training_loss_std = np.std(training_loss)

        acc_clients = []
        for idx, subset in enumerate(client_val_data):
            if len(subset) == 0:
                acc_clients.append(0.0)
                continue

            if val_loaders[idx] is None:
                val_loaders[idx] = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
            _, a, val_flops = evaluate(global_model, val_loaders[idx], device, per_sample_forward_flops)
            acc_clients.append(a)
            round_flops += val_flops

        acc_clients_mean = np.mean(acc_clients) if acc_clients else 0.0
        acc_clients_std = np.std(acc_clients) if acc_clients else 0.0

        acc_servers = [acc]
        acc_servers_mean = np.mean(acc_servers)
        acc_servers_std = np.std(acc_servers)

        report["cos_lowest"] = cos_mean - cos_std
        report["cos_highest"] = cos_mean + cos_std
        report["training_loss_lowest"] = training_loss_mean - training_loss_std
        report["training_loss_highest"] = training_loss_mean + training_loss_std
        report["acc_clients_lowest"] = acc_clients_mean - acc_clients_std
        report["acc_clients_highest"] = acc_clients_mean + acc_clients_std
        report["acc_servers_lowest"] = acc_servers_mean - acc_servers_std
        report["acc_servers_highest"] = acc_servers_mean + acc_servers_std

        model_size_bytes = tensor_dict_bytes(global_state)
        download_traffic = model_size_bytes * active_clients
        upload_traffic = upload_traffic_round
        total_upload_traffic += upload_traffic
        total_download_traffic += download_traffic
        report["upload_traffic"] = upload_traffic
        report["download_traffic"] = download_traffic
        report["upload_traffic_per_client"] = model_size_bytes
        report["overall_traffic"] = total_upload_traffic + total_download_traffic
        report["round_flops"] = round_flops
        total_flops += round_flops
        report["total_flops"] = total_flops

        wandb.log(report)

        print(f"Round {round_idx + 1}, Clients Acc: {acc_clients}, Server Acc: {acc_servers}")
        cleanup_memory()

    print("Training complete.")


if __name__ == "__main__":
    main()
