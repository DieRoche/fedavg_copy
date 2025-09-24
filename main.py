import copy
import gc
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import get_config
from data_utils import get_dataset
from resnet18 import ResNet18

def client_update(model, loader, epochs, device, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return loss / total, correct / total


def fedavg(state_dicts, weights):
    avg = OrderedDict()
    for k in state_dicts[0].keys():
        avg[k] = sum(weight * sd[k] for sd, weight in zip(state_dicts, weights))
    return avg


def tensor_dict_bytes(tensor_dict):
    return sum(t.element_size() * t.nelement() for t in tensor_dict.values())


def dict_to_tensor(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    client_train_data, client_val_data, test_data, n_classes, _, _ = get_dataset(args)
    global_model = ResNet18(num_classes=n_classes).to(device)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    n_clients = len(client_train_data)

    val_loaders = []
    for subset in client_val_data:
        if len(subset) > 0:
            val_loaders.append(DataLoader(subset, batch_size=args.batch_size, shuffle=False))
        else:
            val_loaders.append(None)

    wandb.init(project="compression_FL", config=vars(args))
    total_upload_traffic = 0
    total_download_traffic = 0

    for round_idx in range(args.n_epoch):
        m = max(1, int(args.client_fraction * n_clients))
        selected = random.sample(range(n_clients), m)

        cos = []
        training_loss = []

        global_state = global_model.state_dict()
        global_params = dict_to_tensor({k: v.detach().cpu() for k, v in global_state.items()})
        aggregated_state = OrderedDict(
            (k, torch.zeros_like(v, device="cpu")) for k, v in global_state.items()
        )

        total_selected_size = sum(len(client_train_data[idx]) for idx in selected)
        if total_selected_size == 0:
            print("Skipping round due to empty selected clients.")
            continue
        round_upload_traffic = 0

        for idx in selected:
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=True)
            state_dict = client_update(local_model, loader, args.n_client_epoch, device, args.lr)

            state_dict_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
            weight = len(client_train_data[idx]) / total_selected_size
            for k in aggregated_state:
                aggregated_state[k] += state_dict_cpu[k] * weight

            local_params = dict_to_tensor(state_dict_cpu)
            cos.append(F.cosine_similarity(local_params, global_params, dim=0).item())

            train_loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=False)
            train_loss, _ = evaluate(local_model, train_loader, device)
            training_loss.append(train_loss)

            round_upload_traffic += tensor_dict_bytes(state_dict_cpu)

            del loader, train_loader, local_model, state_dict, state_dict_cpu, local_params
            cleanup_memory()

        global_model.load_state_dict(aggregated_state)

        loss, acc = evaluate(global_model, test_loader, device)
        report = {"round": round_idx + 1, "loss": loss, "accuracy": acc}

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
            _, a = evaluate(global_model, val_loaders[idx], device)
            acc_clients.append(a)

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

        broadcast_bytes = tensor_dict_bytes(aggregated_state)
        download_traffic = broadcast_bytes * len(selected)
        total_upload_traffic += round_upload_traffic
        total_download_traffic += download_traffic
        report["upload_traffic"] = round_upload_traffic
        report["download_traffic"] = download_traffic
        report["overall_traffic"] = total_upload_traffic + total_download_traffic

        wandb.log(report)

        del aggregated_state, global_state

        print(f"Round {round_idx + 1}, Clients Acc: {acc_clients}, Server Acc: {acc_servers}")
        cleanup_memory()

    print("Training complete.")


if __name__ == "__main__":
    main()
