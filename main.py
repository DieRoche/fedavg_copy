import copy
import gc
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from config import get_config
from data_utils import get_dataset
from compression import compress_csr, decompress_csr, pack_csr, unpack_csr
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
def tensor_dict_bytes(tensor_dict):
    return sum(t.element_size() * t.nelement() for t in tensor_dict.values())


def compressed_tensor_bytes(tensor, compression_type):
    if tensor.ndim == 1:
        return tensor.element_size() * tensor.nelement()

    dense_tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        dense = dense_tensor.numpy()
    elif tensor.ndim == 4:
        dense = dense_tensor.reshape(dense_tensor.size(0), -1).numpy()
    else:
        return tensor.element_size() * tensor.nelement()
    if compression_type == "CSR":
        csr = compress_csr(dense)
        packet = pack_csr(csr)
        return len(packet)
    raise ValueError(f"Unknown compression type: {compression_type}")


def quantize_tensor(tensor, bits):
    if bits is None:
        return tensor.clone()
    if bits == 16:
        return tensor.to(torch.float16).to(dtype=tensor.dtype)

    if bits != 8:
        raise ValueError(f"Unsupported quantization bits: {bits}")

    if tensor.numel() == 0:
        return tensor.clone()

    absmax = tensor.abs().max()
    if absmax.item() == 0.0:
        return torch.zeros_like(tensor)

    qmax = (1 << (bits - 1)) - 1
    scale = absmax / qmax
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return (q.to(torch.float32) * scale).to(dtype=tensor.dtype)


def quantize_tensor_for_transport(tensor, bits):
    if bits is None:
        return tensor.clone(), None
    if bits == 16:
        return tensor.to(torch.float16), None
    if bits != 8:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    if tensor.numel() == 0:
        return torch.empty_like(tensor, dtype=torch.int8), 1.0

    absmax = tensor.abs().max()
    if absmax.item() == 0.0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0

    qmax = (1 << (bits - 1)) - 1
    scale = (absmax / qmax).item()
    q = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor_from_transport(q_tensor, scale, bits, target_dtype):
    if bits is None:
        return q_tensor.to(dtype=target_dtype)
    if bits == 16:
        return q_tensor.to(dtype=target_dtype)
    if bits != 8:
        raise ValueError(f"Unsupported quantization bits: {bits}")
    return (q_tensor.to(torch.float32) * float(scale)).to(dtype=target_dtype)


def quantize_state_dict(state_dict, bits):
    return {k: quantize_tensor(v, bits) for k, v in state_dict.items()}


def serialize_tensor_payload(tensor, bits, enable_sparse_masking, dynamic_quantization=False):
    cpu_tensor = tensor.detach().cpu()
    transport_dtype = str(cpu_tensor.dtype)

    use_csr = enable_sparse_masking and cpu_tensor.ndim in (2, 4)
    if use_csr:
        csr_shape = tuple(cpu_tensor.shape)
        dense_2d = cpu_tensor if cpu_tensor.ndim == 2 else cpu_tensor.reshape(cpu_tensor.size(0), -1)
        dense_numel = dense_2d.numel()
        csr = compress_csr(dense_2d.numpy())
        values = torch.from_numpy(csr.values)
        q_values, scale = quantize_tensor_for_transport(values, bits)
        q_csr = type(csr)(
            values=q_values.cpu().numpy(),
            col_indices=csr.col_indices,
            row_ptr=csr.row_ptr,
            shape=csr.shape,
        )
        packet = pack_csr(q_csr, dynamic_quantization=dynamic_quantization, scale=scale)
        return {
            "mode": "csr",
            "packet": packet,
            "bits": bits,
            "transport_dtype": transport_dtype,
            "orig_shape": csr_shape,
            "nnz": int(csr.values.size),
            "dense_numel": int(dense_numel),
        }, len(packet)

    q_tensor, scale = quantize_tensor_for_transport(cpu_tensor, bits)
    payload = {
        "mode": "dense",
        "q_tensor": q_tensor,
        "scale": scale,
        "bits": bits,
        "transport_dtype": transport_dtype,
        "orig_shape": tuple(cpu_tensor.shape),
    }
    payload_bytes = quantized_tensor_bytes(cpu_tensor, bits)
    return payload, payload_bytes


def deserialize_tensor_payload(payload):
    bits = payload.get("bits", None)
    target_dtype = getattr(torch, payload["transport_dtype"].split(".")[-1])

    if payload["mode"] == "csr":
        csr_q, header = unpack_csr(payload["packet"])
        val_bits = header["val_bits"]
        if val_bits == 8:
            if header.get("has_scale", False):
                bits = 8
                scale = header["scale"]
            else:
                bits = None
                scale = None
        elif val_bits == 16:
            bits = 16
            scale = None
        elif val_bits in (32, 64):
            bits = None
            scale = None
        else:
            raise ValueError(f"Unsupported val_bits in packet: {val_bits}")
        q_values = torch.from_numpy(csr_q.values.copy())
        values = dequantize_tensor_from_transport(q_values, scale, bits, target_dtype).numpy()
        csr_deq = type(csr_q)(
            values=values,
            col_indices=csr_q.col_indices,
            row_ptr=csr_q.row_ptr,
            shape=csr_q.shape,
        )
        dense = decompress_csr(csr_deq)
        tensor = torch.from_numpy(dense).reshape(payload["orig_shape"])
        return tensor.to(dtype=target_dtype)

    if payload["mode"] == "dense":
        q_tensor = payload["q_tensor"]
        tensor = dequantize_tensor_from_transport(q_tensor, payload["scale"], bits, target_dtype)
        return tensor.reshape(payload["orig_shape"]).to(dtype=target_dtype)

    raise ValueError(f"Unknown payload mode: {payload['mode']}")


def estimate_quantization_flops(numel, bits):
    if bits is None:
        return 0
    if bits == 16:
        return int(numel)
    if bits == 8:
        # abs/max + scale/div + round/clamp (coarse estimate).
        return int(4 * numel)
    raise ValueError(f"Unsupported quantization bits: {bits}")


def estimate_dequantization_flops(numel, bits):
    if bits is None:
        return 0
    if bits == 16:
        return int(numel)
    if bits == 8:
        # one multiply per value.
        return int(numel)
    raise ValueError(f"Unsupported quantization bits: {bits}")


def estimate_payload_compression_flops(tensor, enable_sparse_masking, bits):
    if not (enable_sparse_masking and tensor.ndim in (2, 4)):
        return estimate_quantization_flops(tensor.numel(), bits)
    dense_2d = tensor if tensor.ndim == 2 else tensor.reshape(tensor.size(0), -1)
    dense_numel = dense_2d.numel()
    nnz = int(torch.count_nonzero(dense_2d).item())
    # Dense scan + value/index materialization for non-zero entries.
    csr_flops = dense_numel + (2 * nnz)
    return csr_flops + estimate_quantization_flops(nnz, bits)


def estimate_payload_decompression_flops(payload):
    if payload["mode"] != "csr":
        bits = payload.get("bits", None)
        return estimate_dequantization_flops(payload["q_tensor"].numel(), bits)
    dense_numel = int(payload.get("dense_numel", 0))
    nnz = int(payload.get("nnz", 0))
    bits = payload.get("bits", None)
    if bits is None:
        packet = payload.get("packet", b"")
        if len(packet) >= 13:
            val_bits = int(packet[12])
            if val_bits == 8:
                bits = 8
            elif val_bits == 16:
                bits = 16
            else:
                bits = None
    # Zero-fill dense buffer + scatter each non-zero value.
    csr_flops = dense_numel + nnz
    return csr_flops + estimate_dequantization_flops(nnz, bits)


def quantized_tensor_bytes(tensor, bits):
    numel = tensor.numel()
    if bits is None:
        return numel * tensor.element_size()
    if bits == 16:
        return numel * 2
    if bits == 8:
        return numel + 4
    raise ValueError(f"Unsupported quantization bits: {bits}")


def compressed_quantized_tensor_bytes(tensor, compression_type, bits, dynamic_quantization=False):
    if tensor.ndim == 1:
        return quantized_tensor_bytes(tensor, bits)

    dense_tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        dense = dense_tensor.numpy()
    elif tensor.ndim == 4:
        dense = dense_tensor.reshape(dense_tensor.size(0), -1).numpy()
    else:
        return quantized_tensor_bytes(tensor, bits)

    if compression_type == "CSR":
        csr = compress_csr(dense)
        values = torch.from_numpy(csr.values)
        q_values, scale = quantize_tensor_for_transport(values, bits)
        q_csr = type(csr)(
            values=q_values.cpu().numpy(),
            col_indices=csr.col_indices,
            row_ptr=csr.row_ptr,
            shape=csr.shape,
        )
        packet = pack_csr(
            q_csr,
            dynamic_quantization=dynamic_quantization,
            scale=scale,
        )
        return len(packet)

    raise ValueError(f"Unknown compression type: {compression_type}")


def tensor_dict_payload_bytes(tensor_dict, args):
    if args.enable_sparse_masking:
        return sum(
            compressed_quantized_tensor_bytes(
                tensor,
                args.sparsity_compression,
                args.quantization_bits,
                args.dynamic_quantization,
            )
            for tensor in tensor_dict.values()
        )
    return sum(
        quantized_tensor_bytes(tensor, args.quantization_bits)
        for tensor in tensor_dict.values()
    )


def dict_to_tensor(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_sparse_mask(delta_dict, param_keys, args):
    """Apply Gauss-Southwell style masking to dense deltas.

    The function keeps the payload dense (identical tensor shapes) but zeros out
    coordinates not selected by the magnitude-based mask.
    """

    abs_delta_flat = torch.cat([delta_dict[k].abs().reshape(-1) for k in param_keys])
    total_params = abs_delta_flat.numel()

    gs_flops = 0

    if not args.enable_sparse_masking or args.sparsity_rate == 0.0:
        mask_flat = torch.ones_like(abs_delta_flat, dtype=torch.bool)
    else:
        gs_flops += total_params  # absolute-value scan
        if args.sparsity_rate >= 1.0:
            threshold = abs_delta_flat.max()
            gs_flops += total_params
        else:
            threshold = torch.quantile(abs_delta_flat, args.sparsity_rate)
            gs_flops += total_params
        mask_flat = abs_delta_flat >= threshold
        gs_flops += total_params

        density = mask_flat.float().mean().item()
        if density < args.sparsity_min_density:
            k = max(1, math.ceil(args.sparsity_min_density * total_params))
            # Recompute mask using top-k to enforce minimum density.
            topk_values, _ = torch.topk(abs_delta_flat, k)
            threshold = topk_values[-1]
            mask_flat = abs_delta_flat >= threshold
            gs_flops += total_params + k

    density = mask_flat.float().mean().item()
    sparsity = 1.0 - density
    assert 0.0 <= sparsity <= 1.0, "Sparsity out of bounds"

    delta_flat = torch.cat([delta_dict[k].reshape(-1) for k in param_keys])
    l2_norm_delta = torch.norm(delta_flat).item()

    delta_sparse = {}
    start = 0
    for key in param_keys:
        numel = delta_dict[key].numel()
        mask_tensor = mask_flat[start : start + numel].reshape(delta_dict[key].shape)
        delta_sparse[key] = delta_dict[key] * mask_tensor
        start += numel

    delta_sparse_flat = torch.cat([delta_sparse[k].reshape(-1) for k in param_keys])
    l2_norm_delta_sparse = torch.norm(delta_sparse_flat).item()

    metrics = {
        "total_params": total_params,
        "nonzero_params": int(mask_flat.sum().item()),
        "density": density,
        "sparsity": sparsity,
        "l2_norm_delta": l2_norm_delta,
        "l2_norm_delta_sparse": l2_norm_delta_sparse,
        "gs_flops": int(gs_flops),
    }

    assert metrics["nonzero_params"] <= metrics["total_params"], "Mask overflow"
    return delta_sparse, metrics


def main():
    args = get_config()

    selected_clients = int(args.n_client * args.client_fraction)
    quantization_bits = getattr(args, "quantization_bits", None)
    sparse_masking_enabled = getattr(args, "enable_sparse_masking", False)
    sparsity_rate = getattr(args, "sparsity_rate", None)
    if sparse_masking_enabled and sparsity_rate is not None and sparsity_rate > 0.0:
        compression_prefix = f"GS{quantization_bits}" if quantization_bits is not None else "GS"
    else:
        compression_prefix = "fedavg"

    run_name_parts = [compression_prefix, args.dataset, args.model]
    if sparse_masking_enabled and sparsity_rate is not None and sparsity_rate > 0.0:
        sparsity_pct = sparsity_rate * 100.0 if sparsity_rate <= 1.0 else sparsity_rate
        sparsity_label = f"{sparsity_pct:g}"
        run_name_parts.append(sparsity_label)
    run_name_parts.append(f"{selected_clients}cl")
    run_name = "_".join(run_name_parts)

    if args.wandb_enabled:
        wandb.init(
            project="Gauss-Southwell",
            name=run_name,
            config={k: v for k, v in vars(args).items()},
        )
    
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

    total_upload_traffic = 0
    total_download_traffic = 0
    total_compression_flops = 0
    total_decompression_flops = 0
    total_gs_flops = 0
    total_flops = 0

    for round_idx in range(args.n_epoch):
        m = max(1, int(args.client_fraction * n_clients))
        selected = random.sample(range(n_clients), m)
        selected_count = len(selected)
        report = {}

        cos = []
        training_loss = []

        selected_sizes = [len(client_train_data[idx]) for idx in selected]
        assert len(selected_sizes) == selected_count
        total_size = sum(selected_sizes)

        global_params = dict_to_tensor(global_model.state_dict())
        global_state_reference = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        global_state_device = {k: v.to(device) for k, v in global_state_reference.items()}
        param_keys = list(global_state_reference.keys())

        aggregated_delta = None
        upload_traffic_round = 0
        compression_flops_round = 0
        decompression_flops_round = 0
        gs_flops_round = 0
        per_client_upload_bytes = []
        client_sparsity_metrics = []

        for client_order, idx in enumerate(selected):
            # Only active/selected clients receive the current server global model.
            local_model = copy.deepcopy(global_model)
            loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=True)
            state_dict = client_update(local_model, loader, args.n_client_epoch, device, args.lr)

            local_params = dict_to_tensor(state_dict)
            cos.append(F.cosine_similarity(local_params, global_params, dim=0).item())

            train_loader = DataLoader(client_train_data[idx], batch_size=args.batch_size, shuffle=False)
            train_loss, _ = evaluate(local_model, train_loader, device)
            training_loss.append(train_loss)

            delta_dict = {k: state_dict[k] - global_state_device[k] for k in param_keys}
            # Apply Gauss-Southwell masking only for the payload that is transmitted back
            # to the server. The dense delta is kept for local metrics and aggregation
            # bookkeeping.
            delta_sparse, metrics = apply_sparse_mask(delta_dict, param_keys, args)
            state_dict_cpu = {k: v.detach().cpu() for k, v in delta_sparse.items()}
            payload_dict = {}
            reconstructed_state_dict = {}
            client_upload_bytes = 0
            for key, tensor in state_dict_cpu.items():
                payload, payload_size = serialize_tensor_payload(
                    tensor,
                    args.quantization_bits,
                    args.enable_sparse_masking,
                    args.dynamic_quantization,
                )
                payload_dict[key] = payload
                compression_flops_round += estimate_payload_compression_flops(
                    tensor,
                    args.enable_sparse_masking,
                    args.quantization_bits,
                )
                decompression_flops_round += estimate_payload_decompression_flops(payload)
                reconstructed_state_dict[key] = deserialize_tensor_payload(payload)
                client_upload_bytes += payload_size
            weight = selected_sizes[client_order] / total_size if total_size > 0 else 0.0

            if aggregated_delta is None:
                aggregated_delta = {k: tensor * weight for k, tensor in reconstructed_state_dict.items()}
            else:
                for key in aggregated_delta.keys():
                    aggregated_delta[key] += reconstructed_state_dict[key] * weight

            upload_traffic_round += client_upload_bytes
            per_client_upload_bytes.append(client_upload_bytes)

            metrics.update({"client_id": idx, "round": round_idx + 1})
            metrics["density"] = metrics.get("density", 0.0)
            metrics["sparsity"] = metrics.get("sparsity", 0.0)
            assert abs(metrics["density"] + metrics["sparsity"] - 1.0) < 1e-6
            client_sparsity_metrics.append(metrics)
            gs_flops_round += metrics.get("gs_flops", 0)

            if args.wandb_enabled:
                wandb.log(
                    {
                        "client_id": idx,
                        "round": round_idx + 1,
                        "sparsity": metrics["sparsity"],
                        "density": metrics["density"],
                    },
                    step=round_idx + 1,
                    commit=False,
                )

            del local_params
            del state_dict
            del loader
            del train_loader
            del state_dict_cpu
            del payload_dict
            del reconstructed_state_dict
            del local_model
            cleanup_memory()

        del global_state_reference
        del global_state_device

        aggregated_delta = aggregated_delta if aggregated_delta is not None else {}
        global_state = global_model.state_dict()
        for key in param_keys:
            delta_tensor = aggregated_delta.get(key, torch.zeros_like(global_state[key]))
            global_state[key] = global_state[key] + delta_tensor.to(global_state[key].device)

        global_model.load_state_dict(global_state)

        loss, acc = evaluate(global_model, test_loader, device)
        
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

        if client_sparsity_metrics:
            sparsities = [m["sparsity"] for m in client_sparsity_metrics]
            densities = [m["density"] for m in client_sparsity_metrics]
            report["sparsity/mean"] = float(np.mean(sparsities))
            report["sparsity/min"] = float(np.min(sparsities))
            report["sparsity/max"] = float(np.max(sparsities))
            report["density/mean"] = float(np.mean(densities))
            delta_norms = [m.get("l2_norm_delta", 0.0) for m in client_sparsity_metrics]
            delta_sparse_norms = [m.get("l2_norm_delta_sparse", 0.0) for m in client_sparsity_metrics]
            report["delta_norm/mean"] = float(np.mean(delta_norms))
            report["delta_sparse_norm/mean"] = float(np.mean(delta_sparse_norms))

        report["cos_lowest"] = cos_mean - cos_std
        report["cos_highest"] = cos_mean + cos_std
        report["training_loss_lowest"] = training_loss_mean - training_loss_std
        report["training_loss_highest"] = training_loss_mean + training_loss_std
        report["acc_clients_lowest"] = acc_clients_mean - acc_clients_std
        report["acc_clients_highest"] = acc_clients_mean + acc_clients_std
        report["acc_servers_lowest"] = acc_servers_mean - acc_servers_std
        report["acc_servers_highest"] = acc_servers_mean + acc_servers_std
        report["round"] = round_idx + 1

        model_size_bytes = tensor_dict_bytes(global_state)
        # The global model is only transmitted to clients selected in this round.
        download_traffic = model_size_bytes * selected_count
        upload_traffic = upload_traffic_round
        total_upload_traffic += upload_traffic
        total_download_traffic += download_traffic
        total_compression_flops += compression_flops_round
        total_decompression_flops += decompression_flops_round
        total_gs_flops += gs_flops_round
        round_flops_compression = compression_flops_round + decompression_flops_round
        round_flops = gs_flops_round + round_flops_compression
        total_flops += round_flops
        total_flops_compression = total_compression_flops + total_decompression_flops
        report["upload_traffic"] = upload_traffic
        report["download_traffic"] = download_traffic
        report["compression_flops"] = compression_flops_round
        report["decompression_flops"] = decompression_flops_round
        report["gs_flops"] = gs_flops_round
        report["compression_plus_decompression_flops"] = round_flops_compression
        report["round_flops"] = round_flops
        report["gs_plus_compression_plus_decompression_flops"] = (
            round_flops
        )
        report["total_compression_flops"] = total_compression_flops
        report["total_decompression_flops"] = total_decompression_flops
        report["total_gs_flops"] = total_gs_flops
        report["total_compression_plus_decompression_flops"] = total_flops_compression
        report["total_flops_compression"] = total_flops_compression
        report["total_flops"] = total_flops
        report["total_gs_plus_compression_plus_decompression_flops"] = (
            total_flops
        )
        report["upload_traffic_per_client"] = float(
            np.mean(per_client_upload_bytes) if per_client_upload_bytes else 0.0
        )
        report["active_clients"] = selected_count
        report["overall_traffic"] = total_upload_traffic + total_download_traffic

        if args.wandb_enabled:
            wandb.log(report, step=round_idx + 1, commit=True)

        print(f"Round {round_idx + 1}, Clients Acc: {acc_clients}, Server Acc: {acc_servers}")
        cleanup_memory()

    print("Training complete.")


if __name__ == "__main__":
    main()
