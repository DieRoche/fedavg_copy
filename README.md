# FedAvg Implementation Audit

## 1. Scope

This document audits the **actual executable FedAvg path** in this repository, using only code-level evidence from:

- `main.py` (training loop, client updates, aggregation, traffic/FLOPs/W&B logging)
- `config.py` (CLI flags/defaults)
- `data_utils.py` (dataset split + client partitioning)
- `compression.py` (compression/packing helpers and whether they are in active use)

The goal is to describe what this code really does for `METHOD_NAME = FedAvg`, not the theoretical paper algorithm.

---

## 2. High-Level Verdict

**Verdict:** The active pipeline is a **FedAvg-style weighted aggregation of client deltas**, but the repository’s “FedAvg” execution path is coupled to optional sparsity/compression/quantization machinery and traffic/FLOPs estimators.

- If run with defaults (`--enable_sparse_masking` off, `--quantization_bits` none), behavior is close to dense FedAvg delta aggregation.
- `--method` exists but is not used to switch algorithm logic; method activation is effectively controlled by other flags.
- Communication is simulated in-memory; no network serialization stack is used beyond optional byte-packet creation for accounting/reconstruction.
- FLOPs logged in `round_flops` / `total_flops` include compute-only training/aggregation/evaluation FLOPs and exclude compression-pipeline FLOPs.

---

## 3. Actual Execution Pipeline

### End-to-end active path (per run)

1. Parse CLI args via `get_config()`.
2. Build dataset partitions with Dirichlet client split and per-client 80/20 train/val split.
3. Initialize `ResNet18` global model.
4. For each round:
   - Sample active clients (`m = int(client_fraction * n_clients)`, min 1).
   - For each selected client:
     - Copy global model.
     - Train locally with SGD for `n_client_epoch` epochs.
     - Compute local delta: `local_state - global_state`.
     - Apply optional sparse masking to delta (`apply_sparse_mask`).
     - Serialize each parameter tensor payload (`serialize_tensor_payload`), estimate compression/decompression FLOPs, and immediately deserialize (`deserialize_tensor_payload`).
     - Weighted-sum aggregate reconstructed deltas by client sample fraction.
   - Update global model by `global += aggregated_delta`.
   - Evaluate on global test split and each client validation split.
   - Compute/report traffic + FLOPs + accuracy/cos/loss metrics; log to W&B if enabled.

### Execution-flow table

| Stage | Function(s) / location | Output object |
|---|---|---|
| Args | `config.get_config` | `args` namespace |
| Data split | `data_utils.get_dataset`, `split_noniid` | `client_train_data`, `client_val_data`, `test_data` |
| Local train | `client_update` | `state_dict` |
| Delta build | `main` loop (`delta_dict = state_dict - global`) | `delta_dict` |
| Optional masking | `apply_sparse_mask` | `delta_sparse`, masking metrics |
| Payload encode | `serialize_tensor_payload` | per-parameter payload dict/packet |
| Payload decode | `deserialize_tensor_payload` | reconstructed per-parameter tensor |
| Server aggregate | round loop in `main` | `aggregated_delta` |
| Global update | `global_state[key] += delta_tensor` | updated global weights |
| Logging | round `report` + `wandb.log` | metric stream |

---

## 4. Method Activation and Required Flags

### FedAvg activation reality

`--method` default is `fedavg`, but there is **no branch on `args.method`** in `main.py`. So “FedAvg” is not activated by method dispatch; it is the only implemented main loop.

### Flags that actually affect behavior

| Flag | Default | Active effect on pipeline |
|---|---:|---|
| `--client_fraction` | `1.0` | Controls number of active clients each round |
| `--n_client_epoch` | `5` | Local epochs per selected client |
| `--lr` | `0.01` | SGD learning rate in `client_update` |
| `--enable_sparse_masking` | `False` | Enables masking in `apply_sparse_mask`; also allows CSR path in serialization |
| `--sparsity_rate` | `0.0` | If >0 with masking enabled, enforces magnitude-based pruning mask |
| `--sparsity_min_density` | `0.0` | Optional minimum density floor via top-k fallback |
| `--quantization_bits` | `None` | Controls dense/CSR value quantization (none/16/8) during payload serialization |
| `--dynamic_quantization` | `False` | Controls adaptive CSR index bit-width (16/32) in packetization |
| `--wandb_enabled` | `True` | Enables W&B logging |

### Specific vs generic vs unused helper

- **FedAvg-specific enough in active path:** weighted aggregation of client updates into global model (`global += weighted_delta`).
- **Generic FL plumbing:** client sampling, local SGD, evaluation, report logging.
- **Helper exists but unused in active path:** `quantize_state_dict`, `compressed_tensor_bytes`, `compressed_quantized_tensor_bytes` (used only via `tensor_dict_payload_bytes`, and that function itself is unused), `tensor_dict_payload_bytes`.

---

## 5. Client-Side Processing After Local Training

Immediately after local training, client object is `state_dict` returned by `client_update`.

Then active transformations before server aggregation:

1. **Delta computation**: `delta_dict[k] = state_dict[k] - global_state_device[k]`.
2. **Optional sparse masking** (`apply_sparse_mask`):
   - Always computes mask metrics.
   - If masking disabled or `sparsity_rate==0`, mask is all-ones (no effective pruning).
   - If enabled with `sparsity_rate>0`, does threshold/top-k style magnitude selection and zeros others.
3. **Payload serialization per tensor** (`serialize_tensor_payload`):
   - Dense mode or CSR mode (CSR only when masking enabled and tensor rank is 2 or 4).
   - Optional value quantization (`None`, 16-bit cast, or int8+scale).
4. **Immediate deserialization** (`deserialize_tensor_payload`) to reconstruct tensors used by server aggregation.

### Required check matrix

| Processing type | Status in FedAvg active path | Notes |
|---|---|---|
| Delta computation | Implemented and used | Core aggregation input |
| Clipping | Missing | No gradient/update clipping |
| Normalization | Missing | No norm scaling or normalization pass |
| Sparsification | Partially implemented | Only if masking enabled and sparsity_rate>0 |
| Masking | Implemented and flag-dependent | Magnitude threshold + optional min-density top-k |
| Quantization | Implemented and flag-dependent | Applied in transport payload, then dequantized before aggregation |
| Low-rank decomposition | Missing | None found |
| Serialization | Implemented and used | Payload dict / CSR byte packet |
| Compression | Implemented and flag-dependent | CSR for 2D/4D when masking enabled |

---

## 6. Client-to-Server Payload and Transmission Logic

There is no real distributed transport layer; handoff is **in-memory within the same process**.

### Actual uploaded payload object

Per client, per parameter key, payload is created by `serialize_tensor_payload` and stored in `payload_dict[key]`.

Payload schema:

- **Dense mode:** dict with `mode='dense'`, `q_tensor`, `scale`, `bits`, `transport_dtype`, `orig_shape`.
- **CSR mode:** dict with `mode='csr'`, `packet` (bytes from `pack_csr`), `bits`, dtype/shape metadata, `nnz`, `dense_numel`.

But server aggregation does **not** consume `payload_dict` directly. Instead each payload is immediately deserialized on client loop side:

- `reconstructed_state_dict[key] = deserialize_tensor_payload(payload)`

Server then aggregates `reconstructed_state_dict` tensors (dense torch tensors).

### Transmission interpretation

- Handoff is simulated by encode/decode function calls in the same loop.
- No socket/HTTP/gRPC.
- Serialization exists as byte packet only for CSR mode; dense mode keeps tensors as torch objects in dict.

---

## 7. Upload Traffic Validation

Definitions in round logic:

- `client_upload_bytes`: sum of `payload_size` from each serialized parameter for one active client.
- `upload_traffic_round`: sum of `client_upload_bytes` over selected clients.
- `report['upload_traffic'] = upload_traffic_round`.
- `report['upload_traffic_per_client'] = mean(per_client_upload_bytes)`.

### Validation points

| Check | Result | Evidence-based explanation |
|---|---|---|
| `upload_traffic_per_client` matches payload | **PASS (estimated from serializer output sizes)** | Uses returned `payload_size` per tensor |
| `upload_traffic = upload_traffic_per_client * number_of_active_users` | **PASS (algebraically)** | Uses sum vs mean*count of selected clients |
| Uses active users, not total users | **PASS** | Sum only over sampled `selected` clients |
| Derived from real transport bytes | **PARTIAL** | Reflects simulated payload size; no external transport framing/network overhead |

---

## 8. Server-Side Reconstruction / Decoding

Reconstruction is performed by `deserialize_tensor_payload` in active path.

Implemented behaviors:

- Dense dequantization (`int8*scale` or fp16 cast back).
- CSR unpack (`unpack_csr`) + value dequantization + CSR-to-dense reconstruction (`decompress_csr`) + reshape.

Not implemented/active:

- Network deserialization stack (protocol buffers, sockets, etc.).
- Low-rank reconstruction.

Because encode/decode occurs in-process, this is logical reconstruction rather than true remote server decode.

---

## 9. Global Aggregation / Global Update Logic

### What server aggregates

- Aggregates **client deltas** (`local - global`), not full local model weights.
- Each client weighted by selected client data size proportion: `len(client_train_data[idx]) / sum(selected_sizes)`.

### Update rule implemented

For each parameter key:

`global_state[key] = global_state[key] + aggregated_delta[key]`

So server performs aggregation/update logic only; **no server optimizer (no SGD/Adam on server)**.

### Classification

- Aggregates deltas: **Yes**.
- Aggregates full weights: **No**.
- Reconstructs compressed updates before aggregation: **Yes (if applicable)**.
- Server optimization step: **No**.
- Direct overwrite with averaged model weights: **No** (it adds aggregated delta to existing global).

---

## 10. Server-to-Client Payload and Download Logic

No explicit outbound payload object is constructed.

What actually happens:

- Next round client receives `copy.deepcopy(global_model)` directly in memory.

No active outbound processing:

- No compression/quantization/sparsification/serialization/masking/low-rank pass before client download.

So download-side representation in execution is a full dense model copy in memory.

---

## 11. Download Traffic and Overall Traffic Validation

Implemented accounting:

- `model_size_bytes = tensor_dict_bytes(global_state)`.
- `download_traffic = model_size_bytes * args.n_client`.
- `report['overall_traffic'] = total_upload_traffic + total_download_traffic` (cumulative totals).

### Validation

| Check | Result | Explanation |
|---|---|---|
| Download uses actual outbound processed payload | **FAIL** | No outbound payload exists; uses dense model-size estimate |
| Multiplies by active users | **FAIL** | Multiplies by total clients `args.n_client`, not sampled clients |
| `overall_traffic = upload + download` | **PASS (cumulative form)** | Uses cumulative total upload + cumulative total download |
| Includes all components | **PARTIAL** | Excludes protocol overhead / any true network effects |

---

## 12. FLOPs Logging Validation

Round/global FLOPs variables:

- `round_flops = local_training_flops_round + aggregation_flops_round + evaluation_flops_round`
- `round_flops_compression = compression_flops_round + decompression_flops_round + serialization_flops_round + gs_flops_round`
- `total_flops += round_flops`
- `total_flops_compression += round_flops_compression`

In `proxy` mode:
- `local_training_flops_round` and `evaluation_flops_round` are layer-aware estimates collected from real ResNet18 forward executions via hooks.
- The estimates use runtime batch sizes and runtime module input/output shapes.
- Dense local training is counted as:
  - `training_flops = forward_flops + backward_flops + optimizer_flops`
  - `backward_flops = 2 * forward_flops`
  - `optimizer_flops = 2 * trainable_parameter_count * optimizer_step_count`

In `profiler` mode:
- Training and evaluation FLOPs are measured with `torch.profiler`, with residual-add FLOPs explicitly added so semantics match proxy-mode ResNet18 accounting.

`round_flops_compression` remains separate and includes:
- compression/decompression,
- serialization/deserialization,
- GS masking overhead.

GS masking and upload compression do **not** change dense local-training FLOPs in this repository because masking is applied to transmitted deltas, not to the dense forward/backward local model execution.

Dense ResNet18 FLOPs vary primarily with:
- selected clients,
- number of local samples,
- local epochs,
- last-batch sizes,
- evaluation coverage (train-loss eval, global test eval, client validation eval),
- model architecture and runtime tensor shapes.

### Inclusion audit for `round_flops` and `total_flops`

| FLOPs category | Included? | Notes |
|---|---|---|
| Local training FLOPs | **Yes** | Layer-aware hook collection in proxy mode; profiler-measured in profiler mode |
| Server aggregation math FLOPs | **Yes** | Analytical FedAvg aggregation estimator |
| Client compression FLOPs | **Yes (estimated)** | `estimate_payload_compression_flops` |
| Server decompression FLOPs | **Yes (estimated)** | `estimate_payload_decompression_flops` |
| Gauss-Southwell masking FLOPs | **Yes (estimated)** | `gs_flops` from masking path |
| Evaluation FLOPs | **Yes** | Layer-aware hook collection in proxy mode; profiler-measured in profiler mode |

---

## 13. Compression / Decompression FLOPs Validation

`total_flops_compression` is updated as:

- `round_flops_compression = compression_flops_round + decompression_flops_round + serialization_flops_round + gs_flops_round`
- `total_flops_compression += round_flops_compression`

Interpretation:

- Includes compression + decompression FLOPs.
- Includes serialization/deserialization FLOPs.
- Includes Gauss-Southwell masking FLOPs (`gs_flops`).

So this metric captures the full round compression pipeline FLOPs tracked in code.

---

## 14. Accuracy Logging Validation

`acc_servers_highest` behavior:

- Built each round from `acc_servers = [acc]`, where `acc` is global model accuracy on `test_loader`.
- Then `acc_servers_mean = mean([acc]) = acc`, `acc_servers_std = 0`, so:
  - `acc_servers_highest = acc`
  - `acc_servers_lowest = acc`

Implications:

- It is **current-round global test accuracy**, not historical best.
- Name `highest` is misleading (not max over rounds).
- Dataset/split: global `test_data` from `get_dataset` split by `train_frac` (default 0.8/0.2).

---

## 15. Experiment Configuration Validation

| Claimed standard setup | Validation result | Actual behavior in code |
|---|---|---|
| Dirichlet alpha = 0.5 | **Config default, not enforced** | `--dirichlet` default `0.5`, can be changed |
| train/validation split = 80/20 | **Partially true** | Global train/test from `train_frac` default 0.8; per-client train/val is hard-coded 80/20 (`val_split=0.2`) |
| optimizer = SGD | **PASS for local training** | `client_update` uses `torch.optim.SGD` |
| learning rate = 0.01 | **Config default, not enforced** | `--lr` default `0.01`, configurable |
| batch size = 128 | **Config default, not enforced** | `--batch_size` default `128`, configurable |

---

## 16. WandB Metrics Audit

### Metrics definitely logged in active path

| Metric/key | Logged to W&B | Notes |
|---|---|---|
| `sparsity`, `density` (per client) | Yes | Logged inside client loop with same step and `commit=False` |
| `upload_traffic`, `download_traffic`, `overall_traffic` | Yes | In round `report` |
| `round_flops`, `total_flops` | Yes | Compute-only FLOPs scope (training + aggregation + evaluation) |
| `total_flops_compression` | Yes | Cumulative `round_flops_compression` (compression + decompression + serialization + GS masking) |
| `compression_flops`, `decompression_flops`, `gs_flops` | Yes | Round-level estimates |
| `acc_servers_highest`, `acc_clients_highest`, etc. | Yes | Statistical forms; some naming mismatch vs semantics |
| `cos_*`, `training_loss_*`, delta/sparsity summaries | Yes | In `report` |

### Exists vs used vs traffic-affecting

| Feature | Exists in repo | Used in active FedAvg path | Logged in W&B | Affects payload/traffic |
|---|---|---|---|---|
| Sparse masking | Yes | Conditional | Yes (density/sparsity, gs_flops) | Yes |
| Quantization | Yes | Conditional | Indirect (size/FLOPs only) | Yes |
| CSR compression | Yes | Conditional (2D/4D + masking enabled) | Indirect | Yes |
| Dynamic CSR index quantization | Yes | Conditional | Indirect | Yes |
| Low-rank methods | No | No | No | No |

---

## 17. Faithfulness to the Intended Method Structure

Expected method structure implied by “FedAvg” name/repo organization vs implementation:

| Stage | Expected in FedAvg concept | Actual in code | Classification |
|---|---|---|---|
| Client sampling | Yes | Yes | Correctly implemented |
| Local SGD training | Yes | Yes | Correctly implemented |
| Client upload of model/update | Yes | Yes (simulated payload) | Correctly implemented |
| Server aggregation | Yes | Weighted delta aggregation then add to global | Correctly implemented (delta form) |
| Pure dense/no extra preprocessing | Typically yes | Optional masking+compression+quantization path present | Implemented differently (optional extensions) |
| Explicit method switch by `--method` | Usually expected in multi-method repos | Absent | Missing |
| True network transport pipeline | Not required in simulators | Not present | Correct for simulation, but traffic is estimated |

---

## 18. Mismatches, Risks, and Ambiguities

1. **Method flag mismatch:** `--method` does not control algorithm selection.
2. **Traffic mismatch on download:** uses total clients, not active clients; estimated from dense model size.
3. **FLOPs scope split:** `round_flops` / `total_flops` are compute-only, while compression/serialization/GS FLOPs are tracked in dedicated compression metrics.
4. **Accuracy naming mismatch:** `acc_servers_highest` equals current test accuracy, not best-over-time.
5. **In-process encode/decode:** payload realism is partial; no true transport overhead.
6. **Potential ambiguity in “FedAvg”:** core update is FedAvg-like, but optional sparsity/compression paths can materially alter behavior.

---

## 19. Final Checklist

| Validation item | Status |
|---|---|
| 1. Method activation and pipeline identified | **PASS** |
| 2. Post-local-training preprocessing documented | **PASS** |
| 3. `round_flops` meaning validated | **PARTIAL** |
| 3. `total_flops` meaning validated | **PARTIAL** |
| 4. Client sparsification status | **PARTIAL (flag-dependent)** |
| 4. Client quantization status | **PARTIAL (flag-dependent)** |
| 4. Client low-rank status | **FAIL (missing)** |
| 5. Flags/config linkage validated | **PASS** |
| 5. W&B linkage for processing validated | **PARTIAL** |
| 6. Client→server payload object identified | **PASS** |
| 6. Real serialization before handoff | **PARTIAL (CSR bytes only; in-memory simulation)** |
| 7. `upload_traffic_per_client` matches payload estimate | **PASS** |
| 7. `upload_traffic = per_client * active_users` | **PASS** |
| 8. Server decoding/reconstruction validated | **PASS** |
| 9. Global update logic validated | **PASS** |
| 10. Server→client processing (compression/etc.) | **FAIL (none active)** |
| 11. Download traffic reflects actual outbound payload | **FAIL** |
| 11. `overall_traffic = upload + download` | **PASS (cumulative)** |
| 12. `total_flops_compression` completeness | **PASS** |
| 13. `acc_servers_highest` semantic correctness | **FAIL (name mismatch)** |
| 14. Standard config assumptions validated | **PARTIAL** |
| 15. Intended-vs-actual method faithfulness section provided | **PASS** |
