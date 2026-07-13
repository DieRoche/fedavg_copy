import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
torch = pytest.importorskip("torch")

from main import (
    select_automatic_compression_format,
    serialize_tensor_payload,
    deserialize_tensor_payload,
    compressed_quantized_tensor_bytes,
)


def sparse(shape, nnz):
    t = torch.zeros(*shape, dtype=torch.float32)
    t.reshape(-1)[:nnz] = torch.arange(1, nnz + 1, dtype=torch.float32)
    return t


def test_high_sparsity_2d_selects_csr():
    t = sparse((100, 100), 100)
    fmt, stats = select_automatic_compression_format(t, min_tensor_size=1024)
    assert fmt == "CSR"
    assert stats["sparsity"] >= 0.90


def test_high_sparsity_4d_selects_csr():
    t = sparse((16, 4, 4, 4), 20)
    fmt, _ = select_automatic_compression_format(t, min_tensor_size=1)
    assert fmt == "CSR"


def test_one_dimensional_high_sparsity_not_csr():
    t = sparse((2000,), 20)
    fmt, _ = select_automatic_compression_format(t, min_tensor_size=1)
    assert fmt != "CSR"


def test_low_sparsity_prefers_bitmask_when_smaller_than_dense():
    t = sparse((10000,), 8000)  # 20% sparse; mask+values is smaller than dense by estimator
    fmt, _ = select_automatic_compression_format(t, bitmask_threshold=0.85, min_tensor_size=1)
    assert fmt == "bitmask_values"


def test_dense_when_bitmask_would_not_reduce_size():
    t = torch.ones(1024, dtype=torch.float32)
    fmt, _ = select_automatic_compression_format(t, min_tensor_size=1)
    assert fmt == "dense"


def test_forced_dense_cases():
    assert select_automatic_compression_format(torch.ones(10), min_tensor_size=1024)[0] == "dense"
    assert select_automatic_compression_format(torch.tensor(1.0), min_tensor_size=0)[0] == "dense"
    assert select_automatic_compression_format(torch.empty(0), min_tensor_size=0)[0] == "dense"
    assert select_automatic_compression_format(torch.ones(2048, dtype=torch.int64), min_tensor_size=1)[0] == "dense"


@pytest.mark.parametrize("tensor,expected", [
    (sparse((100, 100), 100), "csr"),
    (sparse((10000,), 8000), "bitmask_values"),
    (torch.ones(1024), "dense"),
])
def test_automatic_payload_reconstructs(tensor, expected):
    payload, size = serialize_tensor_payload(
        tensor, None, True, False, "Automatic", 0.90, 0.85, 1
    )
    assert payload["mode"] == expected
    assert payload["requested_mode"] == "Automatic"
    assert torch.equal(deserialize_tensor_payload(payload), tensor)
    assert size > 0


def test_quantized_automatic_matches_selected_explicit_behavior():
    t = sparse((100, 100), 100)
    auto_payload, _ = serialize_tensor_payload(t, 8, True, True, "Automatic", 0.90, 0.85, 1)
    explicit_payload, _ = serialize_tensor_payload(t, 8, True, True, "CSR")
    assert auto_payload["mode"] == explicit_payload["mode"] == "csr"
    assert torch.allclose(deserialize_tensor_payload(auto_payload), deserialize_tensor_payload(explicit_payload))


def test_state_dict_bytes_equals_selected_payload_sum():
    class Args:
        enable_sparse_masking = True
        sparsity_compression = "Automatic"
        quantization_bits = None
        dynamic_quantization = False
        automatic_csr_sparsity_threshold = 0.90
        automatic_bitmask_sparsity_threshold = 0.85
        automatic_min_tensor_size = 1
    sd = {"a": sparse((100, 100), 100), "b": torch.ones(1024)}
    estimated = sum(compressed_quantized_tensor_bytes(v, "Automatic", None, False, Args) for v in sd.values())
    actual = sum(serialize_tensor_payload(v, None, True, False, "Automatic", .9, .85, 1)[1] for v in sd.values())
    assert estimated == actual
