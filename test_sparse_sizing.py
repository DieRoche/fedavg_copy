import numpy as np
import torch

from sparse_utils import sparse_tensor_bytes


def test_csc_pointer_bytes_depend_on_shape():
    wide = torch.zeros((1, 1024), dtype=torch.float32)
    tall = torch.zeros((1024, 1), dtype=torch.float32)
    wide_bytes = sparse_tensor_bytes(wide, "CSC")
    tall_bytes = sparse_tensor_bytes(tall, "CSC")
    assert wide_bytes != tall_bytes


def test_scalar_tensor_sizing():
    scalar = torch.tensor(5.0, dtype=torch.float32)
    bytes_csr = sparse_tensor_bytes(scalar, "CSR")
    expected = 4 + 8 + 16
    assert bytes_csr == expected


def test_csr_csc_bytes_match_formula():
    dense = torch.tensor([[1, 0, 2], [0, 0, 3]], dtype=torch.float32)
    csr_bytes = sparse_tensor_bytes(dense, "CSR", index_dtype=np.int32)
    csc_bytes = sparse_tensor_bytes(dense, "CSC", index_dtype=np.int32)
    assert csr_bytes == 36
    assert csc_bytes == 40
