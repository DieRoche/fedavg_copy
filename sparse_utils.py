from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

SPARSE_INDEX_DTYPE = np.int64


def to_2d_for_sparse_sizing(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    dense = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if dense.ndim == 0:
        return dense.reshape(1, 1)
    if dense.ndim == 1:
        return dense.reshape(-1, 1)
    if dense.ndim == 2:
        return dense
    return dense.reshape(dense.shape[0], -1)


def _csr_bytes(dense: np.ndarray, index_dtype: np.dtype) -> int:
    if dense.size == 0:
        return 0
    nnz = int(np.count_nonzero(dense))
    data_bytes = nnz * dense.dtype.itemsize
    index_bytes = nnz * np.dtype(index_dtype).itemsize
    indptr_bytes = (dense.shape[0] + 1) * np.dtype(index_dtype).itemsize
    return data_bytes + index_bytes + indptr_bytes


def _csc_bytes(dense: np.ndarray, index_dtype: np.dtype) -> int:
    if dense.size == 0:
        return 0
    nnz = int(np.count_nonzero(dense))
    data_bytes = nnz * dense.dtype.itemsize
    index_bytes = nnz * np.dtype(index_dtype).itemsize
    indptr_bytes = (dense.shape[1] + 1) * np.dtype(index_dtype).itemsize
    return data_bytes + index_bytes + indptr_bytes


def _bsr_bytes(
    dense: np.ndarray, block_size: Tuple[int, int], index_dtype: np.dtype
) -> int:
    if dense.size == 0:
        return 0
    block_rows, block_cols = block_size
    rows, cols = dense.shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("Matrix shape must be divisible by block size.")
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols
    blocks = dense.reshape(n_block_rows, block_rows, n_block_cols, block_cols)
    block_mask = np.any(blocks != 0, axis=(1, 3))
    nnzb = int(np.count_nonzero(block_mask))
    data_bytes = nnzb * block_rows * block_cols * dense.dtype.itemsize
    index_bytes = nnzb * np.dtype(index_dtype).itemsize
    indptr_bytes = (n_block_rows + 1) * np.dtype(index_dtype).itemsize
    return data_bytes + index_bytes + indptr_bytes


def sparse_tensor_bytes(
    tensor: torch.Tensor | np.ndarray,
    compression_type: str,
    *,
    index_dtype: np.dtype = SPARSE_INDEX_DTYPE,
    block_size: Tuple[int, int] = (1, 1),
) -> int:
    dense = to_2d_for_sparse_sizing(tensor)
    if compression_type == "CSR":
        return _csr_bytes(dense, index_dtype)
    if compression_type == "CSC":
        return _csc_bytes(dense, index_dtype)
    if compression_type == "BSR":
        return _bsr_bytes(dense, block_size, index_dtype)
    raise ValueError(f"Unknown compression type: {compression_type}")


__all__ = [
    "SPARSE_INDEX_DTYPE",
    "sparse_tensor_bytes",
    "to_2d_for_sparse_sizing",
]
