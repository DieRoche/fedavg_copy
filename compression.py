from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CSRMatrix:
    values: np.ndarray
    col_indices: np.ndarray
    row_ptr: np.ndarray
    shape: Tuple[int, int]


@dataclass(frozen=True)
class CSCMatrix:
    values: np.ndarray
    row_indices: np.ndarray
    col_ptr: np.ndarray
    shape: Tuple[int, int]


@dataclass(frozen=True)
class BSRMatrix:
    data: np.ndarray
    col_indices: np.ndarray
    row_ptr: np.ndarray
    block_size: Tuple[int, int]
    shape: Tuple[int, int]


def _to_numpy(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def compress_csr(matrix: np.ndarray | torch.Tensor) -> CSRMatrix:
    dense = _to_numpy(matrix)
    if dense.ndim != 2:
        raise ValueError("CSR compression expects a 2D matrix.")

    rows, _ = dense.shape
    values = []
    col_indices = []
    row_ptr = [0]

    for row_idx in range(rows):
        row = dense[row_idx]
        nz_cols = np.nonzero(row)[0]
        values.extend(row[nz_cols].tolist())
        col_indices.extend(nz_cols.tolist())
        row_ptr.append(len(values))

    return CSRMatrix(
        values=np.array(values, dtype=dense.dtype),
        col_indices=np.array(col_indices, dtype=np.int32),
        row_ptr=np.array(row_ptr, dtype=np.int32),
        shape=dense.shape,
    )


def decompress_csr(csr: CSRMatrix) -> np.ndarray:
    dense = np.zeros(csr.shape, dtype=csr.values.dtype)
    for row_idx in range(csr.shape[0]):
        start = csr.row_ptr[row_idx]
        end = csr.row_ptr[row_idx + 1]
        cols = csr.col_indices[start:end]
        dense[row_idx, cols] = csr.values[start:end]
    return dense


def compress_csc(matrix: np.ndarray | torch.Tensor) -> CSCMatrix:
    dense = _to_numpy(matrix)
    if dense.ndim != 2:
        raise ValueError("CSC compression expects a 2D matrix.")

    _, cols = dense.shape
    values = []
    row_indices = []
    col_ptr = [0]

    for col_idx in range(cols):
        col = dense[:, col_idx]
        nz_rows = np.nonzero(col)[0]
        values.extend(col[nz_rows].tolist())
        row_indices.extend(nz_rows.tolist())
        col_ptr.append(len(values))

    return CSCMatrix(
        values=np.array(values, dtype=dense.dtype),
        row_indices=np.array(row_indices, dtype=np.int64),
        col_ptr=np.array(col_ptr, dtype=np.int64),
        shape=dense.shape,
    )


def decompress_csc(csc: CSCMatrix) -> np.ndarray:
    dense = np.zeros(csc.shape, dtype=csc.values.dtype)
    for col_idx in range(csc.shape[1]):
        start = csc.col_ptr[col_idx]
        end = csc.col_ptr[col_idx + 1]
        rows = csc.row_indices[start:end]
        dense[rows, col_idx] = csc.values[start:end]
    return dense


def compress_bsr(
    matrix: np.ndarray | torch.Tensor, block_size: Tuple[int, int]
) -> BSRMatrix:
    dense = _to_numpy(matrix)
    if dense.ndim != 2:
        raise ValueError("BSR compression expects a 2D matrix.")

    rows, cols = dense.shape
    block_rows, block_cols = block_size
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("Matrix shape must be divisible by block size.")

    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols

    data = []
    col_indices = []
    row_ptr = [0]

    for block_row in range(n_block_rows):
        for block_col in range(n_block_cols):
            row_start = block_row * block_rows
            row_end = row_start + block_rows
            col_start = block_col * block_cols
            col_end = col_start + block_cols
            block = dense[row_start:row_end, col_start:col_end]
            if np.any(block != 0):
                data.append(block.copy())
                col_indices.append(block_col)
        row_ptr.append(len(data))

    if data:
        data_array = np.stack(data, axis=0)
    else:
        data_array = np.empty((0, block_rows, block_cols), dtype=dense.dtype)

    return BSRMatrix(
        data=data_array,
        col_indices=np.array(col_indices, dtype=np.int64),
        row_ptr=np.array(row_ptr, dtype=np.int64),
        block_size=block_size,
        shape=dense.shape,
    )


def decompress_bsr(bsr: BSRMatrix) -> np.ndarray:
    rows, cols = bsr.shape
    block_rows, block_cols = bsr.block_size
    dense = np.zeros((rows, cols), dtype=bsr.data.dtype)
    n_block_rows = rows // block_rows

    for block_row in range(n_block_rows):
        start = bsr.row_ptr[block_row]
        end = bsr.row_ptr[block_row + 1]
        for idx in range(start, end):
            block_col = bsr.col_indices[idx]
            row_start = block_row * block_rows
            row_end = row_start + block_rows
            col_start = block_col * block_cols
            col_end = col_start + block_cols
            dense[row_start:row_end, col_start:col_end] = bsr.data[idx]
    return dense


__all__ = [
    "CSRMatrix",
    "CSCMatrix",
    "BSRMatrix",
    "compress_csr",
    "compress_csc",
    "compress_bsr",
    "decompress_csr",
    "decompress_csc",
    "decompress_bsr",
]
