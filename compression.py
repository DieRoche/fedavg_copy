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


_VAL_BITS_TO_DTYPE = {
    8: np.dtype("int8"),
    16: np.dtype("float16"),
    32: np.dtype("float32"),
    64: np.dtype("float64"),
}
_DTYPE_TO_VAL_BITS = {v: k for k, v in _VAL_BITS_TO_DTYPE.items()}


def _select_index_dtype(
    col_indices: np.ndarray,
    row_ptr: np.ndarray,
    dynamic_quantization: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    col_arr = np.asarray(col_indices, dtype=np.uint32)
    row_arr = np.asarray(row_ptr, dtype=np.uint32)
    if not dynamic_quantization:
        return col_arr, row_arr, 32
    col_max = int(col_arr.max(initial=0))
    row_max = int(row_arr.max(initial=0))
    if col_max <= 65535 and row_max <= 65535:
        return col_arr.astype(np.uint16), row_arr.astype(np.uint16), 16
    return col_arr, row_arr, 32


def pack_csr(csr: CSRMatrix, dynamic_quantization: bool = False, scale: float | None = None) -> bytes:
    values = np.asarray(csr.values)
    dtype = values.dtype
    val_bits = _DTYPE_TO_VAL_BITS.get(dtype)
    if val_bits is None:
        raise ValueError(f"Unsupported dtype for CSR pack: {dtype}")
    has_scale = bool(val_bits == 8 and scale is not None)
    if not has_scale:
        scale = None
    n_rows, n_cols = csr.shape
    row_ptr = np.asarray(csr.row_ptr, dtype=np.uint32)
    col_indices = np.asarray(csr.col_indices, dtype=np.uint32)
    nnz = int(row_ptr[-1]) if row_ptr.size else 0
    if row_ptr.size != n_rows + 1:
        raise ValueError("row_ptr length mismatch")
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr must start at 0 and end at nnz")
    col_enc, row_enc, idx_bits = _select_index_dtype(
        col_indices,
        row_ptr,
        dynamic_quantization=dynamic_quantization,
    )
    row_ptr_bytes = row_enc.tobytes()
    col_bytes = col_enc.tobytes()
    values_bytes = values.tobytes()
    scale_value = float(scale) if scale is not None else 0.0
    header = (
        int(n_rows).to_bytes(4, "little")
        + int(n_cols).to_bytes(4, "little")
        + int(nnz).to_bytes(4, "little")
        + bytes([int(val_bits)])
        + bytes([int(idx_bits)])
        + bytes([1 if has_scale else 0])
        + int(len(values_bytes)).to_bytes(4, "little")
        + int(len(row_ptr_bytes)).to_bytes(4, "little")
        + int(len(col_bytes)).to_bytes(4, "little")
        + np.float32(scale_value).tobytes()
    )
    return header + values_bytes + row_ptr_bytes + col_bytes


def unpack_csr(data: bytes) -> tuple[CSRMatrix, dict]:
    if len(data) < 31:
        raise ValueError("Packet too short for CSR header")
    n_rows = int.from_bytes(data[0:4], "little")
    n_cols = int.from_bytes(data[4:8], "little")
    nnz = int.from_bytes(data[8:12], "little")
    val_bits = int(data[12])
    idx_bits = int(data[13])
    has_scale = bool(data[14])
    values_nbytes = int.from_bytes(data[15:19], "little")
    row_ptr_bytes_len = int.from_bytes(data[19:23], "little")
    col_bytes_len = int.from_bytes(data[23:27], "little")
    scale = float(np.frombuffer(data[27:31], dtype=np.float32, count=1)[0])
    value_dtype = _VAL_BITS_TO_DTYPE.get(val_bits)
    if value_dtype is None:
        raise ValueError("Unknown val_bits in CSR packet")
    if has_scale and val_bits != 8:
        raise ValueError("Scale can only be present for int8 payloads")
    if has_scale and scale <= 0.0:
        raise ValueError("Scaled int8 payload requires positive scale")
    if not has_scale:
        scale = 0.0
    index_dtype = {16: np.uint16, 32: np.uint32}.get(idx_bits)
    if index_dtype is None:
        raise ValueError("Unknown idx_bits in CSR packet")
    offset = 31
    values_end = offset + values_nbytes
    values = np.frombuffer(data[offset:values_end], dtype=value_dtype)
    offset = values_end
    row_ptr_bytes = data[offset : offset + row_ptr_bytes_len]
    offset += row_ptr_bytes_len
    col_bytes = data[offset : offset + col_bytes_len]
    if len(data) != offset + col_bytes_len:
        raise ValueError("Packet length mismatch")
    row_ptr = np.frombuffer(row_ptr_bytes, dtype=index_dtype).astype(np.uint32, copy=False)
    col_indices = np.frombuffer(col_bytes, dtype=index_dtype).astype(np.uint32, copy=False)
    if row_ptr.size != n_rows + 1:
        raise ValueError("row_ptr length mismatch")
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr does not match nnz")
    if col_indices.size != nnz:
        raise ValueError("col_indices length mismatch")
    csr = CSRMatrix(
        values=values,
        col_indices=col_indices,
        row_ptr=row_ptr.astype(np.uint32, copy=False),
        shape=(n_rows, n_cols),
    )
    return csr, {
        "val_bits": val_bits,
        "idx_bits": idx_bits,
        "scale": scale,
        "has_scale": has_scale,
    }


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
    "encode_uvarint",
    "decode_uvarint_stream",
    "pack_csr",
    "unpack_csr",
]
