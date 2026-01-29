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


_DTYPE_TO_CODE = {np.dtype("float32"): 1, np.dtype("float16"): 2, np.dtype("float64"): 3}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint cannot encode negative values")
    out = bytearray()
    v = int(value)
    while True:
        to_write = v & 0x7F
        v >>= 7
        if v:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)


def decode_uvarint_stream(data: bytes, count: int | None = None):
    values = []
    shift = 0
    current = 0
    idx = 0
    for idx, byte in enumerate(data):
        current |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        values.append(current)
        if count is not None and len(values) >= count:
            return values, idx + 1
        current = 0
        shift = 0
    if count is not None and len(values) != count:
        raise ValueError("Unexpected end of varint stream")
    return values, idx + 1 if data else 0


def _delta_encode_row_ptr(row_ptr: np.ndarray) -> np.ndarray:
    if row_ptr[0] != 0:
        raise ValueError("row_ptr must start at 0")
    deltas = np.diff(row_ptr, prepend=row_ptr[0]).astype(np.uint32)
    if np.any(deltas < 0):
        raise ValueError("row_ptr deltas must be non-negative")
    return deltas


def _delta_decode_row_ptr(deltas: np.ndarray) -> np.ndarray:
    return np.cumsum(deltas, dtype=np.uint32)


def _delta_encode_col_indices(col_indices: np.ndarray, row_ptr: np.ndarray) -> np.ndarray:
    deltas = []
    for row_idx in range(len(row_ptr) - 1):
        start = int(row_ptr[row_idx])
        end = int(row_ptr[row_idx + 1])
        row_cols = col_indices[start:end]
        if row_cols.size == 0:
            continue
        if np.any(row_cols[1:] < row_cols[:-1]):
            raise ValueError("col_indices must be nondecreasing within row")
        row_deltas = np.diff(row_cols, prepend=row_cols[0]).astype(np.uint32)
        if np.any(row_deltas < 0):
            raise ValueError("col_indices deltas must be non-negative")
        deltas.append(row_deltas)
    if deltas:
        return np.concatenate(deltas)
    return np.array([], dtype=np.uint32)


def _delta_decode_col_indices(deltas: np.ndarray, row_ptr: np.ndarray) -> np.ndarray:
    col_indices = []
    offset = 0
    for row_idx in range(len(row_ptr) - 1):
        start = int(row_ptr[row_idx])
        end = int(row_ptr[row_idx + 1])
        count = end - start
        if count == 0:
            continue
        row_deltas = deltas[offset : offset + count]
        if row_deltas.size != count:
            raise ValueError("Invalid col delta stream length")
        row_cols = np.cumsum(row_deltas, dtype=np.uint32)
        col_indices.append(row_cols)
        offset += count
    if offset != deltas.size:
        raise ValueError("Unused col delta entries")
    if col_indices:
        return np.concatenate(col_indices)
    return np.array([], dtype=np.uint32)


def pack_csr(csr: CSRMatrix) -> bytes:
    values = np.asarray(csr.values)
    dtype = values.dtype
    if dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype for CSR pack: {dtype}")
    n_rows, n_cols = csr.shape
    row_ptr = np.asarray(csr.row_ptr, dtype=np.uint32)
    col_indices = np.asarray(csr.col_indices, dtype=np.uint32)
    nnz = int(row_ptr[-1]) if row_ptr.size else 0
    if row_ptr.size != n_rows + 1:
        raise ValueError("row_ptr length mismatch")
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr must start at 0 and end at nnz")
    row_ptr_deltas = _delta_encode_row_ptr(row_ptr)
    col_deltas = _delta_encode_col_indices(col_indices, row_ptr)
    if col_deltas.size != nnz:
        raise ValueError("col delta size mismatch")
    row_ptr_bytes = b"".join(encode_uvarint(int(v)) for v in row_ptr_deltas)
    col_bytes = b"".join(encode_uvarint(int(v)) for v in col_deltas)
    values_bytes = values.tobytes()
    header = (
        int(n_rows).to_bytes(4, "little")
        + int(n_cols).to_bytes(4, "little")
        + bytes([_DTYPE_TO_CODE[dtype]])
        + int(nnz).to_bytes(4, "little")
        + int(len(values_bytes)).to_bytes(4, "little")
        + int(len(row_ptr_bytes)).to_bytes(4, "little")
        + int(len(col_bytes)).to_bytes(4, "little")
    )
    return header + values_bytes + row_ptr_bytes + col_bytes


def unpack_csr(data: bytes) -> CSRMatrix:
    if len(data) < 25:
        raise ValueError("Packet too short for CSR header")
    n_rows = int.from_bytes(data[0:4], "little")
    n_cols = int.from_bytes(data[4:8], "little")
    dtype_code = data[8]
    nnz = int.from_bytes(data[9:13], "little")
    values_nbytes = int.from_bytes(data[13:17], "little")
    row_ptr_bytes_len = int.from_bytes(data[17:21], "little")
    col_bytes_len = int.from_bytes(data[21:25], "little")
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError("Unknown dtype code")
    offset = 25
    values_end = offset + values_nbytes
    values = np.frombuffer(data[offset:values_end], dtype=dtype)
    offset = values_end
    row_ptr_bytes = data[offset : offset + row_ptr_bytes_len]
    offset += row_ptr_bytes_len
    col_bytes = data[offset : offset + col_bytes_len]
    row_ptr_deltas, used = decode_uvarint_stream(row_ptr_bytes, count=n_rows + 1)
    if used != row_ptr_bytes_len:
        raise ValueError("row_ptr bytes length mismatch")
    row_ptr = _delta_decode_row_ptr(np.array(row_ptr_deltas, dtype=np.uint32))
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr does not match nnz")
    col_deltas, used = decode_uvarint_stream(col_bytes, count=nnz)
    if used != col_bytes_len:
        raise ValueError("col bytes length mismatch")
    col_indices = _delta_decode_col_indices(np.array(col_deltas, dtype=np.uint32), row_ptr)
    return CSRMatrix(
        values=values,
        col_indices=col_indices.astype(np.uint32, copy=False),
        row_ptr=row_ptr.astype(np.uint32, copy=False),
        shape=(n_rows, n_cols),
    )


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
