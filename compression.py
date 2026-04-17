from dataclasses import dataclass
from heapq import heapify, heappop, heappush
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

_COL_CODEC_RAW_DELTA = 0
_COL_CODEC_DELTA_HUFFMAN = 1
_PTR_CODEC_RAW_DELTA = 0
_PTR_CODEC_DELTA_RLE = 1


def entropy_viability_check(array: np.ndarray, dtype: np.dtype) -> dict:
    arr = np.asarray(array, dtype=dtype)
    n = int(arr.size)
    raw_size = float(n * np.dtype(dtype).itemsize)
    if n == 0:
        return {
            "viable": False,
            "compression_ratio": 1.0,
            "estimated_compressed_size": 0.0,
            "raw_size": raw_size,
            "recommendation": "Not viable: empty delta array.",
            "entropy_bits": 0.0,
            "unique_symbol_count": 0,
        }

    _, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(np.float64) / n
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    unique_symbol_count = int(counts.size)
    theoretical = (entropy_bits * n) / 8.0
    estimated = (theoretical * 1.15) + (2.0 + (unique_symbol_count * 3.0))
    ratio = (raw_size / estimated) if estimated > 0.0 else 1.0
    viable = bool(estimated < raw_size)
    recommendation = (
        f"Viable: estimated {estimated:.2f}B < raw {raw_size:.2f}B."
        if viable
        else f"Not viable: estimated {estimated:.2f}B >= raw {raw_size:.2f}B."
    )
    return {
        "viable": viable,
        "compression_ratio": float(ratio),
        "estimated_compressed_size": float(estimated),
        "raw_size": raw_size,
        "recommendation": recommendation,
        "entropy_bits": entropy_bits,
        "unique_symbol_count": unique_symbol_count,
    }


def delta_encode_col_indices_row_local(col_indices: np.ndarray, row_ptr: np.ndarray) -> np.ndarray:
    cols = np.asarray(col_indices, dtype=np.int32)
    ptr = np.asarray(row_ptr, dtype=np.int32)
    encoded = cols.copy()
    for row_idx in range(max(0, ptr.size - 1)):
        start = int(ptr[row_idx])
        end = int(ptr[row_idx + 1])
        if end - start <= 1:
            continue
        encoded[start + 1 : end] = cols[start + 1 : end] - cols[start : end - 1]
    return encoded


def delta_decode_col_indices_row_local(col_delta: np.ndarray, row_ptr: np.ndarray) -> np.ndarray:
    delta = np.asarray(col_delta, dtype=np.int32)
    ptr = np.asarray(row_ptr, dtype=np.int32)
    decoded = delta.copy()
    for row_idx in range(max(0, ptr.size - 1)):
        start = int(ptr[row_idx])
        end = int(ptr[row_idx + 1])
        if end - start <= 1:
            continue
        decoded[start:end] = np.cumsum(delta[start:end], dtype=np.int64).astype(np.int32)
    return decoded


def delta_encode_row_ptr_global(row_ptr: np.ndarray) -> np.ndarray:
    ptr = np.asarray(row_ptr, dtype=np.int32)
    if ptr.size == 0:
        return ptr.copy()
    delta = np.empty_like(ptr)
    delta[0] = ptr[0]
    if ptr.size > 1:
        delta[1:] = ptr[1:] - ptr[:-1]
    return delta


def delta_decode_row_ptr_global(row_delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(row_delta, dtype=np.int32)
    return np.cumsum(delta, dtype=np.int64).astype(np.int32)


def _canonical_codes_from_lengths(length_items: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    sorted_items = sorted(length_items, key=lambda item: (item[1], item[0]))
    code = 0
    prev_len = 0
    canonical = {}
    for symbol, length in sorted_items:
        if length > prev_len:
            code <<= (length - prev_len)
            prev_len = length
        canonical[symbol] = (code, length)
        code += 1
    return canonical


def _huffman_build_code_lengths(symbols: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(symbols, return_counts=True)
    if values.size == 1:
        return {int(values[0]): 1}
    heap = [(int(count), idx, int(symbol), None, None) for idx, (symbol, count) in enumerate(zip(values, counts))]
    heapify(heap)
    next_idx = len(heap)
    while len(heap) > 1:
        c1, i1, s1, l1, r1 = heappop(heap)
        c2, i2, s2, l2, r2 = heappop(heap)
        node = (c1 + c2, next_idx, None, (s1, l1, r1), (s2, l2, r2))
        next_idx += 1
        heappush(heap, node)
    _, _, root_symbol, left, right = heap[0]
    lengths = {}

    def _walk(symbol, lnode, rnode, depth):
        if symbol is not None:
            lengths[int(symbol)] = max(1, depth)
            return
        ls, ll, lr = lnode
        rs, rl, rr = rnode
        _walk(ls, ll, lr, depth + 1)
        _walk(rs, rl, rr, depth + 1)

    _walk(root_symbol, left, right, 0)
    return lengths


def _huffman_encode_int_array(symbols: np.ndarray) -> tuple[bytes, bytes, int]:
    arr = np.asarray(symbols, dtype=np.int32)
    lengths = _huffman_build_code_lengths(arr)
    length_items = [(sym, ln) for sym, ln in lengths.items()]
    if len(length_items) > 65535:
        raise ValueError("Huffman symbol cardinality exceeds uint16 table capacity.")
    canonical = _canonical_codes_from_lengths(length_items)

    table = bytearray()
    table += int(len(length_items)).to_bytes(2, "little")
    for sym, ln in sorted(length_items, key=lambda item: item[0]):
        table += np.int16(sym).tobytes()
        table += np.uint8(ln).tobytes()

    bitstream = []
    for sym in arr.tolist():
        code, ln = canonical[int(sym)]
        bitstream.append(format(code, f"0{ln}b"))
    bits = "".join(bitstream)
    pad = (8 - (len(bits) % 8)) % 8
    if pad:
        bits += "0" * pad
    packed = bytearray([pad])
    for idx in range(0, len(bits), 8):
        packed.append(int(bits[idx : idx + 8], 2))
    return bytes(table), bytes(packed), int(len(length_items))


def _huffman_decode_int_array(table_bytes: bytes, bitstream_bytes: bytes, expected_len: int) -> np.ndarray:
    if len(table_bytes) < 2:
        raise ValueError("Invalid Huffman table bytes.")
    symbol_count = int.from_bytes(table_bytes[0:2], "little")
    if len(table_bytes) != 2 + (symbol_count * 3):
        raise ValueError("Huffman table length mismatch.")
    lengths = []
    offset = 2
    for _ in range(symbol_count):
        sym = int(np.frombuffer(table_bytes[offset : offset + 2], dtype=np.int16, count=1)[0])
        ln = int(np.frombuffer(table_bytes[offset + 2 : offset + 3], dtype=np.uint8, count=1)[0])
        lengths.append((sym, ln))
        offset += 3
    canonical = _canonical_codes_from_lengths(lengths)
    decode_map = {(code, ln): sym for sym, (code, ln) in canonical.items()}
    if not bitstream_bytes:
        raise ValueError("Missing Huffman bitstream bytes.")
    pad = int(bitstream_bytes[0])
    payload = bitstream_bytes[1:]
    bits = "".join(format(byte, "08b") for byte in payload)
    if pad:
        bits = bits[:-pad]
    out = []
    code = 0
    ln = 0
    for bit in bits:
        code = (code << 1) | (1 if bit == "1" else 0)
        ln += 1
        sym = decode_map.get((code, ln))
        if sym is not None:
            out.append(sym)
            code = 0
            ln = 0
            if len(out) == expected_len:
                break
    if len(out) != expected_len:
        raise ValueError("Decoded Huffman symbol count mismatch.")
    return np.asarray(out, dtype=np.int32)


def _rle_encode_int_array(arr: np.ndarray) -> tuple[bytes, int]:
    data = np.asarray(arr, dtype=np.int32)
    if data.size == 0:
        return (0).to_bytes(4, "little"), 0
    pairs = []
    run_value = int(data[0])
    run_len = 1
    for value in data[1:].tolist():
        if int(value) == run_value and run_len < 65535:
            run_len += 1
            continue
        pairs.append((run_len, run_value))
        run_value = int(value)
        run_len = 1
    pairs.append((run_len, run_value))
    encoded = bytearray()
    encoded += int(len(pairs)).to_bytes(4, "little")
    for length, value in pairs:
        encoded += np.uint16(length).tobytes()
        encoded += np.int16(value).tobytes()
    return bytes(encoded), int(len(pairs))


def _rle_decode_int_array(encoded: bytes, expected_len: int) -> np.ndarray:
    if len(encoded) < 4:
        raise ValueError("Invalid RLE payload.")
    pair_count = int.from_bytes(encoded[0:4], "little")
    expected_bytes = 4 + (pair_count * 4)
    if len(encoded) != expected_bytes:
        raise ValueError("RLE payload length mismatch.")
    offset = 4
    out = []
    for _ in range(pair_count):
        run_len = int(np.frombuffer(encoded[offset : offset + 2], dtype=np.uint16, count=1)[0])
        value = int(np.frombuffer(encoded[offset + 2 : offset + 4], dtype=np.int16, count=1)[0])
        out.extend([value] * run_len)
        offset += 4
    if len(out) != expected_len:
        raise ValueError("Decoded RLE symbol count mismatch.")
    return np.asarray(out, dtype=np.int32)


def _select_index_dtype(
    col_delta: np.ndarray,
    row_delta: np.ndarray,
    dynamic_quantization: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    col_arr = np.asarray(col_delta, dtype=np.int32)
    row_arr = np.asarray(row_delta, dtype=np.int32)
    if not dynamic_quantization:
        return col_arr, row_arr, 32
    col_min = int(col_arr.min(initial=0))
    col_max = int(col_arr.max(initial=0))
    row_min = int(row_arr.min(initial=0))
    row_max = int(row_arr.max(initial=0))
    if col_min >= -32768 and col_max <= 32767 and row_min >= -32768 and row_max <= 32767:
        return col_arr.astype(np.int16), row_arr.astype(np.int16), 16
    return col_arr, row_arr, 32


def pack_csr(
    csr: CSRMatrix,
    dynamic_quantization: bool = False,
    scale: float | None = None,
    return_stats: bool = False,
) -> bytes | tuple[bytes, dict]:
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
    col_indices = np.asarray(csr.col_indices, dtype=np.int32)
    nnz = int(row_ptr[-1]) if row_ptr.size else 0
    if row_ptr.size != n_rows + 1:
        raise ValueError("row_ptr length mismatch")
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr must start at 0 and end at nnz")
    col_delta = delta_encode_col_indices_row_local(col_indices, row_ptr)
    row_delta = delta_encode_row_ptr_global(row_ptr)
    col_ent = entropy_viability_check(col_delta, np.int16 if dynamic_quantization else np.int32)
    ptr_ent = entropy_viability_check(row_delta, np.int16 if dynamic_quantization else np.int32)

    col_enc, row_enc, idx_bits = _select_index_dtype(
        col_delta,
        row_delta,
        dynamic_quantization=dynamic_quantization,
    )
    index_dtype = np.int16 if idx_bits == 16 else np.int32
    col_delta_t = col_delta.astype(index_dtype, copy=False)
    row_delta_t = row_delta.astype(index_dtype, copy=False)

    col_codec = _COL_CODEC_RAW_DELTA
    ptr_codec = _PTR_CODEC_RAW_DELTA
    huffman_table = b""
    col_payload = col_delta_t.tobytes()
    col_unique_symbols = int(np.unique(col_delta_t).size)
    col_symbol_count_fits = col_unique_symbols <= 65535
    if col_ent["viable"] and col_symbol_count_fits and int(col_delta_t.min(initial=0)) >= -32768 and int(col_delta_t.max(initial=0)) <= 32767:
        huffman_table, huffman_payload, _ = _huffman_encode_int_array(col_delta_t.astype(np.int32))
        if len(huffman_table) + len(huffman_payload) < col_delta_t.nbytes:
            col_codec = _COL_CODEC_DELTA_HUFFMAN
            col_payload = huffman_payload
        else:
            huffman_table = b""

    if ptr_ent["viable"] and int(row_delta_t.min(initial=0)) >= -32768 and int(row_delta_t.max(initial=0)) <= 32767:
        rle_payload, _ = _rle_encode_int_array(row_delta_t.astype(np.int32))
        if len(rle_payload) < row_delta_t.nbytes:
            ptr_codec = _PTR_CODEC_DELTA_RLE
            row_ptr_bytes = rle_payload
        else:
            row_ptr_bytes = row_delta_t.tobytes()
    else:
        row_ptr_bytes = row_delta_t.tobytes()

    values_bytes = values.tobytes()
    scale_value = float(scale) if scale is not None else 0.0
    header_len = 45
    header = (
        int(n_rows).to_bytes(4, "little")
        + int(n_cols).to_bytes(4, "little")
        + int(nnz).to_bytes(4, "little")
        + bytes([int(val_bits)])
        + bytes([int(idx_bits)])
        + bytes([1 if has_scale else 0])
        + bytes([int(col_codec)])
        + bytes([int(ptr_codec)])
        + int(nnz).to_bytes(4, "little")
        + int(n_rows + 1).to_bytes(4, "little")
        + int(len(huffman_table)).to_bytes(4, "little")
        + int(len(values_bytes)).to_bytes(4, "little")
        + int(len(row_ptr_bytes)).to_bytes(4, "little")
        + int(len(col_payload)).to_bytes(4, "little")
        + np.float32(scale_value).tobytes()
    )
    packet = header + huffman_table + values_bytes + col_payload + row_ptr_bytes
    if not return_stats:
        return packet
    stats = {
        "header_bytes": header_len,
        "huffman_table_bytes": len(huffman_table),
        "values_payload_bytes": len(values_bytes),
        "col_payload_bytes": len(col_payload),
        "row_ptr_payload_bytes": len(row_ptr_bytes),
        "col_codec": col_codec,
        "ptr_codec": ptr_codec,
        "col_entropy_viable": bool(col_ent["viable"]),
        "ptr_entropy_viable": bool(ptr_ent["viable"]),
        "col_entropy_recommendation": col_ent["recommendation"],
        "ptr_entropy_recommendation": ptr_ent["recommendation"],
        "col_unique_symbols": col_unique_symbols,
        "col_symbol_count_fits_huffman_header": bool(col_symbol_count_fits),
        "raw_col_delta_bytes": int(col_delta_t.nbytes),
        "raw_row_delta_bytes": int(row_delta_t.nbytes),
        "compression_flops_codec": int(col_delta.size + row_delta.size + (4 * (col_delta.size + row_delta.size))),
    }
    return packet, stats


def unpack_csr(data: bytes) -> tuple[CSRMatrix, dict]:
    if len(data) < 45:
        raise ValueError("Packet too short for CSR header")
    n_rows = int.from_bytes(data[0:4], "little")
    n_cols = int.from_bytes(data[4:8], "little")
    nnz = int.from_bytes(data[8:12], "little")
    val_bits = int(data[12])
    idx_bits = int(data[13])
    has_scale = bool(data[14])
    col_codec = int(data[15])
    ptr_codec = int(data[16])
    col_original_len = int.from_bytes(data[17:21], "little")
    row_original_len = int.from_bytes(data[21:25], "little")
    table_bytes_len = int.from_bytes(data[25:29], "little")
    values_nbytes = int.from_bytes(data[29:33], "little")
    row_ptr_bytes_len = int.from_bytes(data[33:37], "little")
    col_bytes_len = int.from_bytes(data[37:41], "little")
    scale = float(np.frombuffer(data[41:45], dtype=np.float32, count=1)[0])
    value_dtype = _VAL_BITS_TO_DTYPE.get(val_bits)
    if value_dtype is None:
        raise ValueError("Unknown val_bits in CSR packet")
    if has_scale and val_bits != 8:
        raise ValueError("Scale can only be present for int8 payloads")
    if has_scale and scale <= 0.0:
        raise ValueError("Scaled int8 payload requires positive scale")
    if not has_scale:
        scale = 0.0
    index_dtype = {16: np.int16, 32: np.int32}.get(idx_bits)
    if index_dtype is None:
        raise ValueError("Unknown idx_bits in CSR packet")
    offset = 45
    huffman_table = data[offset : offset + table_bytes_len]
    offset += table_bytes_len
    values_end = offset + values_nbytes
    values = np.frombuffer(data[offset:values_end], dtype=value_dtype)
    offset = values_end
    col_bytes = data[offset : offset + col_bytes_len]
    offset += col_bytes_len
    row_ptr_bytes = data[offset : offset + row_ptr_bytes_len]
    offset += row_ptr_bytes_len
    if len(data) != offset:
        raise ValueError("Packet length mismatch")
    if ptr_codec == _PTR_CODEC_DELTA_RLE:
        row_delta = _rle_decode_int_array(row_ptr_bytes, row_original_len)
    else:
        row_delta = np.frombuffer(row_ptr_bytes, dtype=index_dtype).astype(np.int32, copy=False)
    row_ptr = delta_decode_row_ptr_global(row_delta).astype(np.int32, copy=False)

    if col_codec == _COL_CODEC_DELTA_HUFFMAN:
        col_delta = _huffman_decode_int_array(huffman_table, col_bytes, col_original_len)
    else:
        col_delta = np.frombuffer(col_bytes, dtype=index_dtype).astype(np.int32, copy=False)
    col_indices = delta_decode_col_indices_row_local(col_delta, row_ptr).astype(np.int32, copy=False)
    if row_ptr.size != n_rows + 1:
        raise ValueError("row_ptr length mismatch")
    if row_ptr[0] != 0 or row_ptr[-1] != nnz:
        raise ValueError("row_ptr does not match nnz")
    if col_indices.size != nnz:
        raise ValueError("col_indices length mismatch")
    csr = CSRMatrix(
        values=values,
        col_indices=col_indices,
        row_ptr=row_ptr.astype(np.int32, copy=False),
        shape=(n_rows, n_cols),
    )
    return csr, {
        "val_bits": val_bits,
        "idx_bits": idx_bits,
        "scale": scale,
        "has_scale": has_scale,
        "col_codec": col_codec,
        "ptr_codec": ptr_codec,
        "col_original_len": col_original_len,
        "row_original_len": row_original_len,
        "huffman_table_bytes": table_bytes_len,
        "header_bytes": 45,
        "decompression_flops_codec": int(col_original_len + row_original_len),
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
