import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from main import (
    compute_download_traffic_for_round,
    serialized_tensor_dict_bytes,
    tensor_dict_bytes,
)


def test_serialized_tensor_dict_bytes_matches_torch_save_payload_length():
    state = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bias": torch.ones(2, dtype=torch.float32),
    }

    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buffer)

    assert serialized_tensor_dict_bytes(state) == len(buffer.getvalue())
    assert serialized_tensor_dict_bytes(state) != tensor_dict_bytes(state)


def test_download_traffic_multiplies_serialized_global_payload_by_active_clients():
    global_payload = {"weight": torch.ones(4, dtype=torch.float32)}
    active_clients = 3

    assert compute_download_traffic_for_round(
        serialized_tensor_dict_bytes(global_payload), active_clients
    ) == serialized_tensor_dict_bytes(global_payload) * active_clients
