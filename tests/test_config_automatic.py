import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from config import get_config


def parse_args(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    return get_config()


def test_automatic_is_valid(monkeypatch):
    args = parse_args(monkeypatch, ["--sparsity_compression", "Automatic"])
    assert args.sparsity_compression == "Automatic"


@pytest.mark.parametrize("flag,value", [
    ("--automatic_csr_sparsity_threshold", "1.1"),
    ("--automatic_csr_sparsity_threshold", "-0.1"),
    ("--automatic_bitmask_sparsity_threshold", "1.1"),
])
def test_invalid_thresholds_rejected(monkeypatch, flag, value):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, [flag, value])


def test_bitmask_threshold_greater_than_csr_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, [
            "--automatic_bitmask_sparsity_threshold", "0.95",
            "--automatic_csr_sparsity_threshold", "0.90",
        ])


def test_negative_min_tensor_size_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        parse_args(monkeypatch, ["--automatic_min_tensor_size", "-1"])
