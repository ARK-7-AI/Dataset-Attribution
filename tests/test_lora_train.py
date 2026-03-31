"""Tests for LoRA training entrypoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.lora_train import parse_args


def test_parse_args_reads_config_path() -> None:
    args = parse_args(["--config", "configs/train_lora.yaml"])
    assert args.config == "configs/train_lora.yaml"
