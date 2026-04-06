"""Tests for LoRA training entrypoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.lora_train import parse_args, run_training


def test_parse_args_reads_config_path() -> None:
    args = parse_args(["--config", "configs/train_lora.yaml"])
    assert args.config == "configs/train_lora.yaml"


def test_run_training_writes_expected_artifacts() -> None:
    run_dir = run_training("configs/train_lora.yaml")

    assert run_dir.exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "adapter_checkpoint.bin").exists()
