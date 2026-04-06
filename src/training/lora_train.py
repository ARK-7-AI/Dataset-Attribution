"""LoRA training entrypoint module."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import yaml


def build_parser() -> ArgumentParser:
    """Build CLI parser for the LoRA training entrypoint."""
    parser = ArgumentParser(description="LoRA training entrypoint")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Parse CLI arguments for training."""
    return build_parser().parse_args(argv)


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML mapping.")

    return data


def _build_run_id() -> str:
    """Build a unique run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _get_config_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract required training parameters from config."""
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    return {
        "experiment_name": config.get("experiment_name", "lora_run"),
        "base_model_path": config.get("base_model_path")
        or config.get("model_name_or_path", ""),
        "lora_rank": int(lora_cfg.get("rank", 8)),
        "lora_alpha": float(lora_cfg.get("alpha", 16)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.0)),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
        "epochs": int(train_cfg.get("epochs", 1)),
        "seed": int(config.get("seed", 42)),
    }


def _persist_run_outputs(
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Persist train artifacts for a run and return run directory."""
    run_dir = Path("outputs") / "runs" / run_id / "train"
    run_dir.mkdir(parents=True, exist_ok=True)

    params_path = run_dir / "params.json"
    metrics_path = run_dir / "metrics.json"
    adapter_path = run_dir / "adapter_checkpoint.bin"

    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    adapter_path.write_bytes(b"placeholder-adapter-checkpoint")

    return run_dir


def run_training(config_path: str) -> Path:
    """Run a placeholder LoRA training flow and return the output directory."""
    config = _load_config(config_path)
    params = _get_config_params(config)
    run_id = _build_run_id()

    metrics = {
        "status": "completed",
        "train_loss": 0.0,
        "epochs_completed": params["epochs"],
        "checkpoint": "adapter_checkpoint.bin",
    }

    run_dir = _persist_run_outputs(run_id=run_id, params=params, metrics=metrics)
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    """Run LoRA training workflow."""
    args = parse_args(argv)
    run_dir = run_training(args.config)
    print(f"LoRA training finished. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
