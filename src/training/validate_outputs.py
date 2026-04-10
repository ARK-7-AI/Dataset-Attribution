"""Training artifact validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_ARTIFACTS = {
    "adapter": "adapter",
    "tokenizer": "tokenizer",
    "metrics": "metrics.json",
    "params": "params.json",
    "trainer_state": "trainer_state.json",
}

_REQUIRED_PARAMS_KEYS = {
    "run_id",
    "output_dir",
    "base_model_path",
    "tokenizer_name_or_path",
    "dataset_json_path",
}

_REQUIRED_METRICS_KEYS = {
    "train_runtime",
    "train_loss",
    "steps",
    "train_steps_per_second",
    "throughput",
}


@dataclass(frozen=True)
class TrainingArtifacts:
    """Resolved training artifact locations."""

    train_dir: Path
    run_root: Path
    adapter_dir: Path
    tokenizer_dir: Path
    metrics_path: Path
    params_path: Path
    trainer_state_path: Path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_required_keys(payload: dict[str, Any], *, required: set[str], label: str) -> None:
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing adapter artifact directory: {adapter_dir}")

    expected_files = [
        adapter_dir / "adapter_model.bin",
        adapter_dir / "adapter_model.safetensors",
    ]
    if not any(path.exists() for path in expected_files):
        raise ValueError(
            "Adapter directory is present but no adapter weights were found. "
            f"Expected one of: {[path.name for path in expected_files]}"
        )


def _validate_tokenizer_dir(tokenizer_dir: Path) -> None:
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Missing tokenizer artifact directory: {tokenizer_dir}")

    expected_files = [
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
        tokenizer_dir / "special_tokens_map.json",
    ]
    if not any(path.exists() for path in expected_files):
        raise ValueError(
            "Tokenizer directory is present but no tokenizer files were found. "
            f"Expected one of: {[path.name for path in expected_files]}"
        )


def validate_training_outputs(train_dir: str | Path) -> TrainingArtifacts:
    """Validate train artifacts and schema required by downstream stages."""
    train_path = Path(train_dir)
    if not train_path.exists() or not train_path.is_dir():
        raise FileNotFoundError(f"Train artifact directory not found: {train_path}")

    missing_artifacts = []
    for label, relative in _REQUIRED_ARTIFACTS.items():
        if not (train_path / relative).exists():
            missing_artifacts.append(label)
    if missing_artifacts:
        raise FileNotFoundError(f"Missing required training artifacts: {missing_artifacts}")

    artifacts = TrainingArtifacts(
        train_dir=train_path,
        run_root=train_path.parent,
        adapter_dir=train_path / "adapter",
        tokenizer_dir=train_path / "tokenizer",
        metrics_path=train_path / "metrics.json",
        params_path=train_path / "params.json",
        trainer_state_path=train_path / "trainer_state.json",
    )

    _validate_adapter_dir(artifacts.adapter_dir)
    _validate_tokenizer_dir(artifacts.tokenizer_dir)

    params = _read_json(artifacts.params_path, label="params.json")
    metrics = _read_json(artifacts.metrics_path, label="metrics.json")
    trainer_state = _read_json(artifacts.trainer_state_path, label="trainer_state.json")

    _validate_required_keys(params, required=_REQUIRED_PARAMS_KEYS, label="params.json")
    _validate_required_keys(metrics, required=_REQUIRED_METRICS_KEYS, label="metrics.json")

    throughput = metrics.get("throughput")
    if not isinstance(throughput, dict):
        raise ValueError("metrics.json key 'throughput' must be an object")
    _validate_required_keys(
        throughput,
        required={"steps_per_second", "tokens_per_second"},
        label="metrics.json.throughput",
    )

    if "global_step" not in trainer_state and "log_history" not in trainer_state:
        raise ValueError(
            "trainer_state.json missing both 'global_step' and 'log_history'; "
            "downstream monitoring tools cannot consume it"
        )

    resolved_config = train_path / "resolved_config.yaml"
    if resolved_config.exists():
        with resolved_config.open("r", encoding="utf-8") as handle:
            loaded_config = yaml.safe_load(handle) or {}
        if not isinstance(loaded_config, dict):
            raise ValueError("resolved_config.yaml must deserialize to a mapping")

    return artifacts
