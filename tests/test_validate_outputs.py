"""Tests for training output artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.validate_outputs import validate_training_outputs


def _build_train_dir(tmp_path: Path) -> Path:
    train_dir = tmp_path / "runs" / "testrun" / "train"
    (train_dir / "adapter").mkdir(parents=True)
    (train_dir / "tokenizer").mkdir(parents=True)

    (train_dir / "adapter" / "adapter_model.bin").write_bytes(b"adapter")
    (train_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    (train_dir / "params.json").write_text(
        json.dumps(
            {
                "run_id": "testrun",
                "output_dir": str(tmp_path / "runs"),
                "base_model_path": "fake-model",
                "tokenizer_name_or_path": "fake-model",
                "dataset_json_path": str(tmp_path / "dataset.json"),
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_runtime": 1.5,
                "train_loss": 0.1,
                "steps": 2,
                "train_steps_per_second": 3.0,
                "throughput": {"steps_per_second": 3.0, "tokens_per_second": 900.0},
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 2, "log_history": [{"loss": 0.1}]}),
        encoding="utf-8",
    )
    return train_dir


def test_validate_training_outputs_success(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    artifacts = validate_training_outputs(train_dir)
    assert artifacts.train_dir == train_dir
    assert artifacts.adapter_dir.is_dir()


def test_validate_training_outputs_missing_required_artifact(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    (train_dir / "metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match="Missing required training artifacts"):
        validate_training_outputs(train_dir)


def test_validate_training_outputs_missing_required_schema_key(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    (train_dir / "metrics.json").write_text(
        json.dumps({"train_runtime": 1.0, "steps": 2, "throughput": {"steps_per_second": 2.0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metrics.json missing required keys"):
        validate_training_outputs(train_dir)


def test_validate_training_outputs_fails_downstream_schema_load(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    (train_dir / "trainer_state.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    with pytest.raises(ValueError, match="downstream monitoring tools"):
        validate_training_outputs(train_dir)
