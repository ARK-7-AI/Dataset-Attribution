"""Tests for final-run gating checks."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.final_run_check import FinalRunThresholds, evaluate_final_run


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
                "train_runtime": 12.0,
                "train_loss": 0.6,
                "steps": 120,
                "epochs_completed": 1.0,
                "train_steps_per_second": 10.0,
                "train_tokens_per_second": 2048.0,
                "timing_breakdown_s": {"train": 12.0},
                "throughput": {"steps_per_second": 10.0, "tokens_per_second": 2048.0},
            }
        ),
        encoding="utf-8",
    )
    (train_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 120, "log_history": [{"loss": 1.0}, {"loss": 0.8}, {"loss": 0.6}]}),
        encoding="utf-8",
    )

    return train_dir


def test_evaluate_final_run_passes_for_valid_artifacts(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    result = evaluate_final_run(train_dir)

    assert result.passed is True
    assert all(check["passed"] for check in result.checks)


def test_evaluate_final_run_fails_for_missing_progress(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    thresholds = FinalRunThresholds(min_steps=200, min_epochs=2.0, max_final_loss=3.0)

    result = evaluate_final_run(train_dir, thresholds=thresholds)

    assert result.passed is False
    failed_names = {check["name"] for check in result.checks if not check["passed"]}
    assert "minimum_steps_epochs" in failed_names


def test_evaluate_final_run_fails_for_bad_loss_behavior(tmp_path: Path) -> None:
    train_dir = _build_train_dir(tmp_path)
    (train_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train_runtime": 12.0,
                "train_loss": 5.1,
                "steps": 120,
                "epochs_completed": 1.0,
                "train_steps_per_second": 10.0,
                "train_tokens_per_second": 2048.0,
                "timing_breakdown_s": {"train": 12.0},
                "throughput": {"steps_per_second": 10.0, "tokens_per_second": 2048.0},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_final_run(train_dir)

    assert result.passed is False
    failed_names = {check["name"] for check in result.checks if not check["passed"]}
    assert "acceptable_loss_behavior" in failed_names
