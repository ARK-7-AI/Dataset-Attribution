"""Final-run gating checks used before attribution starts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.training.validate_outputs import validate_training_outputs


_REQUIRED_RUNTIME_METRIC_KEYS = {
    "train_runtime",
    "train_loss",
    "steps",
    "epochs_completed",
    "train_steps_per_second",
    "train_tokens_per_second",
    "timing_breakdown_s",
    "throughput",
}

_REQUIRED_THROUGHPUT_KEYS = {"steps_per_second", "tokens_per_second"}


@dataclass(frozen=True)
class FinalRunThresholds:
    """Thresholds used by the final-run gate."""

    min_steps: int = 100
    min_epochs: float = 1.0
    max_final_loss: float = 3.0


@dataclass(frozen=True)
class FinalRunCheckResult:
    """Result payload for final-run quality gates."""

    passed: bool
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _append_check(checks: list[dict[str, Any]], *, name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def _extract_loss_series(trainer_state: dict[str, Any]) -> list[float]:
    series: list[float] = []
    log_history = trainer_state.get("log_history", [])
    if not isinstance(log_history, list):
        return series

    for row in log_history:
        if not isinstance(row, dict):
            continue
        loss = row.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            series.append(float(loss))
    return series


def evaluate_final_run(train_dir: str | Path, thresholds: FinalRunThresholds | None = None) -> FinalRunCheckResult:
    """Evaluate final-run checklist criteria from training artifacts."""
    train_path = Path(train_dir)
    limits = thresholds or FinalRunThresholds()
    checks: list[dict[str, Any]] = []

    try:
        artifacts = validate_training_outputs(train_path)
        _append_check(
            checks,
            name="required_artifacts",
            passed=True,
            detail=f"Validated required artifacts in {artifacts.train_dir}",
        )
        _append_check(
            checks,
            name="downstream_validation_pass",
            passed=True,
            detail="validate_training_outputs gate passed",
        )
    except Exception as exc:  # pragma: no cover - single error branch for CLI clarity
        _append_check(checks, name="required_artifacts", passed=False, detail=str(exc))
        _append_check(
            checks,
            name="downstream_validation_pass",
            passed=False,
            detail="Skipped because required artifacts validation failed",
        )
        return FinalRunCheckResult(passed=False, checks=checks)

    metrics = _read_json(artifacts.metrics_path, label="metrics.json")
    trainer_state = _read_json(artifacts.trainer_state_path, label="trainer_state.json")

    missing_runtime_keys = sorted(key for key in _REQUIRED_RUNTIME_METRIC_KEYS if key not in metrics)
    throughput = metrics.get("throughput")
    missing_throughput_keys: list[str] = []
    if isinstance(throughput, dict):
        missing_throughput_keys = sorted(
            key for key in _REQUIRED_THROUGHPUT_KEYS if key not in throughput
        )
    else:
        missing_throughput_keys = sorted(_REQUIRED_THROUGHPUT_KEYS)

    runtime_metrics_ok = not missing_runtime_keys and not missing_throughput_keys
    runtime_detail = "All required runtime metrics captured"
    if missing_runtime_keys or missing_throughput_keys:
        runtime_detail = (
            f"Missing metrics keys={missing_runtime_keys} "
            f"missing throughput keys={missing_throughput_keys}"
        )
    _append_check(
        checks,
        name="required_runtime_metrics",
        passed=runtime_metrics_ok,
        detail=runtime_detail,
    )

    steps = int(metrics.get("steps", 0))
    epochs_completed = float(metrics.get("epochs_completed", 0.0))
    min_progress_ok = steps >= limits.min_steps and epochs_completed >= limits.min_epochs
    _append_check(
        checks,
        name="minimum_steps_epochs",
        passed=min_progress_ok,
        detail=(
            f"steps={steps} (min={limits.min_steps}), "
            f"epochs_completed={epochs_completed:.3f} (min={limits.min_epochs:.3f})"
        ),
    )

    train_loss = float(metrics.get("train_loss", float("inf")))
    loss_series = _extract_loss_series(trainer_state)
    finite_loss = math.isfinite(train_loss)
    bounded_loss = finite_loss and train_loss <= limits.max_final_loss
    trend_ok = True
    if loss_series:
        trend_ok = loss_series[-1] <= loss_series[0]

    loss_ok = bounded_loss and trend_ok
    _append_check(
        checks,
        name="acceptable_loss_behavior",
        passed=loss_ok,
        detail=(
            f"train_loss={train_loss:.6f} (max={limits.max_final_loss:.6f}), "
            f"loss_points={len(loss_series)}, trend_ok={trend_ok}"
        ),
    )

    return FinalRunCheckResult(
        passed=all(bool(item.get("passed")) for item in checks),
        checks=checks,
    )
