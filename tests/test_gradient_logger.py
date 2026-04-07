"""Smoke tests for gradient extraction artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attribution.gradient_logger import run_gradient_logging


def _write_train_manifest(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "license"])
        writer.writeheader()
        for idx in range(count):
            writer.writerow(
                {
                    "sample_id": f"sample-{idx:05d}",
                    "source": "source_a" if idx % 2 == 0 else "source_b",
                    "license": "cc-by" if idx % 3 == 0 else "mit",
                }
            )


def test_gradient_logger_creates_expected_artifacts_for_600_subset(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "unit-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "adapter" / "adapter_model.bin").write_bytes(b"adapter")
    _write_train_manifest(run_root / "splits" / "train.csv", count=3600)

    config_path = tmp_path / "attribution.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "unit-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "test-model",
                "seed": 2026,
                "gradients": {
                    "gradient_subset_size": 600,
                    "lora_only": True,
                    "save_format": "npy",
                    "max_seq_len": 1024,
                    "batch_size": 128,
                    "dtype": "float32",
                    "layer_filter": "lora",
                },
            }
        ),
        encoding="utf-8",
    )

    gradients_dir = run_gradient_logging(config_path)

    manifest = gradients_dir / "subset_manifest.csv"
    metadata = gradients_dir / "metadata.json"

    assert manifest.exists()
    assert metadata.exists()

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        sample_ids = [row["sample_id"] for row in reader]

    assert len(sample_ids) == 600

    chunk_files = sorted(gradients_dir.glob("gradients_*.npy"))
    assert chunk_files

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["gradient_subset_size"] == 600
    assert payload["lora_only"] is True
    assert payload["save_format"] == "npy"
    assert payload["adapter_artifact"].endswith("/train/adapter")


def test_gradient_logger_supports_legacy_adapter_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "legacy-run"
    (run_root / "train").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "adapter_checkpoint.bin").write_bytes(b"adapter")
    _write_train_manifest(run_root / "splits" / "train.csv", count=10)

    config_path = tmp_path / "attribution_legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "legacy-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "gradients": {"gradient_subset_size": 5},
            }
        ),
        encoding="utf-8",
    )

    gradients_dir = run_gradient_logging(config_path)
    payload = json.loads((gradients_dir / "metadata.json").read_text(encoding="utf-8"))
    assert payload["adapter_artifact"].endswith("/train/adapter_checkpoint.bin")
