"""Tests for LogIX engine integration artifacts and CLI behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.attribution.logix_engine import run_logix_engine


class _FakeLogIX:
    @staticmethod
    def setup(**kwargs):
        return {"session": "fake", **kwargs}

    @staticmethod
    def run(**kwargs):
        sample_ids = kwargs.get("sample_ids", [])
        return {
            "status": "ok",
            "num_ranked": len(sample_ids),
            "top_sample_id": sample_ids[0] if sample_ids else None,
        }


def _write_train_manifest(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "license"])
        writer.writeheader()
        for idx in range(count):
            writer.writerow(
                {
                    "sample_id": f"sample-{idx:05d}",
                    "source": "source",
                    "license": "cc-by",
                }
            )


def test_logix_engine_writes_results_and_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-test"
    train_manifest_path = run_root / "splits" / "train.csv"
    _write_train_manifest(train_manifest_path, count=4)

    config_path = tmp_path / "attribution_logix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "seed": 2026,
                "train_manifest_path": str(train_manifest_path),
                "logix": {
                    "setup": {"device": "cpu"},
                    "run": {"top_k": 2},
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)

    assert artifacts.output_dir.exists()
    assert artifacts.results_path.exists()
    assert artifacts.metadata_path.exists()

    results = json.loads(artifacts.results_path.read_text(encoding="utf-8"))
    assert results["run_id"] == "logix-test"
    assert results["num_samples"] == 4
    assert results["results"]["status"] == "ok"
    assert results["results"]["num_ranked"] == 4

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_name_or_path"] == "fake-model"
    assert metadata["seed"] == 2026
    assert metadata["setup_kwargs"]["device"] == "cpu"
    assert metadata["run_kwargs"]["top_k"] == 2


def test_logix_engine_without_manifest_still_runs(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_nomani.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-no-manifest",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    results = json.loads(artifacts.results_path.read_text(encoding="utf-8"))
    assert results["num_samples"] == 0
    assert results["results"]["top_sample_id"] is None
