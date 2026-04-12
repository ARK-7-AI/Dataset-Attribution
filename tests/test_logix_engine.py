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


def _write_manifest(path: Path, count: int, start: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "license"])
        writer.writeheader()
        for idx in range(start, start + count):
            writer.writerow(
                {
                    "sample_id": f"sample-{idx:05d}",
                    "source": "source",
                    "license": "cc-by",
                }
            )


def test_logix_engine_writes_results_and_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-test"
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=4, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "seed": 2026,
                "top_k": 2,
                "train_subset_size": 4,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {
                    "setup": {"device": "cpu"},
                    "run": {},
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)

    assert artifacts.output_dir.exists()
    assert artifacts.influence_scores_path.exists()
    assert artifacts.topk_path.exists()
    assert artifacts.metadata_path.exists()

    influence_lines = artifacts.influence_scores_path.read_text(encoding="utf-8").strip().splitlines()
    assert influence_lines[0] == "test_id,train_id,influence_score,rank"
    assert len(influence_lines) == 1 + (2 * 4)

    topk = json.loads(artifacts.topk_path.read_text(encoding="utf-8"))
    assert len(topk) == 2
    first_test = sorted(topk.keys())[0]
    assert len(topk[first_test]) == 2
    assert topk[first_test][0]["rank"] == 1

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "logix-test"
    assert metadata["model_name_or_path"] == "fake-model"
    assert metadata["seed"] == 2026
    assert metadata["top_k"] == 2
    assert metadata["setup_kwargs"]["device"] == "cpu"
    assert metadata["versions"]["python"]
    assert metadata["timing"]["total_seconds"] >= 0.0
    assert metadata["artifacts"]["execute_logix_result"]["status"] == "ok"


def test_logix_engine_without_manifest_still_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-with-default-splits"
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_nomani.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-with-default-splits",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "top_k": 2,
                "train_subset_size": 8,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["artifacts"]["num_samples"] == 3
    assert metadata["artifacts"]["execute_logix_result"]["top_sample_id"] is not None


def test_logix_engine_rejects_invalid_run_id(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_bad_id.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "bad/run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": False},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Invalid run_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid run_id")


def test_logix_engine_rejects_unsupported_influence_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_bad_mode.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {"mode": "dense_hessian", "ihvp": {}},
                "lora": {"lora_only": False},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Unsupported influence mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported influence mode")


def test_logix_engine_rejects_missing_train_artifacts_for_lora_only(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_missing_adapter.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except FileNotFoundError as exc:
        assert "Adapter artifact not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing adapter path")
