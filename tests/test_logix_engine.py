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
    assert "logix" in metadata["versions"]
    assert "transformers" in metadata["versions"]
    assert "torch" in metadata["versions"]
    assert metadata["timing"]["setup_seconds"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["setup"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["extraction"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["influence_scoring"] >= 0.0
    assert "platform" in metadata["hardware"]
    assert "cuda_available" in metadata["hardware"]
    assert "git_commit_hash" in metadata["git"]
    assert "git_dirty" in metadata["git"]
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
    run_root = tmp_path / "outputs" / "runs" / "logix-test"
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

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
        assert "LoRA adapter artifacts" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing adapter path")


def test_logix_engine_preflight_reports_run_id_mismatch_with_fix_hint(tmp_path: Path) -> None:
    valid_run_root = tmp_path / "outputs" / "runs" / "aligned-run"
    (valid_run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (valid_run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (valid_run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(valid_run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(valid_run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_bad_run_linkage.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "wrong-run",
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
        message = str(exc)
        assert "selected run_id='wrong-run'" in message
        assert "train split manifest" in message
        assert "test split manifest" in message
        assert "LoRA adapter artifacts" in message
        assert "same run_id for splits, training artifacts, and attribution" in message
        assert "train_manifest_path/test_manifest_path overrides" in message
    else:
        raise AssertionError("Expected preflight mismatch error for run_id with missing inputs")


def test_logix_engine_supports_split_path_overrides(tmp_path: Path) -> None:
    split_run_root = tmp_path / "outputs" / "runs" / "split-source"
    _write_manifest(split_run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(split_run_root / "splits" / "test.csv", count=2, start=100)

    artifact_run_root = tmp_path / "outputs" / "runs" / "artifact-source"
    (artifact_run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (artifact_run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (artifact_run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    config_path = tmp_path / "attribution_logix_override.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "artifact-source",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "train_manifest_path": str(split_run_root / "splits" / "train.csv"),
                "test_manifest_path": str(split_run_root / "splits" / "test.csv"),
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_manifest_path"].endswith("split-source/splits/train.csv")
    assert metadata["test_manifest_path"].endswith("split-source/splits/test.csv")
    assert metadata["train_manifest_path_override"].endswith("split-source/splits/train.csv")
    assert metadata["test_manifest_path_override"].endswith("split-source/splits/test.csv")


def test_logix_engine_mixed_run_layout_uses_explicit_manifest_paths(tmp_path: Path) -> None:
    final_report_root = tmp_path / "outputs" / "runs" / "final_report_run"
    (final_report_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (final_report_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (final_report_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    default_split_root = tmp_path / "outputs" / "runs" / "default_run"
    _write_manifest(default_split_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(default_split_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_mixed_run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "final_report_run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 2,
                "train_subset_size": 3,
                "train_manifest_path": "outputs/runs/default_run/splits/train.csv",
                "test_manifest_path": "outputs/runs/default_run/splits/test.csv",
                "shadow_manifest_path": "outputs/runs/default_run/splits/shadow.csv",
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
                    "adapter_path": "outputs/runs/final_report_run/train/adapter",
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        # Config paths in this regression use repo-relative paths.
        import os

        os.chdir(tmp_path)
        artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    finally:
        os.chdir(cwd)

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_manifest_path"] == "outputs/runs/default_run/splits/train.csv"
    assert metadata["test_manifest_path"] == "outputs/runs/default_run/splits/test.csv"
    assert metadata["train_manifest_path_override"] == "outputs/runs/default_run/splits/train.csv"
    assert metadata["test_manifest_path_override"] == "outputs/runs/default_run/splits/test.csv"


def test_logix_engine_tiny_smoke_flow_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    run_id = "smoke-flow"
    run_root = output_root / run_id
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    # Tiny sample manifests to keep this as a smoke flow.
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_payload = {
        "run_id": run_id,
        "output_root": str(output_root),
        "model_name_or_path": "fake-model",
        "seed": 1337,
        "top_k": 2,
        "train_subset_size": 3,
        "influence": {
            "mode": "ihvp",
            "ihvp": {
                "damping": 0.01,
                "scale": 10.0,
                "recursion_depth": 4,
                "num_samples": 1,
            },
        },
        "lora": {"lora_only": True},
        "logix": {
            "setup": {"device": "cpu"},
            "run": {},
            "test_subset_size": 2,
        },
    }
    config_path = tmp_path / "attribution_logix_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    first = run_logix_engine(config_path, logix_module=_FakeLogIX)
    second = run_logix_engine(config_path, logix_module=_FakeLogIX)

    expected_output_dir = output_root / run_id / "attribution" / "logix"
    assert first.output_dir == expected_output_dir
    assert first.metadata_path == expected_output_dir / "metadata.json"
    assert first.influence_scores_path == expected_output_dir / "influence_scores.csv"
    assert first.topk_path == expected_output_dir / "topk.json"

    assert first.metadata_path.exists()
    assert first.influence_scores_path.exists()
    assert first.topk_path.exists()

    csv_lines = first.influence_scores_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines
    assert csv_lines[0] == "test_id,train_id,influence_score,rank"
    assert len(csv_lines) > 1

    topk = json.loads(first.topk_path.read_text(encoding="utf-8"))
    assert topk
    for test_id, ranking in topk.items():
        assert test_id
        assert ranking
        assert all("train_id" in item and "influence_score" in item and "rank" in item for item in ranking)

    # Fixed seed/config should always produce the same ordering for train subset and rankings.
    metadata_first = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata_second = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata_first["artifacts"]["train_subset_ids"] == metadata_second["artifacts"]["train_subset_ids"]
    assert first.influence_scores_path.read_text(encoding="utf-8") == second.influence_scores_path.read_text(
        encoding="utf-8"
    )
    assert first.topk_path.read_text(encoding="utf-8") == second.topk_path.read_text(encoding="utf-8")
