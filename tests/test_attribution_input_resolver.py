"""Tests for attribution input resolver cross-stage sample_id mapping."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.attribution.input_resolver import resolve_attribution_inputs


def _write_manifest(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        for sample_id in sample_ids:
            writer.writerow([sample_id])


def test_resolver_loads_train_test_and_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "run-a"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", ["s1", "s2", "s3"])
    _write_manifest(run_root / "splits" / "test.csv", ["s4", "s5"])

    resolved = resolve_attribution_inputs(output_root=tmp_path / "outputs" / "runs", run_id="run-a", require_gradients=False)

    assert resolved.adapter_artifact == run_root / "train" / "adapter"
    assert resolved.tokenizer_artifact == run_root / "train" / "tokenizer"
    assert resolved.train_sample_ids == ["s1", "s2", "s3"]
    assert resolved.test_sample_ids == ["s4", "s5"]
    assert resolved.sample_id_to_train_index == {"s1": 0, "s2": 1, "s3": 2}


def test_resolver_validates_gradient_subset_membership(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "run-grad"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", ["s1", "s2", "s3"])
    _write_manifest(run_root / "splits" / "test.csv", ["s4", "s5"])
    _write_manifest(run_root / "gradients" / "subset_manifest.csv", ["s1", "s3"])
    (run_root / "gradients" / "metadata.json").write_text(
        json.dumps({"gradient_subset_size": 2}),
        encoding="utf-8",
    )

    resolved = resolve_attribution_inputs(output_root=tmp_path / "outputs" / "runs", run_id="run-grad", require_gradients=True)

    assert resolved.gradient_subset_ids == ["s1", "s3"]
    assert resolved.gradient_subset_path == run_root / "gradients" / "subset_manifest.csv"
