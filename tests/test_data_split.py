"""Tests for deterministic dataset splitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import SplitConfig, SplitRatios, create_splits, run_split, select_subset


def _build_records(total: int = 100) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for idx in range(total):
        records.append(
            {
                "sample_id": f"sample-{idx}",
                "source": "source_a" if idx % 2 == 0 else "source_b",
                "license": "cc-by" if idx % 4 < 2 else "mit",
                "text": f"example {idx}",
            }
        )
    return records


def test_split_sizes_and_determinism() -> None:
    records = _build_records(100)
    ratios = SplitRatios(train=0.8, test=0.1, shadow=0.1)

    first = create_splits(records, ratios=ratios, seed=7)
    second = create_splits(records, ratios=ratios, seed=7)

    assert len(first["train"]) == 80
    assert len(first["test"]) == 10
    assert len(first["shadow"]) == 10

    assert [row["sample_id"] for row in first["train"]] == [
        row["sample_id"] for row in second["train"]
    ]
    assert [row["sample_id"] for row in first["test"]] == [
        row["sample_id"] for row in second["test"]
    ]
    assert [row["sample_id"] for row in first["shadow"]] == [
        row["sample_id"] for row in second["shadow"]
    ]


def test_architecture_counts_after_subset_selection() -> None:
    records = _build_records(5000)
    subset = select_subset(records, subset_size=3000, seed=2026)

    splits = create_splits(
        subset,
        ratios=SplitRatios(train=0.8, test=0.1, shadow=0.1),
        seed=2026,
        target_counts={"train": 2400, "test": 300, "shadow": 300},
    )

    assert len(splits["train"]) == 2400
    assert len(splits["test"]) == 300
    assert len(splits["shadow"]) == 300


def test_invalid_explicit_counts_raise() -> None:
    with pytest.raises(ValueError):
        create_splits(
            _build_records(30),
            ratios=SplitRatios(train=0.8, test=0.1, shadow=0.1),
            seed=5,
            target_counts={"train": 20, "test": 5, "shadow": 4},
        )



def test_read_dataset_json_support(tmp_path: Path) -> None:
    records = _build_records(5)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(records), encoding="utf-8")

    config = SplitConfig(
        dataset_path=dataset_path,
        run_id="json-run",
        output_root=tmp_path / "outputs",
        seed=3,
        ratios=SplitRatios(train=0.6, test=0.2, shadow=0.2),
        stratify_by=("source", "license"),
    )

    result = run_split(config)
    assert result["counts"] == {"train": 3, "test": 1, "shadow": 1}


def test_run_split_writes_manifests(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    records = _build_records(20)

    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "license", "text"])
        writer.writeheader()
        writer.writerows(records)

    config = SplitConfig(
        dataset_path=dataset_path,
        run_id="unit-test-run",
        output_root=tmp_path / "outputs",
        seed=11,
        ratios=SplitRatios(train=0.6, test=0.2, shadow=0.2),
        stratify_by=("source", "license"),
    )

    result = run_split(config)

    split_dir = Path(result["split_dir"])
    assert split_dir.exists()

    for split_name in ("train", "test", "shadow"):
        manifest = split_dir / f"{split_name}.csv"
        assert manifest.exists()
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ["sample_id", "source", "license"]
