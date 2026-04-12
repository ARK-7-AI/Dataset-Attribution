"""Shared input resolver for attribution stages."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AttributionInputs:
    """Resolved run artifacts and sample-id mappings for attribution."""

    run_root: Path
    adapter_artifact: Path
    tokenizer_artifact: Path
    train_split_path: Path
    test_split_path: Path
    train_sample_ids: list[str]
    test_sample_ids: list[str]
    sample_id_to_train_index: dict[str, int]
    gradients_dir: Path | None
    gradient_subset_path: Path | None
    gradient_subset_ids: list[str]


def resolve_attribution_inputs(
    *,
    output_root: Path,
    run_id: str,
    require_gradients: bool,
) -> AttributionInputs:
    """Resolve run inputs and validate cross-stage sample-id consistency."""

    run_root = output_root / run_id
    train_dir = run_root / "train"

    adapter_artifact = _resolve_adapter_artifact(train_dir)
    tokenizer_artifact = _resolve_tokenizer_artifact(train_dir)

    splits_dir = run_root / "splits"
    train_split_path = splits_dir / "train.csv"
    test_split_path = splits_dir / "test.csv"
    train_sample_ids = _read_split_ids(train_split_path, split_name="train")
    test_sample_ids = _read_split_ids(test_split_path, split_name="test")

    train_id_set = set(train_sample_ids)
    overlap = train_id_set.intersection(test_sample_ids)
    if overlap:
        raise ValueError(
            "Train/test split overlap detected for sample_id(s): "
            + ", ".join(sorted(overlap)[:10])
        )

    sample_id_to_train_index = {sample_id: idx for idx, sample_id in enumerate(train_sample_ids)}

    gradients_dir = run_root / "gradients"
    gradient_subset_path: Path | None = None
    gradient_subset_ids: list[str] = []
    if require_gradients:
        if not gradients_dir.is_dir():
            raise FileNotFoundError(f"Gradients directory not found: {gradients_dir}")
        gradient_subset_path = gradients_dir / "subset_manifest.csv"
        gradient_subset_ids = _read_split_ids(gradient_subset_path, split_name="gradient_subset")

        missing_from_train = [sample_id for sample_id in gradient_subset_ids if sample_id not in train_id_set]
        if missing_from_train:
            raise ValueError(
                "Gradient subset contains sample_ids missing from train split: "
                + ", ".join(sorted(missing_from_train)[:10])
            )

        metadata_path = gradients_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_count = int(metadata.get("gradient_subset_size", len(gradient_subset_ids)))
            if expected_count != len(gradient_subset_ids):
                raise ValueError(
                    "Gradient subset size mismatch between metadata.json and subset_manifest.csv "
                    f"({expected_count} != {len(gradient_subset_ids)})"
                )

    return AttributionInputs(
        run_root=run_root,
        adapter_artifact=adapter_artifact,
        tokenizer_artifact=tokenizer_artifact,
        train_split_path=train_split_path,
        test_split_path=test_split_path,
        train_sample_ids=train_sample_ids,
        test_sample_ids=test_sample_ids,
        sample_id_to_train_index=sample_id_to_train_index,
        gradients_dir=gradients_dir if require_gradients else None,
        gradient_subset_path=gradient_subset_path,
        gradient_subset_ids=gradient_subset_ids,
    )


def _resolve_adapter_artifact(train_dir: Path) -> Path:
    adapter_dir = train_dir / "adapter"
    if adapter_dir.is_dir():
        return adapter_dir

    legacy_checkpoint = train_dir / "adapter_checkpoint.bin"
    if legacy_checkpoint.is_file():
        return legacy_checkpoint

    raise FileNotFoundError(
        "Adapter artifact not found. Expected either "
        f"directory: {adapter_dir} or file: {legacy_checkpoint}"
    )


def _resolve_tokenizer_artifact(train_dir: Path) -> Path:
    tokenizer_dir = train_dir / "tokenizer"
    if tokenizer_dir.is_dir():
        return tokenizer_dir

    legacy_tokenizer = train_dir / "tokenizer.json"
    if legacy_tokenizer.is_file():
        return legacy_tokenizer

    raise FileNotFoundError(
        "Tokenizer artifact not found. Expected either "
        f"directory: {tokenizer_dir} or file: {legacy_tokenizer}"
    )


def _read_split_ids(path: Path, *, split_name: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} split manifest not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{split_name} split manifest must include sample_id column")
        sample_ids = [str(row["sample_id"]).strip() for row in reader if (row.get("sample_id") or "").strip()]

    if not sample_ids:
        raise ValueError(f"{split_name} split manifest is empty")

    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_id in sample_ids:
        if sample_id in seen and sample_id not in duplicates:
            duplicates.append(sample_id)
        seen.add(sample_id)

    if duplicates:
        raise ValueError(
            f"{split_name} split manifest contains duplicate sample_id values: "
            + ", ".join(sorted(duplicates)[:10])
        )

    return sample_ids
