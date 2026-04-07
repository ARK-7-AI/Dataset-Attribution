"""Utilities for deterministic train/test/shadow splitting."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SplitRatios:
    """Ratios for train, test, and shadow splits."""

    train: float
    test: float
    shadow: float

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "test": self.test, "shadow": self.shadow}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for dataset splitting."""

    dataset_path: Path
    run_id: str
    output_root: Path
    seed: int
    ratios: SplitRatios
    stratify_by: tuple[str, ...]
    subset_size: int | None = None
    target_counts: dict[str, int] | None = None


def load_split_config(config_path: str | Path) -> SplitConfig:
    """Load split configuration from YAML."""

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load configs/data.yaml") from exc

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    dataset = raw["dataset"]
    split = raw["split"]
    ratios = split["ratios"]

    target_counts = split.get("counts")
    if target_counts is not None:
        target_counts = {
            "train": int(target_counts["train"]),
            "test": int(target_counts["test"]),
            "shadow": int(target_counts["shadow"]),
        }

    return SplitConfig(
        dataset_path=Path(dataset["path"]),
        run_id=str(raw["run_id"]),
        output_root=Path(raw.get("output_root", "outputs/runs")),
        seed=int(split["seed"]),
        ratios=SplitRatios(
            train=float(ratios["train"]),
            test=float(ratios["test"]),
            shadow=float(ratios["shadow"]),
        ),
        stratify_by=tuple(split.get("stratify_by", ["source", "license"])),
        subset_size=(int(dataset["subset_size"]) if dataset.get("subset_size") else None),
        target_counts=target_counts,
    )


def _coerce_record(record: dict[str, Any]) -> dict[str, str]:
    return {key: "" if value is None else str(value) for key, value in record.items()}


def _validate_dataset_rows(rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Dataset must contain at least one row")

    required_keys = ("sample_id", "source", "license")
    for index, row in enumerate(rows):
        missing_keys = [key for key in required_keys if key not in row]
        if missing_keys:
            missing_str = ", ".join(missing_keys)
            raise ValueError(f"Invalid dataset row at index {index}: missing required keys: {missing_str}")


def read_dataset(dataset_path: str | Path) -> list[dict[str, str]]:
    """Read CSV or JSON dataset into a list of records."""

    path = Path(dataset_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
        _validate_dataset_rows(rows)
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)

        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            data = parsed.get("data")
            if not isinstance(data, list):
                raise ValueError("JSON dataset object must contain a 'data' list")
            items = data
        else:
            raise ValueError("JSON dataset must be a list or an object with a 'data' list")

        if not items:
            raise ValueError("Dataset must contain at least one row")

        rows: list[dict[str, str]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid dataset row at index {index}: row must be a JSON object")
            rows.append(_coerce_record(dict(item)))

        _validate_dataset_rows(rows)
        return rows

    raise ValueError(f"Unsupported dataset format for '{path.name}'. Use .csv or .json")


def _allocate_counts(size: int, ratios: SplitRatios) -> dict[str, int]:
    ratio_map = ratios.as_dict()
    if not math.isclose(sum(ratio_map.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    exact = {split: ratio_map[split] * size for split in ratio_map}
    counts = {split: int(math.floor(value)) for split, value in exact.items()}
    remainder = size - sum(counts.values())
    ranked = sorted(
        ratio_map,
        key=lambda split: (exact[split] - counts[split], ratio_map[split], split),
        reverse=True,
    )
    for split_name in ranked[:remainder]:
        counts[split_name] += 1

    return counts


def _resolve_target_counts(
    total_size: int, ratios: SplitRatios, target_counts: dict[str, int] | None
) -> dict[str, int]:
    if target_counts is None:
        return _allocate_counts(total_size, ratios)

    resolved = {name: int(target_counts[name]) for name in ("train", "test", "shadow")}
    if sum(resolved.values()) != total_size:
        raise ValueError(
            "Configured split.counts must sum to the selected dataset size "
            f"({sum(resolved.values())} != {total_size})"
        )
    return resolved


def select_subset(
    records: list[dict[str, str]],
    subset_size: int,
    seed: int,
    stratify_by: tuple[str, ...] = ("source", "license"),
) -> list[dict[str, str]]:
    """Select a deterministic subset while preserving stratum proportions."""

    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size >= len(records):
        return list(records)

    strata: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for record in records:
        key = tuple(record.get(column, "__MISSING__") for column in stratify_by)
        strata.setdefault(key, []).append(record)

    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)

    # Ratio-like target per stratum based on representation.
    exact = {key: (len(rows) / len(records)) * subset_size for key, rows in strata.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = subset_size - sum(counts.values())
    for key in sorted(exact, key=lambda item: (exact[item] - counts[item], len(strata[item])), reverse=True)[:remainder]:
        counts[key] += 1

    selected: list[dict[str, str]] = []
    for key in sorted(strata):
        selected.extend(strata[key][: counts[key]])

    random.Random(seed + 101).shuffle(selected)
    return selected


def create_splits(
    records: list[dict[str, str]],
    ratios: SplitRatios,
    seed: int,
    stratify_by: tuple[str, ...] = ("source", "license"),
    target_counts: dict[str, int] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Create deterministic stratified train/test/shadow splits."""

    strata: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for record in records:
        key = tuple(record.get(column, "__MISSING__") for column in stratify_by)
        strata.setdefault(key, []).append(record)

    rng = random.Random(seed)
    split_records: dict[str, list[dict[str, str]]] = {"train": [], "test": [], "shadow": []}
    split_names = ("train", "test", "shadow")
    target_counts_resolved = _resolve_target_counts(len(records), ratios, target_counts)
    remaining = dict(target_counts_resolved)

    ordered_strata = sorted(strata)
    for stratum_index, stratum_key in enumerate(ordered_strata):
        bucket = list(strata[stratum_key])
        rng.shuffle(bucket)

        if stratum_index == len(ordered_strata) - 1:
            counts = dict(remaining)
        else:
            desired = _allocate_counts(len(bucket), ratios)
            counts = {name: min(desired[name], remaining[name]) for name in split_names}
            deficit = len(bucket) - sum(counts.values())
            if deficit > 0:
                for split_name in sorted(
                    split_names,
                    key=lambda name: (
                        remaining[name] - counts[name],
                        target_counts_resolved[name],
                        ratios.as_dict()[name],
                        name,
                    ),
                    reverse=True,
                ):
                    if deficit == 0:
                        break
                    spare = remaining[split_name] - counts[split_name]
                    if spare <= 0:
                        continue
                    add = min(spare, deficit)
                    counts[split_name] += add
                    deficit -= add

        for split_name in split_names:
            remaining[split_name] -= counts[split_name]

        start = 0
        for split_name in split_names:
            end = start + counts[split_name]
            split_records[split_name].extend(bucket[start:end])
            start = end

    for index, split_name in enumerate(split_names, start=1):
        random.Random(seed + index).shuffle(split_records[split_name])

    return split_records


def write_split_manifests(
    split_records: dict[str, list[dict[str, str]]], output_dir: str | Path
) -> dict[str, Path]:
    """Write split manifests with sample and metadata columns."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    required_columns = ["sample_id", "source", "license"]
    manifest_paths: dict[str, Path] = {}

    for split_name, records in split_records.items():
        manifest = output_path / f"{split_name}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=required_columns)
            writer.writeheader()
            for record in records:
                row = {column: record.get(column, "") for column in required_columns}
                writer.writerow(row)
        manifest_paths[split_name] = manifest

    return manifest_paths


def run_split(config: SplitConfig) -> dict[str, Any]:
    """Run full split pipeline from source dataset to manifest files."""

    records = read_dataset(config.dataset_path)
    if config.subset_size is not None:
        records = select_subset(
            records,
            subset_size=config.subset_size,
            seed=config.seed,
            stratify_by=config.stratify_by,
        )

    split_records = create_splits(
        records=records,
        ratios=config.ratios,
        seed=config.seed,
        stratify_by=config.stratify_by,
        target_counts=config.target_counts,
    )
    split_dir = config.output_root / config.run_id / "splits"
    manifests = write_split_manifests(split_records, split_dir)

    return {
        "counts": {split: len(rows) for split, rows in split_records.items()},
        "selected_records": len(records),
        "split_dir": split_dir,
        "manifests": manifests,
    }
