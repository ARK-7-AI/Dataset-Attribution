"""Normalization and validation helpers for JSON training datasets."""

from __future__ import annotations

from typing import Any


REQUIRED_TEXT_CANDIDATE_FIELDS: tuple[str, ...] = (
    "text",
    "prompt",
    "instruction",
    "input",
    "response",
    "output",
    "completion",
    "target",
)


def _dataset_prefix(dataset_name: str) -> str:
    prefix = dataset_name.strip().replace(" ", "-")
    return prefix or "dataset"


def _has_training_text(record: dict[str, Any]) -> bool:
    for field in REQUIRED_TEXT_CANDIDATE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def normalize_records(records: list[dict], dataset_name: str) -> list[dict]:
    """Normalize dataset rows and enforce minimal schema for training ingestion.

    Each returned row is a shallow copy of the original row with required metadata:
    - sample_id: deterministic default of "{dataset_name}-{index:06d}"
    - source: defaults to dataset_name
    - license: defaults to "unknown"

    Validation is strict and index-aware:
    - non-dict rows are rejected
    - rows with no non-empty text payload candidates are rejected
    """

    if not records:
        raise ValueError("Dataset must contain at least one row")

    normalized: list[dict] = []
    source_default = _dataset_prefix(dataset_name)

    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid dataset row at index {index}: expected a JSON object")

        record = dict(row)
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            sample_id = f"{source_default}-{index:06d}"
        record["sample_id"] = sample_id

        source = str(record.get("source") or "").strip()
        record["source"] = source or source_default

        license_name = str(record.get("license") or "").strip()
        record["license"] = license_name or "unknown"

        if not _has_training_text(record):
            raise ValueError(
                "Invalid dataset row at index "
                f"{index}: missing required non-empty text payload "
                f"(checked fields: {', '.join(REQUIRED_TEXT_CANDIDATE_FIELDS)})"
            )

        normalized.append(record)

    return normalized
