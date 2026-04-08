"""Tests for dataset normalization used by training ingestion."""

from __future__ import annotations

import pytest

from src.data.normalize import normalize_records


def test_normalize_autogenerates_missing_sample_id() -> None:
    rows = [{"instruction": "Say hi", "output": "Hi!"}]

    normalized = normalize_records(rows, dataset_name="alpaca")

    assert normalized[0]["sample_id"] == "alpaca-000000"


def test_normalize_fills_source_and_license_defaults() -> None:
    rows = [{"sample_id": "abc-1", "instruction": "Say hi", "output": "Hi!"}]

    normalized = normalize_records(rows, dataset_name="alpaca")

    assert normalized[0]["source"] == "alpaca"
    assert normalized[0]["license"] == "unknown"


def test_normalize_raises_clear_indexed_error_for_malformed_record() -> None:
    rows = [{"instruction": "ok", "output": "ok"}, "not-a-dict"]

    with pytest.raises(ValueError, match=r"index 1"):
        normalize_records(rows, dataset_name="alpaca")


def test_normalize_raises_when_text_payload_is_empty() -> None:
    rows = [{"sample_id": "s1", "instruction": "  ", "output": ""}]

    with pytest.raises(ValueError, match=r"index 0: missing required non-empty text payload"):
        normalize_records(rows, dataset_name="alpaca")
