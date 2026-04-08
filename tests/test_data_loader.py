"""Tests for split-aware instruction dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.data_loader import load_instruction_datasets, preflight_validate_data_paths


class FakeTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        values = [1] * max(1, len(text.split()))
        return {"input_ids": values, "attention_mask": [1] * len(values)}


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict[str, object]]) -> "FakeDataset":
        return cls(rows)


class FakeDatasetsModule:
    Dataset = FakeDataset


def test_loader_fails_when_split_has_unknown_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([{"sample_id": "s1", "prompt": "p", "response": "r"}]), encoding="utf-8"
    )

    splits_dir = tmp_path / "runs" / "run-a" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\nmissing\n", encoding="utf-8")

    config = {
        "run_id": "run-a",
        "output_root": str(tmp_path / "runs"),
        "data": {
            "path": str(dataset_path),
            "train_manifest_path": str(splits_dir / "train.csv"),
            "text_fields": ["prompt", "response"],
            "prompt_field": "prompt",
            "response_field": "response",
        },
        "training": {"max_seq_len": 8},
    }

    monkeypatch.setattr("src.training.data_loader.importlib.import_module", lambda _: FakeDatasetsModule)

    with pytest.raises(ValueError, match="not present in original dataset"):
        load_instruction_datasets(config=config, tokenizer=FakeTokenizer())


def test_preflight_validation_reports_missing_dataset_and_manifest_paths(tmp_path: Path) -> None:
    config = {
        "data": {
            "dataset_json_path": str(tmp_path / "alpaca_data.json"),
            "train_manifest_path": str(tmp_path / "splits" / "train.csv"),
            "test_manifest_path": str(tmp_path / "splits" / "test.csv"),
            "shadow_manifest_path": str(tmp_path / "splits" / "shadow.csv"),
        }
    }

    with pytest.raises(FileNotFoundError, match="Dataset JSON not found"):
        preflight_validate_data_paths(config)

    dataset_path = tmp_path / "alpaca_data.json"
    dataset_path.write_text("[]", encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ns1\n", encoding="utf-8")
    (splits_dir / "test.csv").write_text("sample_id\ns1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Shadow manifest not found"):
        preflight_validate_data_paths(config)


def test_preflight_validation_rejects_src_dataset_paths(tmp_path: Path) -> None:
    config = {
        "data": {
            "dataset_json_path": "src/data/alpaca_data.json",
            "train_manifest_path": str(tmp_path / "splits" / "train.csv"),
            "test_manifest_path": str(tmp_path / "splits" / "test.csv"),
            "shadow_manifest_path": str(tmp_path / "splits" / "shadow.csv"),
        }
    }

    with pytest.raises(ValueError, match=r"must not live under 'src/'"):
        preflight_validate_data_paths(config)


def test_loader_normalizes_missing_sample_id_source_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "alpaca.json"
    dataset_path.write_text(
        json.dumps([{"instruction": "Write a haiku", "response": "Soft rain at dusk"}]),
        encoding="utf-8",
    )

    splits_dir = tmp_path / "runs" / "run-a" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\nalpaca-000000\n", encoding="utf-8")

    config = {
        "run_id": "run-a",
        "output_root": str(tmp_path / "runs"),
        "data": {
            "path": str(dataset_path),
            "train_manifest_path": str(splits_dir / "train.csv"),
            "text_fields": ["instruction", "response"],
            "prompt_field": "instruction",
            "response_field": "response",
            "normalized_snapshot_path": str(tmp_path / "data" / "processed" / "alpaca_normalized.json"),
        },
        "training": {"max_seq_len": 8},
    }

    monkeypatch.setattr("src.training.data_loader.importlib.import_module", lambda _: FakeDatasetsModule)

    train_dataset, eval_dataset = load_instruction_datasets(config=config, tokenizer=FakeTokenizer())

    assert eval_dataset is None
    assert train_dataset.rows[0]["sample_id"] == "alpaca-000000"

    normalized_snapshot = Path(config["data"]["normalized_snapshot_path"])
    parsed_snapshot = json.loads(normalized_snapshot.read_text(encoding="utf-8"))
    assert parsed_snapshot[0]["sample_id"] == "alpaca-000000"
    assert parsed_snapshot[0]["source"] == "alpaca"
    assert parsed_snapshot[0]["license"] == "unknown"


def test_loader_rejects_malformed_record_with_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(["bad-record"]), encoding="utf-8")

    splits_dir = tmp_path / "runs" / "run-a" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ndataset-000000\n", encoding="utf-8")

    config = {
        "run_id": "run-a",
        "output_root": str(tmp_path / "runs"),
        "data": {
            "path": str(dataset_path),
            "train_manifest_path": str(splits_dir / "train.csv"),
            "text_fields": ["prompt", "response"],
            "prompt_field": "prompt",
            "response_field": "response",
        },
        "training": {"max_seq_len": 8},
    }

    monkeypatch.setattr("src.training.data_loader.importlib.import_module", lambda _: FakeDatasetsModule)

    with pytest.raises(ValueError, match=r"index 0"):
        load_instruction_datasets(config=config, tokenizer=FakeTokenizer())


def test_loader_accepts_alpaca_output_via_response_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([{"sample_id": "s1", "instruction": "Summarize this", "output": "Done."}]),
        encoding="utf-8",
    )

    splits_dir = tmp_path / "runs" / "run-a" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ns1\n", encoding="utf-8")

    config = {
        "run_id": "run-a",
        "output_root": str(tmp_path / "runs"),
        "data": {
            "path": str(dataset_path),
            "train_manifest_path": str(splits_dir / "train.csv"),
            "text_fields": ["instruction", "output"],
            "prompt_field": "instruction",
            "response_field": "response",
            "response_fallback_fields": ["output", "answer", "completion"],
        },
        "training": {"max_seq_len": 8},
    }

    monkeypatch.setattr("src.training.data_loader.importlib.import_module", lambda _: FakeDatasetsModule)

    train_dataset, eval_dataset = load_instruction_datasets(config=config, tokenizer=FakeTokenizer())

    assert eval_dataset is None
    assert train_dataset.rows[0]["sample_id"] == "s1"


def test_loader_errors_with_sample_id_and_tested_fields_when_target_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([{"sample_id": "s1", "instruction": "Summarize this"}]),
        encoding="utf-8",
    )

    splits_dir = tmp_path / "runs" / "run-a" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ns1\n", encoding="utf-8")

    config = {
        "run_id": "run-a",
        "output_root": str(tmp_path / "runs"),
        "data": {
            "path": str(dataset_path),
            "train_manifest_path": str(splits_dir / "train.csv"),
            "text_fields": ["instruction"],
            "prompt_field": "instruction",
            "response_field": "response",
            "response_fallback_fields": ["output", "answer", "completion"],
        },
        "training": {"max_seq_len": 8},
    }

    monkeypatch.setattr("src.training.data_loader.importlib.import_module", lambda _: FakeDatasetsModule)

    with pytest.raises(
        ValueError,
        match=r"Sample 's1' is missing target answer \(tested_fields=\['response', 'output', 'answer', 'completion'\]\)",
    ):
        load_instruction_datasets(config=config, tokenizer=FakeTokenizer())
