"""Dataset loading utilities for LoRA instruction tuning."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def _read_dataset_records(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Original dataset JSON not found: {dataset_path}")

    if dataset_path.suffix.lower() != ".json":
        raise ValueError(
            "Original dataset must be JSON (.json) so records can be joined by sample_id"
        )

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    else:
        raise ValueError("Original dataset JSON must be a list or an object with a 'data' list")

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid dataset row at index {index}: expected a JSON object")
        records.append(dict(row))

    if not records:
        raise ValueError("Original dataset is empty")

    return records


def _read_split_ids(split_path: Path, split_name: str) -> list[str]:
    if not split_path.exists():
        if split_name == "test":
            return []
        raise FileNotFoundError(f"Required split manifest not found: {split_path}")

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError(
                f"Split manifest '{split_path}' must include a 'sample_id' column. "
                f"Found columns: {reader.fieldnames or []}"
            )

        sample_ids: list[str] = []
        for row_index, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(
                    f"Split manifest '{split_path}' has an empty sample_id at CSV line {row_index}"
                )
            sample_ids.append(sample_id)

    if split_name == "train" and not sample_ids:
        raise ValueError(f"Train split manifest '{split_path}' has no sample IDs")

    return sample_ids


def _build_record_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row_index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(
                f"Original dataset row at index {row_index} is missing required 'sample_id'"
            )
        if sample_id in index:
            raise ValueError(f"Duplicate sample_id in original dataset: '{sample_id}'")
        index[sample_id] = record
    return index


def _normalize_example(sample_id: str, record: dict[str, Any]) -> dict[str, str]:
    text = str(record.get("text") or "").strip()
    prompt = str(record.get("prompt") or record.get("instruction") or "").strip()
    response = str(record.get("response") or record.get("completion") or record.get("output") or "").strip()

    if not prompt and text:
        prompt = text

    if not prompt:
        raise ValueError(
            f"Sample '{sample_id}' is missing prompt text ('prompt'/'instruction'/'text')"
        )
    if not response:
        raise ValueError(
            f"Sample '{sample_id}' is missing target answer ('response'/'completion'/'output')"
        )

    prompt_text = PROMPT_TEMPLATE.format(instruction=prompt)
    full_text = f"{prompt_text}{response}"

    return {
        "sample_id": sample_id,
        "prompt": prompt_text,
        "target": response,
        "text": full_text,
    }


def _build_examples_for_split(
    split_name: str,
    sample_ids: list[str],
    record_index: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    missing = sorted(sample_id for sample_id in sample_ids if sample_id not in record_index)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{split_name} split contains sample_ids not present in original dataset: "
            f"{preview}{suffix}"
        )

    return [_normalize_example(sample_id, record_index[sample_id]) for sample_id in sample_ids]


def _tokenize_examples(
    examples: list[dict[str, str]],
    tokenizer: Any,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    tokenized_rows: list[dict[str, Any]] = []
    for example in examples:
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        encoded["labels"] = list(encoded["input_ids"])
        encoded["sample_id"] = example["sample_id"]
        tokenized_rows.append(encoded)
    return tokenized_rows


def load_instruction_datasets(config: dict[str, Any], tokenizer: Any) -> tuple[Any, Any | None]:
    """Load train/test datasets by joining split sample IDs to original JSON records."""
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    run_id = str(config.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Config must define 'run_id' for split-manifest loading")

    output_root = Path(str(config.get("output_root") or config.get("output_dir") or "outputs/runs"))
    splits_dir = output_root / run_id / "splits"

    dataset_path_raw = data_cfg.get("path") or config.get("dataset_path")
    if not dataset_path_raw:
        raise ValueError("Config must define original dataset JSON path at 'data.path' or 'dataset_path'")
    dataset_path = Path(str(dataset_path_raw))

    max_seq_len = int(train_cfg.get("max_seq_len", 512))

    records = _read_dataset_records(dataset_path)
    record_index = _build_record_index(records)

    train_ids = _read_split_ids(splits_dir / "train.csv", split_name="train")
    test_ids = _read_split_ids(splits_dir / "test.csv", split_name="test")

    train_examples = _build_examples_for_split("train", train_ids, record_index)
    test_examples = _build_examples_for_split("test", test_ids, record_index) if test_ids else []

    datasets_mod = importlib.import_module("datasets")

    train_dataset = datasets_mod.Dataset.from_list(
        _tokenize_examples(train_examples, tokenizer=tokenizer, max_seq_len=max_seq_len)
    )
    eval_dataset = None
    if test_examples:
        eval_dataset = datasets_mod.Dataset.from_list(
            _tokenize_examples(test_examples, tokenizer=tokenizer, max_seq_len=max_seq_len)
        )

    return train_dataset, eval_dataset
