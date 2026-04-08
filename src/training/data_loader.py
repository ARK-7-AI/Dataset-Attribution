"""Dataset loading utilities for LoRA instruction tuning."""

from __future__ import annotations

import csv
import difflib
import importlib
import json
from pathlib import Path
from typing import Any

from src.data.normalize import normalize_records


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def _is_under_src(path: Path) -> bool:
    normalized = Path(*path.parts)
    return any(part == "src" for part in normalized.parts)


def _reject_src_data_path(path: Path, *, label: str) -> None:
    if _is_under_src(path):
        raise ValueError(
            f"{label} must not live under 'src/'. Move data files to 'data/raw/' and point config there: '{path}'."
        )


def _closest_existing_path_suggestion(path: Path) -> str | None:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent

    if not probe.exists():
        return None

    if probe.is_file():
        return str(probe)

    candidate_names = [candidate.name for candidate in probe.iterdir()]
    match = difflib.get_close_matches(path.name, candidate_names, n=1, cutoff=0.4)
    if match:
        return str(probe / match[0])
    return str(probe)


def _raise_missing_path_error(path: Path, *, label: str) -> None:
    suggestion = _closest_existing_path_suggestion(path)
    if suggestion:
        raise FileNotFoundError(
            f"{label} not found: '{path}'. Closest existing path suggestion: '{suggestion}'."
        )
    raise FileNotFoundError(f"{label} not found: '{path}'.")


def preflight_validate_data_paths(config: dict[str, Any]) -> dict[str, Path | None]:
    """Resolve and validate all configured dataset/split paths before training."""
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Missing required 'data' config mapping")

    dataset_path_raw = (
        data_cfg.get("dataset_json_path")
        or data_cfg.get("dataset_path")
        or data_cfg.get("path")
        or config.get("dataset_path")
    )
    if not dataset_path_raw:
        raise ValueError("Config must define original dataset JSON path at 'data.dataset_json_path'")
    dataset_path = Path(str(dataset_path_raw))
    _reject_src_data_path(dataset_path, label="Dataset JSON")
    if not dataset_path.exists():
        _raise_missing_path_error(dataset_path, label="Dataset JSON")

    train_manifest_raw = data_cfg.get("train_manifest_path")
    if not train_manifest_raw:
        raise ValueError("Config must define 'data.train_manifest_path'")
    train_manifest_path = Path(str(train_manifest_raw))
    _reject_src_data_path(train_manifest_path, label="Train manifest")
    if not train_manifest_path.exists():
        _raise_missing_path_error(train_manifest_path, label="Train manifest")

    test_manifest_raw = data_cfg.get("test_manifest_path") or data_cfg.get("eval_manifest_path")
    test_manifest_path = Path(str(test_manifest_raw)) if test_manifest_raw else None
    if test_manifest_path:
        _reject_src_data_path(test_manifest_path, label="Test manifest")
    if test_manifest_path and not test_manifest_path.exists():
        _raise_missing_path_error(test_manifest_path, label="Test manifest")

    shadow_manifest_raw = data_cfg.get("shadow_manifest_path")
    shadow_manifest_path = Path(str(shadow_manifest_raw)) if shadow_manifest_raw else None
    if shadow_manifest_path:
        _reject_src_data_path(shadow_manifest_path, label="Shadow manifest")
    if shadow_manifest_path and not shadow_manifest_path.exists():
        _raise_missing_path_error(shadow_manifest_path, label="Shadow manifest")

    return {
        "dataset_path": dataset_path,
        "train_manifest_path": train_manifest_path,
        "test_manifest_path": test_manifest_path,
        "shadow_manifest_path": shadow_manifest_path,
    }


def _read_dataset_records(dataset_path: Path, *, dataset_name: str) -> list[dict[str, Any]]:
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

    if not rows:
        raise ValueError("Original dataset is empty")

    return normalize_records(list(rows), dataset_name=dataset_name)



def _persist_normalized_snapshot(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _normalize_example(
    sample_id: str,
    record: dict[str, Any],
    *,
    text_fields: list[str],
    prompt_field: str,
    input_field: str | None,
    response_field: str,
    response_fallback_fields: list[str],
) -> dict[str, str]:
    prompt = str(record.get(prompt_field) or "").strip()
    prompt_input = str(record.get(input_field) or "").strip() if input_field else ""
    response = str(record.get(response_field) or "").strip()
    tested_response_fields = [response_field, *response_fallback_fields]
    if not response:
        for fallback_field in response_fallback_fields:
            candidate = str(record.get(fallback_field) or "").strip()
            if candidate:
                response = candidate
                break
    fallback_text = ""
    for field in text_fields:
        value = str(record.get(field) or "").strip()
        if value:
            fallback_text = value
            break

    if not prompt and fallback_text:
        prompt = fallback_text

    if not prompt:
        raise ValueError(
            f"Sample '{sample_id}' is missing prompt text (prompt_field='{prompt_field}')"
        )
    if not response:
        raise ValueError(
            f"Sample '{sample_id}' is missing target answer "
            f"(tested_fields={tested_response_fields})"
        )

    if prompt_input:
        prompt = f"{prompt}\n\nInput:\n{prompt_input}"

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
    *,
    text_fields: list[str],
    prompt_field: str,
    input_field: str | None,
    response_field: str,
    response_fallback_fields: list[str],
) -> list[dict[str, str]]:
    missing = sorted(sample_id for sample_id in sample_ids if sample_id not in record_index)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{split_name} split contains sample_ids not present in original dataset: "
            f"{preview}{suffix}"
        )

    return [
        _normalize_example(
            sample_id,
            record_index[sample_id],
            text_fields=text_fields,
            prompt_field=prompt_field,
            input_field=input_field,
            response_field=response_field,
            response_fallback_fields=response_fallback_fields,
        )
        for sample_id in sample_ids
    ]


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
    resolved_paths = preflight_validate_data_paths(config)
    dataset_path = resolved_paths["dataset_path"]
    train_manifest_path = resolved_paths["train_manifest_path"]
    eval_manifest_path = resolved_paths["test_manifest_path"]

    text_fields = [str(field).strip() for field in data_cfg.get("text_fields", []) if str(field).strip()]
    prompt_field = str(data_cfg.get("prompt_field") or "").strip()
    input_field = str(data_cfg.get("input_field") or "").strip() or None
    response_field = str(data_cfg.get("response_field") or "").strip()
    response_fallback_fields = [
        str(field).strip() for field in data_cfg.get("response_fallback_fields", []) if str(field).strip()
    ]
    normalize_response_key = bool(data_cfg.get("normalize_response_key", False))

    max_seq_len = int(train_cfg.get("max_seq_len", 512))

    dataset_name = str(data_cfg.get("dataset_name") or dataset_path.stem).strip()
    records = _read_dataset_records(dataset_path, dataset_name=dataset_name)
    if normalize_response_key:
        for record in records:
            response = str(record.get(response_field) or "").strip()
            if not response:
                for fallback_field in response_fallback_fields:
                    candidate = str(record.get(fallback_field) or "").strip()
                    if candidate:
                        response = candidate
                        break
            record["response"] = response

    normalized_snapshot_path_raw = data_cfg.get("normalized_snapshot_path")
    if normalized_snapshot_path_raw:
        _persist_normalized_snapshot(records, Path(str(normalized_snapshot_path_raw)))
    record_index = _build_record_index(records)

    train_ids = _read_split_ids(train_manifest_path, split_name="train")
    test_ids = _read_split_ids(eval_manifest_path, split_name="test") if eval_manifest_path else []

    train_examples = _build_examples_for_split(
        "train",
        train_ids,
        record_index,
        text_fields=text_fields,
        prompt_field=prompt_field,
        input_field=input_field,
        response_field=response_field,
        response_fallback_fields=response_fallback_fields,
    )
    test_examples = (
        _build_examples_for_split(
            "test",
            test_ids,
            record_index,
            text_fields=text_fields,
            prompt_field=prompt_field,
            input_field=input_field,
            response_field=response_field,
            response_fallback_fields=response_fallback_fields,
        )
        if test_ids
        else []
    )

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
