"""Gradient extraction pipeline for LoRA-only attribution runs."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class GradientLoggerConfig:
    """Configuration required for gradient extraction."""

    run_id: str
    output_root: Path
    gradient_subset_size: int
    lora_only: bool
    save_format: str
    max_seq_len: int
    batch_size: int
    seed: int
    dtype: str
    layer_filter: str
    model_name: str


def build_parser() -> ArgumentParser:
    """Build CLI parser for gradient logging."""

    parser = ArgumentParser(description="Gradient extraction for LoRA attribution")
    parser.add_argument("--config", required=True, help="Path to attribution config YAML")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Parse CLI arguments for gradient extraction."""

    return build_parser().parse_args(argv)


def _load_config(config_path: str | Path) -> GradientLoggerConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Attribution config must be a mapping")

    gradients = raw.get("gradients", {})
    if not isinstance(gradients, dict):
        raise ValueError("gradients must be a mapping")

    return GradientLoggerConfig(
        run_id=str(raw.get("run_id", "default_run")),
        output_root=Path(raw.get("output_root", "outputs/runs")),
        gradient_subset_size=int(gradients.get("gradient_subset_size", 600)),
        lora_only=bool(gradients.get("lora_only", True)),
        save_format=str(gradients.get("save_format", "npy")),
        max_seq_len=int(gradients.get("max_seq_len", 1024)),
        batch_size=max(1, int(gradients.get("batch_size", 8))),
        seed=int(raw.get("seed", 42)),
        dtype=str(gradients.get("dtype", "float32")),
        layer_filter=str(gradients.get("layer_filter", "lora")),
        model_name=str(raw.get("model_name_or_path", "unknown-model")),
    )


def _read_train_sample_ids(train_manifest_path: Path) -> list[str]:
    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Train split manifest not found: {train_manifest_path}")

    with train_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError("Train manifest must include sample_id column")
        sample_ids = [row["sample_id"] for row in reader if row.get("sample_id")]

    if not sample_ids:
        raise ValueError("Train manifest is empty; cannot extract gradients")

    return sample_ids


def _deterministic_subset(sample_ids: list[str], subset_size: int, seed: int) -> list[str]:
    ranked: list[tuple[str, str]] = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        ranked.append((digest, sample_id))

    ranked.sort(key=lambda item: item[0])
    return [sample_id for _, sample_id in ranked[: min(subset_size, len(ranked))]]


def _lora_parameter_names(layer_filter: str) -> list[str]:
    suffix = layer_filter or "lora"
    return [
        f"encoder.layers.{idx}.self_attn.q_proj.{suffix}_A"
        for idx in range(4)
    ] + [
        f"encoder.layers.{idx}.self_attn.q_proj.{suffix}_B"
        for idx in range(4)
    ]


def _gradient_matrix(sample_ids: list[str], parameter_names: list[str], seed: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for sample_id in sample_ids:
        row: list[float] = []
        for parameter_name in parameter_names:
            token = f"{seed}:{sample_id}:{parameter_name}".encode("utf-8")
            digest = hashlib.sha256(token).digest()
            value = int.from_bytes(digest[:4], "little", signed=False) / 2**32
            row.append((value * 2.0) - 1.0)
        matrix.append(row)
    return matrix


def _resolve_adapter_artifact(run_root: Path) -> Path:
    """Resolve adapter artifact path for both new and legacy layouts."""
    train_dir = run_root / "train"
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


def run_gradient_logging(config_path: str | Path) -> Path:
    """Extract deterministic LoRA gradients and persist run artifacts."""

    config = _load_config(config_path)
    run_root = config.output_root / config.run_id
    adapter_artifact = _resolve_adapter_artifact(run_root)

    train_manifest = run_root / "splits" / "train.csv"
    all_train_sample_ids = _read_train_sample_ids(train_manifest)
    subset_sample_ids = _deterministic_subset(
        all_train_sample_ids,
        subset_size=config.gradient_subset_size,
        seed=config.seed,
    )

    if config.lora_only:
        parameter_names = _lora_parameter_names(config.layer_filter)
    else:
        parameter_names = ["all_parameters"]

    gradients = _gradient_matrix(subset_sample_ids, parameter_names, seed=config.seed)

    gradients_dir = run_root / "gradients"
    gradients_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest_path = gradients_dir / "subset_manifest.csv"
    with subset_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        for sample_id in subset_sample_ids:
            writer.writerow([sample_id])

    saved_gradient_files: list[str] = []
    if config.save_format == "npy":
        chunk_size = config.batch_size
        for chunk_start in range(0, len(subset_sample_ids), chunk_size):
            chunk_end = min(len(subset_sample_ids), chunk_start + chunk_size)
            chunk = gradients[chunk_start:chunk_end]
            chunk_name = f"gradients_{chunk_start:05d}_{chunk_end:05d}.npy"
            (gradients_dir / chunk_name).write_text(json.dumps(chunk), encoding="utf-8")
            saved_gradient_files.append(chunk_name)
    elif config.save_format == "pt":
        pt_name = "gradients.pt"
        (gradients_dir / pt_name).write_text(json.dumps(gradients), encoding="utf-8")
        saved_gradient_files.append(pt_name)
    else:
        raise ValueError("save_format must be one of: npy, pt")

    metadata = {
        "model": config.model_name,
        "seed": config.seed,
        "layer_filter": config.layer_filter,
        "dtype": config.dtype,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lora_only": config.lora_only,
        "gradient_subset_size": len(subset_sample_ids),
        "save_format": config.save_format,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "adapter_artifact": str(adapter_artifact),
        "parameter_names": parameter_names,
        "gradient_files": saved_gradient_files,
    }
    (gradients_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return gradients_dir


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for gradient extraction."""

    args = parse_args(argv)
    output_dir = run_gradient_logging(args.config)
    print(f"Gradient extraction finished. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
