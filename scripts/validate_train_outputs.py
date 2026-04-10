#!/usr/bin/env python
"""Validate LoRA train artifacts for downstream compatibility."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.training.validate_outputs import validate_training_outputs


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Validate LoRA train outputs")
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Path to outputs/runs/<run_id>/train directory",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifacts = validate_training_outputs(Path(args.train_dir))
    print(f"Validated train artifacts: {artifacts.train_dir}")


if __name__ == "__main__":
    main()
