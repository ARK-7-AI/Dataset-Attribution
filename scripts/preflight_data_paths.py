#!/usr/bin/env python3
"""Preflight checks for dataset and split manifest paths."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.data_loader import preflight_validate_data_paths


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return payload


def main() -> int:
    parser = ArgumentParser(description="Validate dataset and split paths before training")
    parser.add_argument("--config", required=True, help="Training config path (YAML)")
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    resolved = preflight_validate_data_paths(config)

    print("Preflight checks passed:")
    for key, value in resolved.items():
        if value is not None:
            print(f"- {key}: {value}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Preflight check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
