#!/usr/bin/env python
"""Gate final training runs with objective pass/fail criteria."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from src.training.final_run_check import FinalRunThresholds, evaluate_final_run


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate final-run training checklist")
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Path to outputs/runs/<run_id>/train directory",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=100,
        help="Minimum required completed optimizer steps",
    )
    parser.add_argument(
        "--min-epochs",
        type=float,
        default=1.0,
        help="Minimum required completed epochs",
    )
    parser.add_argument(
        "--max-final-loss",
        type=float,
        default=3.0,
        help="Maximum acceptable final train_loss",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="Optional output path for JSON result (default: <train_dir>/final_run_check.json)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    thresholds = FinalRunThresholds(
        min_steps=args.min_steps,
        min_epochs=args.min_epochs,
        max_final_loss=args.max_final_loss,
    )
    result = evaluate_final_run(args.train_dir, thresholds=thresholds)

    result_path = Path(args.result_path) if args.result_path else Path(args.train_dir) / "final_run_check.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print(f"Final run gate: {'PASS' if result.passed else 'FAIL'}")
    for check in result.checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f" - [{status}] {check['name']}: {check['detail']}")
    print(f"Saved result to: {result_path}")

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
