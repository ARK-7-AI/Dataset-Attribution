"""LoRA training entrypoint module."""

from argparse import ArgumentParser, Namespace
from typing import Sequence


def build_parser() -> ArgumentParser:
    """Build CLI parser for the LoRA training entrypoint."""
    parser = ArgumentParser(description="LoRA training entrypoint")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Parse CLI arguments for training."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run LoRA training workflow (placeholder)."""
    args = parse_args(argv)
    print(f"[placeholder] run LoRA training with config: {args.config}")


if __name__ == "__main__":
    main()
