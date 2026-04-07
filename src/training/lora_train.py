"""LoRA training entrypoint module."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import yaml

from src.training.data_loader import load_instruction_datasets


def build_parser() -> ArgumentParser:
    """Build CLI parser for the LoRA training entrypoint."""
    parser = ArgumentParser(description="LoRA training entrypoint")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    """Parse CLI arguments for training."""
    return build_parser().parse_args(argv)


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML mapping.")

    return data


def _build_run_id() -> str:
    """Build a unique run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _get_config_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract resolved training parameters from config."""
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_dir = config.get("output_dir") or config.get("output_root") or "outputs/runs"

    return {
        "config_path": config.get("config_path"),
        "experiment_name": config.get("experiment_name", "lora_run"),
        "base_model_path": config.get("base_model_path")
        or config.get("model_name_or_path", ""),
        "output_dir": str(output_dir),
        "run_id": config.get("run_id"),
        "train_data_path": data_cfg.get("train_path")
        or config.get("train_data_path")
        or config.get("prepared_train_path"),
        "eval_data_path": data_cfg.get("eval_path")
        or config.get("eval_data_path")
        or config.get("prepared_eval_path"),
        "lora_rank": int(lora_cfg.get("rank", 8)),
        "lora_alpha": float(lora_cfg.get("alpha", 16)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.0)),
        "lora_target_modules": lora_cfg.get("target_modules"),
        "lora_bias": str(lora_cfg.get("bias", "none")),
        "max_seq_len": int(train_cfg.get("max_seq_len", 512)),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
        "epochs": int(train_cfg.get("epochs", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "eval_steps": int(train_cfg.get("eval_steps", 100)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.0)),
        "device_map": config.get("device_map", "auto"),
        "torch_dtype": config.get("torch_dtype", "auto"),
        "seed": int(config.get("seed", 42)),
    }


def _persist_run_outputs(
    run_dir: Path,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Persist train artifacts for a run and return run directory."""
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    params_path = train_dir / "params.json"
    metrics_path = train_dir / "metrics.json"

    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return train_dir


def run_training(config_path: str) -> Path:
    """Run LoRA training flow and return the train artifact directory."""
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")

    config = _load_config(config_path)
    config["config_path"] = str(config_path)
    params = _get_config_params(config)
    run_id = str(params["run_id"] or _build_run_id())
    run_dir = Path(str(params["output_dir"])) / run_id
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    if not params["base_model_path"]:
        raise ValueError("base_model_path (or model_name_or_path) must be set in config")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(params["base_model_path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(str(params["torch_dtype"]).lower(), "auto")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(params["base_model_path"]),
        device_map=params["device_map"],
        torch_dtype=resolved_dtype,
    )

    lora_config = peft.LoraConfig(
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        bias=params["lora_bias"],
        task_type=peft.TaskType.CAUSAL_LM,
        target_modules=params["lora_target_modules"],
    )
    model = peft.get_peft_model(model, lora_config)

    train_dataset, eval_dataset = load_instruction_datasets(config=config, tokenizer=tokenizer)

    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = transformers.TrainingArguments(
        output_dir=str(train_dir / "checkpoints"),
        num_train_epochs=float(params["epochs"]),
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        learning_rate=params["learning_rate"],
        logging_steps=params["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=params["eval_steps"],
        save_steps=params["save_steps"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        seed=params["seed"],
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    training_metrics = train_result.metrics if train_result and train_result.metrics else {}

    adapter_dir = train_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(train_dir / "tokenizer")

    metrics = {
        "train_runtime": float(training_metrics.get("train_runtime", 0.0)),
        "train_loss": float(training_metrics.get("train_loss", 0.0)),
        "steps": int(training_metrics.get("global_step", 0)),
        "epochs_completed": float(training_metrics.get("epoch", params["epochs"])),
    }

    _persist_run_outputs(run_dir=run_dir, params=params, metrics=metrics)
    return train_dir


def main(argv: Sequence[str] | None = None) -> None:
    """Run LoRA training workflow."""
    args = parse_args(argv)
    run_dir = run_training(args.config)
    print(f"LoRA training finished. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
