"""LoRA training entrypoint module."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Sequence
from uuid import uuid4

import yaml

from src.training.data_loader import (
    load_instruction_datasets,
    preflight_validate_data_paths,
    validate_trainer_dataset,
)


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



def _validate_training_config(config: dict[str, Any]) -> None:
    """Validate required training config fields and fail fast on unsafe defaults."""

    def _require_non_empty(root: dict[str, Any], key: str, *, label: str | None = None) -> Any:
        value = root.get(key)
        name = label or key
        if value is None:
            raise ValueError(f"Missing required config field: '{name}'")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"Config field '{name}' must not be empty")
        return value

    model_name = _require_non_empty(config, "model_name_or_path")
    if not isinstance(model_name, str):
        raise ValueError("'model_name_or_path' must be a string")

    output_root = _require_non_empty(config, "output_root")
    if not isinstance(output_root, str):
        raise ValueError("'output_root' must be a string")

    run_id = config.get("run_id")
    auto_run_id = bool(config.get("auto_run_id", False))
    if run_id is None and not auto_run_id:
        raise ValueError("Set either 'run_id' or 'auto_run_id: true'")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("'run_id' must be a non-empty string when provided")

    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Missing required 'data' config mapping")

    _require_non_empty(data_cfg, "dataset_json_path", label="data.dataset_json_path")
    _require_non_empty(data_cfg, "train_manifest_path", label="data.train_manifest_path")
    if not data_cfg.get("test_manifest_path") and not data_cfg.get("eval_manifest_path"):
        raise ValueError("Missing required config field: 'data.test_manifest_path' (or legacy 'data.eval_manifest_path')")
    _require_non_empty(data_cfg, "shadow_manifest_path", label="data.shadow_manifest_path")

    text_fields = _require_non_empty(data_cfg, "text_fields", label="data.text_fields")
    if not isinstance(text_fields, list) or not text_fields or any(
        not isinstance(field, str) or not field.strip() for field in text_fields
    ):
        raise ValueError("'data.text_fields' must be a non-empty list of field names")

    _require_non_empty(data_cfg, "prompt_field", label="data.prompt_field")
    _require_non_empty(data_cfg, "response_field", label="data.response_field")
    response_fallback_fields = data_cfg.get("response_fallback_fields", [])
    if response_fallback_fields is not None:
        if not isinstance(response_fallback_fields, list) or any(
            not isinstance(field, str) or not field.strip() for field in response_fallback_fields
        ):
            raise ValueError("'data.response_fallback_fields' must be a list of non-empty field names")

    train_cfg = config.get("training")
    if not isinstance(train_cfg, dict):
        raise ValueError("Missing required 'training' config mapping")

    required_training_fields = [
        "gradient_accumulation_steps",
        "warmup_ratio",
        "weight_decay",
        "lr_scheduler_type",
        "logging_steps",
        "save_steps",
        "eval_strategy",
    ]
    for field in required_training_fields:
        _require_non_empty(train_cfg, field, label=f"training.{field}")

    if "fp16" not in train_cfg and "bf16" not in train_cfg:
        raise ValueError("Set at least one precision field: 'training.fp16' or 'training.bf16'")

    fp16 = bool(train_cfg.get("fp16", False))
    bf16 = bool(train_cfg.get("bf16", False))
    if fp16 and bf16:
        raise ValueError("Only one precision mode can be enabled: set either fp16 or bf16")

    if "max_steps" in train_cfg and train_cfg.get("max_steps") is not None:
        max_steps = int(train_cfg["max_steps"])
        if max_steps <= 0:
            raise ValueError("'training.max_steps' must be > 0 when provided")

    eval_strategy = str(train_cfg.get("eval_strategy", "")).lower()
    if eval_strategy not in {"no", "steps", "epoch"}:
        raise ValueError("'training.eval_strategy' must be one of: no, steps, epoch")


def _build_run_id() -> str:
    """Build a unique run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _get_config_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract resolved training parameters from config."""
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    output_dir = config.get("output_root") or config.get("output_dir")

    return {
        "config_path": config.get("config_path"),
        "experiment_name": config.get("experiment_name", "lora_run"),
        "base_model_path": config.get("model_name_or_path") or config.get("base_model_path", ""),
        "tokenizer_name_or_path": config.get("tokenizer_name_or_path") or config.get("model_name_or_path") or config.get("base_model_path", ""),
        "output_dir": str(output_dir),
        "run_id": config.get("run_id"),
        "dataset_json_path": data_cfg.get("dataset_json_path") or data_cfg.get("path") or config.get("dataset_path"),
        "train_manifest_path": data_cfg.get("train_manifest_path"),
        "test_manifest_path": data_cfg.get("test_manifest_path") or data_cfg.get("eval_manifest_path"),
        "shadow_manifest_path": data_cfg.get("shadow_manifest_path"),
        "text_fields": data_cfg.get("text_fields", []),
        "prompt_field": data_cfg.get("prompt_field"),
        "input_field": data_cfg.get("input_field"),
        "response_field": data_cfg.get("response_field"),
        "response_fallback_fields": data_cfg.get("response_fallback_fields", []),
        "normalize_response_key": bool(data_cfg.get("normalize_response_key", False)),
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
        "warmup_ratio": float(train_cfg.get("warmup_ratio")),
        "max_steps": int(train_cfg.get("max_steps", -1)) if train_cfg.get("max_steps") is not None else -1,
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type")),
        "fp16": bool(train_cfg.get("fp16", False)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "eval_strategy": str(train_cfg.get("eval_strategy", "no")),
        "device_map": config.get("device_map", "auto"),
        "torch_dtype": config.get("torch_dtype", "auto"),
        "load_in_4bit": bool(config.get("load_in_4bit", False)),
        "load_in_8bit": bool(config.get("load_in_8bit", False)),
        "seed": int(config.get("seed", 42)),
    }

def _validate_model_compatibility(config: dict[str, Any], transformers: Any) -> None:
    """Validate tokenizer/model compatibility before launching training."""
    base_model_path = config.get("model_name_or_path") or config.get("base_model_path")
    tokenizer_path = config.get("tokenizer_name_or_path") or base_model_path

    if not base_model_path:
        raise ValueError("base_model_path (or model_name_or_path) must be set in config")

    try:
        transformers.AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as exc:  # pragma: no cover - exercised with fakes in tests
        raise ValueError(
            f"Unsupported tokenizer for '{tokenizer_path}'. Ensure the tokenizer is accessible and valid."
        ) from exc

    try:
        model_config = transformers.AutoConfig.from_pretrained(str(base_model_path))
    except Exception as exc:  # pragma: no cover - exercised with fakes in tests
        raise ValueError(
            f"Unable to load model config for '{base_model_path}'. Ensure the model id/path is valid."
        ) from exc

    is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
    architectures = [str(arch) for arch in (getattr(model_config, "architectures", None) or [])]
    is_causal_architecture = any("CausalLM" in arch for arch in architectures)
    if is_encoder_decoder or not is_causal_architecture:
        arch_label = ", ".join(architectures) if architectures else "unknown"
        raise ValueError(
            "Model compatibility check failed: expected a decoder-only causal LM, "
            f"but got architectures=[{arch_label}] and is_encoder_decoder={is_encoder_decoder}."
        )


def _resolve_package_version(module: Any) -> str:
    """Return package version string when available."""
    return str(getattr(module, "__version__", "unknown"))


def _build_model_device_summary(model: Any) -> str:
    """Build a compact model-device summary string for logging."""
    try:
        parameter_devices = sorted({str(param.device) for param in model.parameters()})
    except Exception:
        parameter_devices = ["unknown"]

    try:
        first_param_device = str(next(model.parameters()).device)
    except StopIteration:
        first_param_device = "none"
    except Exception:
        first_param_device = "unknown"

    return (
        f"first_param_device={first_param_device} "
        f"unique_parameter_devices={parameter_devices}"
    )


def _build_trainer_kwargs(
    transformers: Any,
    *,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    tokenizer: Any,
    data_collator: Any,
) -> dict[str, Any]:
    """Build Trainer kwargs that are compatible across Transformers versions."""
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    trainer_signature = inspect.signature(transformers.Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer

    return kwargs


def _persist_run_outputs(
    run_dir: Path,
    params: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    """Persist train artifacts for a run and return run directory."""
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    params_path = train_dir / "params.json"
    metrics_path = train_dir / "metrics.json"
    trainer_state_path = train_dir / "trainer_state.json"
    resolved_config_path = train_dir / "resolved_config.yaml"

    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer_state_payload = trainer_state if trainer_state is not None else {}
    trainer_state_path.write_text(json.dumps(trainer_state_payload, indent=2), encoding="utf-8")
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    return train_dir


def run_training(config_path: str) -> Path:
    """Run LoRA training flow and return the train artifact directory."""
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    accelerate = importlib.import_module("accelerate")

    print(
        "[startup] effective_versions "
        f"transformers={_resolve_package_version(transformers)} "
        f"accelerate={_resolve_package_version(accelerate)} "
        f"peft={_resolve_package_version(peft)}"
    )

    config = _load_config(config_path)
    config["config_path"] = str(config_path)
    _validate_training_config(config)
    params = _get_config_params(config)
    run_id = str(params["run_id"] or _build_run_id())
    params["run_id"] = run_id
    config["run_id"] = run_id
    data_cfg = config.get("data", {})
    for path_key in ("train_manifest_path", "test_manifest_path", "eval_manifest_path", "shadow_manifest_path"):
        path_value = data_cfg.get(path_key)
        if isinstance(path_value, str):
            data_cfg[path_key] = path_value.replace("<run_id>", run_id)

    preflight_validate_data_paths(config)

    run_dir = Path(str(params["output_dir"])) / run_id
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    compatibility_start = time.perf_counter()
    print("[timing] phase=compatibility_checks status=start")
    _validate_model_compatibility(config, transformers)
    compatibility_elapsed_s = time.perf_counter() - compatibility_start
    print(
        "[timing] phase=compatibility_checks status=end "
        f"elapsed_s={compatibility_elapsed_s:.3f}"
    )

    model_load_start = time.perf_counter()
    print("[timing] phase=model_load status=start")

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(params["tokenizer_name_or_path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(str(params["torch_dtype"]).lower(), "auto")

    if params["load_in_4bit"] and params["load_in_8bit"]:
        raise ValueError("Only one quantized load mode can be enabled: load_in_4bit or load_in_8bit")

    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
    }
    torch_cuda = getattr(torch, "cuda", None)
    cuda_available = bool(torch_cuda and torch_cuda.is_available())
    cuda_device_count = int(torch_cuda.device_count()) if cuda_available else 0
    quantized_load = bool(params["load_in_4bit"] or params["load_in_8bit"])
    single_gpu_non_quantized = cuda_available and cuda_device_count == 1 and not quantized_load

    if not single_gpu_non_quantized:
        model_load_kwargs["device_map"] = params["device_map"]
    if params["load_in_4bit"]:
        model_load_kwargs["load_in_4bit"] = True
    if params["load_in_8bit"]:
        model_load_kwargs["load_in_8bit"] = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(params["base_model_path"]),
        **model_load_kwargs,
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
    if single_gpu_non_quantized:
        model = model.to("cuda")
    model_load_elapsed_s = time.perf_counter() - model_load_start
    print(
        "[timing] phase=model_load status=end "
        f"elapsed_s={model_load_elapsed_s:.3f}"
    )

    tokenize_start = time.perf_counter()
    print("[timing] phase=dataset_load_normalize_tokenize status=start")
    train_dataset, eval_dataset = load_instruction_datasets(config=config, tokenizer=tokenizer)
    padding_strategy = str(config.get("data", {}).get("padding", "max_length")).strip().lower()
    normalized_padding = "dynamic" if padding_strategy in {"dynamic", "longest"} else "max_length"
    validate_trainer_dataset(
        train_dataset,
        split_name="train",
        max_seq_len=params["max_seq_len"],
        padding=normalized_padding,
    )
    if eval_dataset is not None:
        validate_trainer_dataset(
            eval_dataset,
            split_name="eval",
            max_seq_len=params["max_seq_len"],
            padding=normalized_padding,
        )
    tokenize_elapsed_s = time.perf_counter() - tokenize_start
    print(
        "[timing] phase=dataset_load_normalize_tokenize status=end "
        f"elapsed_s={tokenize_elapsed_s:.3f}"
    )

    eval_strategy = str(params["eval_strategy"]).lower()
    if eval_dataset is None and eval_strategy != "no":
        raise ValueError("Eval strategy requires eval data. Set training.eval_strategy=no or provide eval manifest")

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
        max_steps=params["max_steps"],
        lr_scheduler_type=params["lr_scheduler_type"],
        fp16=params["fp16"],
        bf16=params["bf16"],
        seed=params["seed"],
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = transformers.Trainer(
        **_build_trainer_kwargs(
            transformers,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    )
    print(
        "[startup] model_device_summary "
        f"cuda_available={cuda_available} "
        f"cuda_device_count={cuda_device_count} "
        f"single_gpu_non_quantized={single_gpu_non_quantized} "
        f"quantized_load={quantized_load} "
        f"{_build_model_device_summary(model)}"
    )

    train_start = time.perf_counter()
    print("[timing] phase=trainer_train status=start")
    train_result = trainer.train()
    train_elapsed_s = time.perf_counter() - train_start
    print(
        "[timing] phase=trainer_train status=end "
        f"elapsed_s={train_elapsed_s:.3f}"
    )
    training_metrics = train_result.metrics if train_result and train_result.metrics else {}

    adapter_dir = train_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(train_dir / "tokenizer")

    metrics = {
        "train_runtime": float(training_metrics.get("train_runtime", 0.0)),
        "train_loss": float(training_metrics.get("train_loss", 0.0)),
        "steps": int(training_metrics.get("global_step", 0)),
        "epochs_completed": float(training_metrics.get("epoch", params["epochs"])),
        "time_compatibility_checks_s": float(compatibility_elapsed_s),
        "time_model_load_s": float(model_load_elapsed_s),
        "time_tokenize_s": float(tokenize_elapsed_s),
        "time_train_s": float(train_elapsed_s),
    }

    trainer_state_payload: dict[str, Any] | None = None
    trainer_state = getattr(trainer, "state", None)
    if trainer_state is not None:
        if hasattr(trainer_state, "to_dict"):
            trainer_state_payload = trainer_state.to_dict()
        else:
            log_history = getattr(trainer_state, "log_history", None)
            if log_history is not None:
                trainer_state_payload = {"log_history": log_history}

    _persist_run_outputs(
        run_dir=run_dir,
        params=params,
        metrics=metrics,
        config=config,
        trainer_state=trainer_state_payload,
    )
    return train_dir


def main(argv: Sequence[str] | None = None) -> None:
    """Run LoRA training workflow."""
    args = parse_args(argv)
    run_dir = run_training(args.config)
    print(f"LoRA training finished. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
