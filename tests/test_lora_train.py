"""Tests for LoRA training entrypoint."""

import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attribution.gradient_logger import run_gradient_logging
from src.training import lora_train
from src.training.lora_train import parse_args, run_training


@pytest.fixture
def tiny_dataset_and_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create tiny synthetic dataset/config artifacts for bounded training tests."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"sample_id": "s1", "prompt": "Prompt 1", "response": "Response 1"},
                {"sample_id": "s2", "prompt": "Prompt 2", "response": "Response 2"},
            ]
        ),
        encoding="utf-8",
    )

    run_id = "testrun"
    run_root = tmp_path / "runs" / run_id
    splits_dir = run_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ns1\ns2\n", encoding="utf-8")
    (splits_dir / "test.csv").write_text("sample_id\ns1\n", encoding="utf-8")
    (splits_dir / "shadow.csv").write_text("sample_id\ns2\n", encoding="utf-8")

    config_path = tmp_path / "train_lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: test",
                "model_name_or_path: fake-model",
                f"run_id: {run_id}",
                f"output_root: {(tmp_path / 'runs').as_posix()}",
                "data:",
                f"  dataset_json_path: {dataset_path.as_posix()}",
                f"  train_manifest_path: {splits_dir.as_posix()}/train.csv",
                f"  test_manifest_path: {splits_dir.as_posix()}/test.csv",
                f"  shadow_manifest_path: {splits_dir.as_posix()}/shadow.csv",
                "  text_fields: [prompt, response]",
                "  prompt_field: prompt",
                "  response_field: response",
                "lora:",
                "  rank: 4",
                "  alpha: 8",
                "  dropout: 0.1",
                "  bias: none",
                "training:",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  epochs: 1",
                "  max_steps: 2",
                "  gradient_accumulation_steps: 1",
                "  warmup_ratio: 0.0",
                "  weight_decay: 0.0",
                "  lr_scheduler_type: linear",
                "  logging_steps: 1",
                "  save_steps: 1",
                "  eval_strategy: 'no'",
                "  fp16: true",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )

    return dataset_path, run_root, config_path


@pytest.fixture
def patch_training_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy training imports with tiny, deterministic fakes."""

    class FakeTensor:
        def __init__(self, value: object) -> None:
            self.value = value

    class FakeTorch:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"
        long = "long"

        class utils:
            class data:
                class Dataset:
                    pass

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> FakeTensor:
            return FakeTensor((value, dtype))

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "<eos>"

        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            values = [1, 2, 3] if text else [0]
            max_length = kwargs.get("max_length")
            truncation = bool(kwargs.get("truncation"))
            padding = kwargs.get("padding")
            if truncation and isinstance(max_length, int):
                values = values[:max_length]
            if padding == "max_length" and isinstance(max_length, int):
                values = values + ([0] * max(0, max_length - len(values)))
            return {"input_ids": values, "attention_mask": [1] * len(values)}

        def save_pretrained(self, output: Path) -> None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeModel:
        def save_pretrained(self, output: Path) -> None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.bin").write_bytes(b"adapter")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_: str) -> FakeTokenizer:
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_: str, **__: object) -> FakeModel:
            return FakeModel()

    class FakeConfig:
        is_encoder_decoder = False
        architectures = ["Qwen2ForCausalLM"]

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_: str) -> FakeConfig:
            return FakeConfig()

    class FakeTrainingArguments:
        def __init__(self, **_: object) -> None:
            pass

    class FakeTrainResult:
        metrics = {
            "train_runtime": 1.5,
            "train_loss": 0.25,
            "global_step": 2,
            "epoch": 1.0,
            "train_steps_per_second": 3.0,
        }

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            if "tokenizer" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'tokenizer'")
            if "data_collator" not in kwargs:
                raise AssertionError("data_collator must be set explicitly")
            self.state = type(
                "FakeState",
                (),
                {"to_dict": lambda self: {"global_step": 2, "log_history": [{"loss": 0.25}]}},
            )()

        def train(self) -> FakeTrainResult:
            return FakeTrainResult()

    class FakeTransformers:
        __version__ = "4.47.0"
        AutoTokenizer = FakeAutoTokenizer
        AutoConfig = FakeAutoConfig
        AutoModelForCausalLM = FakeAutoModel
        TrainingArguments = FakeTrainingArguments
        Trainer = FakeTrainer

        class DataCollatorForLanguageModeling:
            def __init__(self, tokenizer: object, mlm: bool) -> None:
                self.tokenizer = tokenizer
                self.mlm = mlm

            def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
                max_len = max(len(feature["input_ids"]) for feature in features)
                input_ids = []
                attention_mask = []
                labels = []
                for feature in features:
                    tokens = list(feature["input_ids"])
                    mask = list(feature["attention_mask"])
                    pad = max_len - len(tokens)
                    input_ids.append(tokens + ([0] * pad))
                    attention_mask.append(mask + ([0] * pad))
                    labels.append(tokens + ([-100] * pad))
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

    class FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class FakePeft:
        __version__ = "0.13.0"
        TaskType = FakeTaskType

        class LoraConfig:
            def __init__(self, **_: object) -> None:
                pass

        @staticmethod
        def get_peft_model(model: FakeModel, _: object) -> FakeModel:
            return model

    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        @classmethod
        def from_list(cls, rows: list[dict[str, object]]) -> "FakeDataset":
            return cls(rows)

    class FakeDatasets:
        Dataset = FakeDataset

    class FakeAccelerate:
        __version__ = "1.1.0"

    def fake_import_module(name: str) -> object:
        mapping = {
            "transformers": FakeTransformers,
            "peft": FakePeft,
            "torch": FakeTorch,
            "datasets": FakeDatasets,
            "accelerate": FakeAccelerate,
        }
        return mapping[name]

    monkeypatch.setattr(lora_train.importlib, "import_module", fake_import_module)


def test_parse_args_reads_config_path() -> None:
    args = parse_args(["--config", "configs/train_lora.dev.yaml", "--final-report"])
    assert args.config == "configs/train_lora.dev.yaml"
    assert args.final_report is True


def test_build_trainer_kwargs_uses_processing_class_when_supported() -> None:
    class FakeTrainer:
        def __init__(self, *, processing_class: object | None = None, **kwargs: object) -> None:
            self.processing_class = processing_class
            self.kwargs = kwargs

    class FakeTransformers:
        Trainer = FakeTrainer

    tokenizer = object()
    kwargs = lora_train._build_trainer_kwargs(
        FakeTransformers,
        model=object(),
        training_args=object(),
        train_dataset=[],
        eval_dataset=[],
        tokenizer=tokenizer,
        data_collator=object(),
    )
    assert kwargs["processing_class"] is tokenizer
    assert "tokenizer" not in kwargs


def test_build_trainer_kwargs_omits_processing_class_when_not_supported() -> None:
    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTransformers:
        Trainer = FakeTrainer

    kwargs = lora_train._build_trainer_kwargs(
        FakeTransformers,
        model=object(),
        training_args=object(),
        train_dataset=[],
        eval_dataset=[],
        tokenizer=object(),
        data_collator=object(),
    )
    assert "processing_class" not in kwargs
    assert "tokenizer" not in kwargs


def test_run_training_writes_expected_artifacts(
    tiny_dataset_and_config: tuple[Path, Path, Path],
    patch_training_runtime: None,
) -> None:
    _, _, config_path = tiny_dataset_and_config

    run_dir = run_training(str(config_path))

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))

    assert run_dir.exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "trainer_state.json").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "adapter" / "adapter_model.bin").exists()
    assert (run_dir / "tokenizer" / "tokenizer.json").exists()

    assert metrics["train_runtime"] > 0
    assert metrics["train_loss"] > 0
    assert metrics["steps"] > 0
    assert metrics["epochs_completed"] > 0
    assert metrics["train_steps_per_second"] > 0
    assert metrics["padding_strategy"] in {"dynamic", "max_length"}
    assert metrics["padding_benchmark"]["benchmark_target"] == "max_length_baseline"
    assert metrics["padding_benchmark"]["observed_steps_per_second"] > 0
    assert metrics["time_config_path_validation_s"] >= 0
    assert metrics["time_model_tokenizer_load_s"] >= 0
    assert metrics["time_dataset_load_tokenization_s"] >= 0
    assert metrics["time_trainer_train_loop_s"] >= 0
    assert metrics["timing_breakdown_s"]["config_path_validation"] >= 0
    assert metrics["timing_breakdown_s"]["model_tokenizer_load"] >= 0
    assert metrics["timing_breakdown_s"]["dataset_load_tokenization"] >= 0
    assert metrics["timing_breakdown_s"]["trainer_train_loop"] >= 0

    assert params["lora_rank"] == 4
    assert params["lora_alpha"] == 8.0
    assert params["lora_dropout"] == 0.1
    assert params["lora_bias"] == "none"
    assert params["reproducibility"]["python_version"]
    assert params["reproducibility"]["package_versions"]["transformers"] == "4.47.0"
    assert params["reproducibility"]["package_versions"]["peft"] == "0.13.0"
    assert params["reproducibility"]["package_versions"]["accelerate"] == "1.1.0"
    assert params["reproducibility"]["cuda"]["cuda_available"] is False
    assert params["reproducibility"]["git"]["git_commit_hash"]
    assert "reproducibility" in yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))


def test_lora_training_outputs_are_gradient_logger_compatible(
    tiny_dataset_and_config: tuple[Path, Path, Path],
    patch_training_runtime: None,
    tmp_path: Path,
) -> None:
    _, run_root, config_path = tiny_dataset_and_config
    run_dir = run_training(str(config_path))

    attribution_config = tmp_path / "attribution.yaml"
    attribution_config.write_text(
        "\n".join(
            [
                "run_id: testrun",
                f"output_root: {(tmp_path / 'runs').as_posix()}",
                "model_name_or_path: fake-model",
                "gradients:",
                "  gradient_subset_size: 2",
                "  lora_only: true",
                "  save_format: npy",
                "  max_seq_len: 64",
                "  batch_size: 1",
                "  dtype: float32",
                "  layer_filter: lora",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )

    gradients_dir = run_gradient_logging(attribution_config)
    metadata = json.loads((gradients_dir / "metadata.json").read_text(encoding="utf-8"))

    assert run_dir == run_root / "train"
    assert (gradients_dir / "subset_manifest.csv").exists()
    assert metadata["lora_only"] is True
    assert metadata["adapter_artifact"] == str(run_dir / "adapter")
    assert metadata["gradient_subset_size"] == 2
    assert metadata["parameter_names"]


def test_training_profiles_use_ungated_model_id() -> None:
    dev_config = yaml.safe_load(Path("configs/train_lora.dev.yaml").read_text(encoding="utf-8"))
    final_config = yaml.safe_load(Path("configs/train_lora.final.yaml").read_text(encoding="utf-8"))
    assert dev_config["model_name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert final_config["model_name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert dev_config["profile"] == "dev"
    assert final_config["profile"] == "final"


def test_final_report_guard_rejects_non_final_profile() -> None:
    with pytest.raises(ValueError, match="profile: final"):
        lora_train._validate_training_config(
            {
                "profile": "dev",
                "model_name_or_path": "fake",
                "output_root": "outputs/runs",
                "run_id": "run",
                "data": {
                    "dataset_json_path": "data/raw/alpaca_data.json",
                    "train_manifest_path": "outputs/runs/default_run/splits/train.csv",
                    "test_manifest_path": "outputs/runs/default_run/splits/test.csv",
                    "shadow_manifest_path": "outputs/runs/default_run/splits/shadow.csv",
                    "text_fields": ["prompt", "response"],
                    "prompt_field": "prompt",
                    "response_field": "response",
                },
                "training": {
                    "gradient_accumulation_steps": 1,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.0,
                    "lr_scheduler_type": "linear",
                    "logging_steps": 1,
                    "save_steps": 1,
                    "eval_strategy": "no",
                    "fp16": True,
                },
            },
            config_path="configs/train_lora.dev.yaml",
            enforce_final_report=True,
        )


def test_model_compatibility_guard_rejects_non_causal_architecture() -> None:
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(_: str) -> object:
            return object()

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["T5ForConditionalGeneration"]

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_: str) -> FakeConfig:
            return FakeConfig()

    class FakeTransformers:
        AutoTokenizer = FakeTokenizer
        AutoConfig = FakeAutoConfig

    config = {"model_name_or_path": "fake-model", "tokenizer_name_or_path": None}

    with pytest.raises(ValueError, match="decoder-only causal LM"):
        lora_train._validate_model_compatibility(config, FakeTransformers)


def test_explicit_offload_requested_detection() -> None:
    assert lora_train._explicit_offload_requested(None) is False
    assert lora_train._explicit_offload_requested("auto") is False
    assert lora_train._explicit_offload_requested("cuda:0") is False
    assert lora_train._explicit_offload_requested("cpu") is True
    assert lora_train._explicit_offload_requested({"": "cuda:0"}) is False
    assert lora_train._explicit_offload_requested({"": "cpu"}) is True


def test_resolve_model_loading_strategy_prefers_single_gpu_cuda_without_offload() -> None:
    strategy = lora_train._resolve_model_loading_strategy(
        cuda_available=True,
        cuda_device_count=1,
        quantized_load=False,
        device_map="auto",
    )

    assert strategy["single_gpu_training"] is True
    assert strategy["explicit_offload"] is False
    assert strategy["use_single_gpu_cuda_placement"] is True


def test_resolve_model_loading_strategy_respects_explicit_offload() -> None:
    strategy = lora_train._resolve_model_loading_strategy(
        cuda_available=True,
        cuda_device_count=1,
        quantized_load=False,
        device_map="cpu",
    )

    assert strategy["single_gpu_training"] is True
    assert strategy["explicit_offload"] is True
    assert strategy["use_single_gpu_cuda_placement"] is False


def test_preflight_validate_batch_collation_raises_clear_error() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> object:
            raise RuntimeError("Unable to create tensor")

    class FakeCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
            return {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
                "labels": [[1, 2], [3]],
            }

    class FakeDataset:
        rows = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]

    with pytest.raises(ValueError, match="Preflight tensor creation failed"):
        lora_train.preflight_validate_batch_collation(
            dataset=FakeDataset(),
            data_collator=FakeCollator(),
            torch_mod=FakeTorch,
            batch_size=1,
            split_name="train",
        )


def test_preflight_validate_batch_collation_detects_missing_labels() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> object:
            return {"value": value, "dtype": dtype}

    class MissingLabelsCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
            return {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
            }

    class FakeDataset:
        rows = [{"input_ids": [1], "attention_mask": [1]}]

    with pytest.raises(ValueError, match="missing required batch key: labels"):
        lora_train.preflight_validate_batch_collation(
            dataset=FakeDataset(),
            data_collator=MissingLabelsCollator(),
            torch_mod=FakeTorch,
            batch_size=1,
            split_name="train",
        )


def test_preflight_validate_batch_collation_accepts_mapping_batch_encoding_like() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> object:
            return {"value": value, "dtype": dtype}

    class FakeBatchEncoding(dict):
        """Minimal BatchEncoding-like mapping."""

    class MappingCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> FakeBatchEncoding:
            return FakeBatchEncoding(
                {
                    "input_ids": [feature["input_ids"] for feature in features],
                    "attention_mask": [feature["attention_mask"] for feature in features],
                    "labels": [feature["input_ids"] for feature in features],
                }
            )

    class FakeDataset:
        rows = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]

    lora_train.preflight_validate_batch_collation(
        dataset=FakeDataset(),
        data_collator=MappingCollator(),
        torch_mod=FakeTorch,
        batch_size=1,
        split_name="train",
    )


def test_preflight_validate_batch_collation_accepts_plain_dict() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> object:
            return {"value": value, "dtype": dtype}

    class DictCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
            return {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
                "labels": [feature["input_ids"] for feature in features],
            }

    class FakeDataset:
        rows = [{"input_ids": [1], "attention_mask": [1]}]

    lora_train.preflight_validate_batch_collation(
        dataset=FakeDataset(),
        data_collator=DictCollator(),
        torch_mod=FakeTorch,
        batch_size=1,
        split_name="train",
    )


def test_preflight_validate_batch_collation_rejects_non_mapping() -> None:
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: object, dtype: object | None = None) -> object:
            return {"value": value, "dtype": dtype}

    class NonMappingCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> list[dict[str, list[int]]]:
            return features

    class FakeDataset:
        rows = [{"input_ids": [1], "attention_mask": [1]}]

    with pytest.raises(
        ValueError, match=r"expected Mapping \(e\.g\., dict or BatchEncoding\)"
    ):
        lora_train.preflight_validate_batch_collation(
            dataset=FakeDataset(),
            data_collator=NonMappingCollator(),
            torch_mod=FakeTorch,
            batch_size=1,
            split_name="train",
        )
