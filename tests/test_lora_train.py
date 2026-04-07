"""Tests for LoRA training entrypoint."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import lora_train
from src.training.lora_train import parse_args, run_training


def test_parse_args_reads_config_path() -> None:
    args = parse_args(["--config", "configs/train_lora.yaml"])
    assert args.config == "configs/train_lora.yaml"


def test_run_training_writes_expected_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_data = tmp_path / "train.jsonl"
    train_data.write_text(
        json.dumps({"text": "sample 1"}) + "\n" + json.dumps({"text": "sample 2"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "train_lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: test",
                "base_model_path: fake-model",
                f"output_dir: {tmp_path.as_posix()}/runs",
                f"train_data_path: {train_data.as_posix()}",
                "lora:",
                "  rank: 4",
                "  alpha: 8",
                "  dropout: 0.0",
                "training:",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  epochs: 1",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )

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

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            values = [1, 2, 3] if text else [0]
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

    class FakeTrainingArguments:
        def __init__(self, **_: object) -> None:
            pass

    class FakeTrainResult:
        metrics = {"train_runtime": 1.5, "train_loss": 0.25, "global_step": 2, "epoch": 1.0}

    class FakeTrainer:
        def __init__(self, **_: object) -> None:
            pass

        def train(self) -> FakeTrainResult:
            return FakeTrainResult()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoModelForCausalLM = FakeAutoModel
        TrainingArguments = FakeTrainingArguments
        Trainer = FakeTrainer

    class FakeTaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class FakePeft:
        TaskType = FakeTaskType

        class LoraConfig:
            def __init__(self, **_: object) -> None:
                pass

        @staticmethod
        def get_peft_model(model: FakeModel, _: object) -> FakeModel:
            return model

    def fake_import_module(name: str) -> object:
        mapping = {"transformers": FakeTransformers, "peft": FakePeft, "torch": FakeTorch}
        return mapping[name]

    monkeypatch.setattr(lora_train.importlib, "import_module", fake_import_module)

    run_dir = run_training(str(config_path))

    assert run_dir.exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "adapter" / "adapter_model.bin").exists()
