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
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps([
            {"sample_id": "s1", "prompt": "Prompt 1", "response": "Response 1"},
            {"sample_id": "s2", "prompt": "Prompt 2", "response": "Response 2"},
        ]),
        encoding="utf-8",
    )

    splits_dir = tmp_path / "runs" / "testrun" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("sample_id\ns1\ns2\n", encoding="utf-8")
    (splits_dir / "test.csv").write_text("sample_id\ns2\n", encoding="utf-8")
    config_path = tmp_path / "train_lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: test",
                "model_name_or_path: fake-model",
                "run_id: testrun",
                f"output_root: {tmp_path.as_posix()}/runs",
                "data:",
                f"  dataset_json_path: {dataset_path.as_posix()}",
                f"  train_manifest_path: {splits_dir.as_posix()}/train.csv",
                "  text_fields: [prompt, response]",
                "  prompt_field: prompt",
                "  response_field: response",
                "lora:",
                "  rank: 4",
                "  alpha: 8",
                "  dropout: 0.0",
                "training:",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  epochs: 1",
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
            self.state = type(
                "FakeState",
                (),
                {"to_dict": lambda self: {"global_step": 2, "log_history": [{"loss": 0.25}]}}
            )()

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


    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        @classmethod
        def from_list(cls, rows: list[dict[str, object]]) -> "FakeDataset":
            return cls(rows)

    class FakeDatasets:
        Dataset = FakeDataset
    def fake_import_module(name: str) -> object:
        mapping = {
            "transformers": FakeTransformers,
            "peft": FakePeft,
            "torch": FakeTorch,
            "datasets": FakeDatasets,
        }
        return mapping[name]

    monkeypatch.setattr(lora_train.importlib, "import_module", fake_import_module)

    run_dir = run_training(str(config_path))

    assert run_dir.exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "trainer_state.json").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "adapter" / "adapter_model.bin").exists()
    assert (run_dir / "tokenizer" / "tokenizer.json").exists()
