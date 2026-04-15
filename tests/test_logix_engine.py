"""Tests for LogIX engine integration artifacts and CLI behavior."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.attribution.logix_engine import (
    _INITIALIZED_LOGIX_MODULE_IDS,
    LogIXEngineConfig,
    execute_logix,
    run_logix_engine,
    setup_logix,
)


class _FakeLogIX:
    @staticmethod
    def init(**kwargs):
        return {"status": "initialized", **kwargs}

    @staticmethod
    def setup(**kwargs):
        return {"session": "fake", **kwargs}

    @staticmethod
    def run(**kwargs):
        sample_ids = kwargs.get("sample_ids", [])
        return {
            "status": "ok",
            "num_ranked": len(sample_ids),
            "top_sample_id": sample_ids[0] if sample_ids else None,
        }


class _OrderedLogIX:
    events: list[str] = []
    init_payload: dict[str, object] = {}

    @classmethod
    def init(cls, **kwargs):
        cls.events.append("init")
        cls.init_payload = dict(kwargs)
        return kwargs

    @classmethod
    def setup(cls, **kwargs):
        cls.events.append("setup")
        return {"session": "ordered", **kwargs}

    @classmethod
    def extract_log(cls, **kwargs):
        cls.events.append("extract_log")
        return {"status": "ok", "kwargs": kwargs}

    @classmethod
    def score_influence(cls, **kwargs):
        cls.events.append("score_influence")
        return {
            test_id: {train_id: float(index + 1) for index, train_id in enumerate(kwargs["train_sample_ids"])}
            for test_id in kwargs["test_sample_ids"]
        }

    @classmethod
    def run(cls, **kwargs):
        cls.events.append("run")
        return {"status": "ok", "num_ranked": len(kwargs.get("sample_ids", []))}


class _NoInitLogIX:
    @staticmethod
    def setup(**kwargs):
        return kwargs

    @staticmethod
    def run(**kwargs):
        return {"status": "ok", "kwargs": kwargs}


class _PathConfigLogIX:
    __version__ = "0.1.1"
    init_payload: dict[str, object] = {}

    @classmethod
    def init(cls, project: str, config: str = "./config.yaml"):
        cls.init_payload = {"project": project, "config": config}
        return cls.init_payload

    @staticmethod
    def setup(**kwargs):
        return {"session": "path-config", **kwargs}

    @staticmethod
    def run(**kwargs):
        return {"status": "ok", "num_ranked": len(kwargs.get("sample_ids", []))}


class _ProjectOnlyInitLogIX:
    init_payload: dict[str, object] = {}

    @classmethod
    def init(cls, project: str):
        cls.init_payload = {"project": project}
        return cls.init_payload

    @staticmethod
    def setup(**kwargs):
        return {"session": "project-only", **kwargs}

    @staticmethod
    def run(**kwargs):
        return {"status": "ok", "num_ranked": len(kwargs.get("sample_ids", []))}


class _PermissiveButStrictKwargLogIX:
    init_payload: dict[str, object] = {}
    run_payloads: list[dict[str, object]] = []

    @classmethod
    def init(cls, **kwargs):
        cls.init_payload = dict(kwargs)
        return cls.init_payload

    @staticmethod
    def setup(*args, **kwargs):
        if "device" in kwargs:
            raise TypeError("setup() got an unexpected keyword argument 'device'")
        return {"session": "compat-ok", "kwargs": kwargs}

    @classmethod
    def run(cls, **kwargs):
        cls.run_payloads.append(dict(kwargs))
        return {"status": "ok", "num_ranked": len(kwargs.get("sample_ids", []))}


class _LegacySetupLogIX011:
    __version__ = "0.1.1"

    @staticmethod
    def setup(*, model_name_or_path: str, seed: int, output_dir: str, log_option_kwargs: dict):
        return {
            "status": "legacy-setup",
            "model_name_or_path": model_name_or_path,
            "seed": seed,
            "output_dir": output_dir,
            "log_option_kwargs": log_option_kwargs,
        }


class _ModuleExecuteOnlyLogIX:
    @staticmethod
    def execute(*, sample_ids: list[str], top_k: int):
        return {"status": "ok", "num_ranked": len(sample_ids), "top_k": top_k}


class _ContextRunOnly:
    def run(self, *, sample_ids: list[str]):
        return {"status": "ok", "num_ranked": len(sample_ids), "path": "context.run"}


class _StrictKwargsLogIX:
    @staticmethod
    def run(**kwargs):
        if "influence_mode" in kwargs:
            raise TypeError("run() got an unexpected keyword argument 'influence_mode'")
        return {
            "status": "ok",
            "num_ranked": len(kwargs.get("sample_ids", [])),
            "top_k": kwargs.get("top_k"),
        }


class _LegacyInstrumentationLogIX:
    __version__ = "0.1.1"
    analyses: list[dict[str, object]] = []
    logs: dict[str, dict[str, object]] = {}

    @staticmethod
    def init(**kwargs):
        return kwargs

    @staticmethod
    def setup(**kwargs):
        _ = kwargs
        return None

    @classmethod
    def log(cls, *, split: str, sample_ids: list[str]):
        cls.logs[split] = {"sample_ids": list(sample_ids)}
        return cls.logs[split]

    @classmethod
    def get_log(cls, *, split: str):
        return cls.logs.get(split, {"sample_ids": []})

    @classmethod
    def add_analysis(cls, *args, **kwargs):
        cls.analyses.append({"args": args, "kwargs": kwargs})

    @classmethod
    def finalize(cls):
        if not cls.analyses:
            return {"status": "legacy-empty", "scores": {}}
        kwargs = cls.analyses[-1]["kwargs"]
        train_sample_ids = list(kwargs.get("train_sample_ids", []))
        scores = {
            test_id: {train_id: float(index + 1) for index, train_id in enumerate(train_sample_ids)}
            for test_id in kwargs.get("test_sample_ids", [])
        }
        return {"status": "legacy-ok", "scores": scores}



class _LegacyInstrumentationMarkerSetupLogIX:
    __version__ = "0.2.0"
    analyses: list[dict[str, object]] = []
    logs: dict[str, dict[str, object]] = {}

    @staticmethod
    def init(**kwargs):
        return kwargs

    @staticmethod
    def setup(**kwargs):
        _ = kwargs
        return {"__logix_mode__": "legacy_instrumentation", "marker": "explicit"}

    @classmethod
    def log(cls, *, split: str, sample_ids: list[str]):
        cls.logs[split] = {"sample_ids": list(sample_ids)}
        return cls.logs[split]

    @classmethod
    def get_log(cls, *, split: str):
        return cls.logs.get(split, {"sample_ids": []})

    @classmethod
    def add_analysis(cls, *args, **kwargs):
        cls.analyses.append({"args": args, "kwargs": kwargs})

    @classmethod
    def finalize(cls):
        if not cls.analyses:
            return {"status": "legacy-empty", "scores": {}}
        kwargs = cls.analyses[-1]["kwargs"]
        train_sample_ids = list(kwargs.get("train_sample_ids", []))
        scores = {
            test_id: {train_id: float(index + 1) for index, train_id in enumerate(train_sample_ids)}
            for test_id in kwargs.get("test_sample_ids", [])
        }
        return {"status": "legacy-ok", "scores": scores}


class _LegacyInstrumentationStrictSplitLogIX:
    analyses: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    log_calls: list[dict[str, object]] = []
    get_log_calls: list[dict[str, object]] = []

    @staticmethod
    def init(**kwargs):
        return kwargs

    @staticmethod
    def setup(**kwargs):
        _ = kwargs
        return {"__logix_mode__": "legacy_instrumentation"}

    @classmethod
    def log(cls, **kwargs):
        cls.log_calls.append(dict(kwargs))
        if "split" in kwargs:
            raise TypeError("log() got an unexpected keyword argument 'split'")
        sample_ids = list(kwargs.get("sample_ids", []))
        entry = {"sample_ids": sample_ids}
        cls.logs.append(entry)
        return entry

    @classmethod
    def get_log(cls, **kwargs):
        cls.get_log_calls.append(dict(kwargs))
        if "split" in kwargs:
            raise TypeError("get_log() got an unexpected keyword argument 'split'")
        if cls.logs:
            return cls.logs.pop(0)
        return {"sample_ids": []}

    @classmethod
    def add_analysis(cls, *args, **kwargs):
        cls.analyses.append({"args": args, "kwargs": kwargs})

    @classmethod
    def finalize(cls):
        if not cls.analyses:
            return {"status": "legacy-empty", "scores": {}}
        kwargs = cls.analyses[-1]["kwargs"]
        train_sample_ids = list(kwargs.get("train_sample_ids", []))
        scores = {
            test_id: {train_id: float(index + 1) for index, train_id in enumerate(train_sample_ids)}
            for test_id in kwargs.get("test_sample_ids", [])
        }
        return {"status": "legacy-ok", "scores": scores}


def _make_engine_config(tmp_path: Path, setup_kwargs: dict[str, object]) -> LogIXEngineConfig:
    return LogIXEngineConfig(
        run_id="setup-test",
        project_name="dataset_attribution",
        output_root=tmp_path,
        model_name_or_path="fake-model",
        seed=123,
        top_k=1,
        train_subset_size=1,
        influence_mode="ihvp",
        ihvp_controls={
            "damping": 0.01,
            "scale": 10.0,
            "recursion_depth": 8,
            "num_samples": 1,
        },
        lora_only=True,
        lora_adapter_path=None,
        tokenizer_path=None,
        train_split_path_override=None,
        test_split_path_override=None,
        shadow_split_path_override=None,
        init_config_path=None,
        setup_kwargs=setup_kwargs,
        run_kwargs={},
        patch_kwargs={},
        extract_kwargs={},
        score_kwargs={},
        test_subset_size=None,
        source_config_path=tmp_path / "attribution_setup.yaml",
    )


def test_setup_logix_fully_compatible_setup_payload(tmp_path: Path, caplog) -> None:
    class _CompatibleLogIX:
        @staticmethod
        def setup(**kwargs):
            return {"status": "ok", "kwargs": kwargs}

    config = _make_engine_config(tmp_path, {"device": "cpu"})
    caplog.set_level("INFO")

    context = setup_logix(config, tmp_path / "out", _CompatibleLogIX)

    assert context["status"] == "ok"
    assert context["kwargs"]["device"] == "cpu"
    assert "LogIX setup API path=modern" in caplog.text
    assert "legacy compatibility path" not in caplog.text


def test_setup_logix_retries_after_unexpected_kwarg_rejection(tmp_path: Path, caplog) -> None:
    class _StrictWrapperLogIX:
        @staticmethod
        def setup(*, model_name_or_path: str, seed: int, output_dir: str):
            return {
                "model_name_or_path": model_name_or_path,
                "seed": seed,
                "output_dir": output_dir,
            }

    config = _make_engine_config(tmp_path, {"device": "cpu"})
    caplog.set_level("INFO")

    context = setup_logix(config, tmp_path / "out", _StrictWrapperLogIX)

    assert context["model_name_or_path"] == "fake-model"
    assert "log_option_kwargs" not in context
    assert "attempting legacy compatibility path" in caplog.text
    assert "LogIX setup API path=legacy_0_1_1" in caplog.text


def test_setup_logix_falls_back_to_legacy_setup_call_when_parse_fails(tmp_path: Path, caplog) -> None:
    class _LegacyFallbackLogIX:
        attempts = 0

        @classmethod
        def setup(cls, **kwargs):
            cls.attempts += 1
            if kwargs:
                raise TypeError("got an unexpected keyword argument")
            return {"status": "legacy-ok"}

    config = _make_engine_config(tmp_path, {"device": "cpu"})
    caplog.set_level("INFO")

    context = setup_logix(config, tmp_path / "out", _LegacyFallbackLogIX)

    assert context["status"] == "legacy-ok"
    assert _LegacyFallbackLogIX.attempts == 3
    assert "legacy path could not parse rejected kwargs; retrying with empty payload" in caplog.text


def test_setup_logix_uses_legacy_0_1_1_payload_shape(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {"device": "cpu", "batch_size": 2})
    caplog.set_level("INFO")

    context = setup_logix(config, tmp_path / "out", _LegacySetupLogIX011)

    assert context["status"] == "legacy-setup"
    assert context["log_option_kwargs"]["device"] == "cpu"
    assert context["log_option_kwargs"]["batch_size"] == 2
    assert "LogIX setup API path=legacy_0_1_1" in caplog.text


def test_execute_logix_prefers_context_run_over_module_entrypoints(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")

    result = execute_logix(
        context=_ContextRunOnly(),
        config=config,
        sample_ids=["a", "b", "c"],
        test_sample_ids=["t1"],
        logix_module=_ModuleExecuteOnlyLogIX,
    )

    assert result["path"] == "context.run"
    assert "selected_api_path=context.run" in caplog.text


def test_execute_logix_uses_module_execute_for_modern_api(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")

    result = execute_logix(
        context={},
        config=config,
        sample_ids=["a", "b", "c"],
        test_sample_ids=["t1"],
        logix_module=_ModuleExecuteOnlyLogIX,
    )

    assert result["status"] == "ok"
    assert result["num_ranked"] == 1
    assert "selected_api_path=logix.execute" in caplog.text


def test_execute_logix_retries_strict_kwargs_rejection(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")

    result = execute_logix(
        context={},
        config=config,
        sample_ids=["a", "b", "c"],
        test_sample_ids=["t1"],
        logix_module=_StrictKwargsLogIX,
    )

    assert result["status"] == "ok"
    assert result["top_k"] == 1
    assert "strict-kwargs rejection path=logix.run" in caplog.text


def test_execute_logix_legacy_instrumentation_path(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")
    _LegacyInstrumentationLogIX.analyses = []
    _LegacyInstrumentationLogIX.logs = {}
    context = setup_logix(config, tmp_path / "out", _LegacyInstrumentationLogIX)

    assert context is None

    result = execute_logix(
        context=context,
        config=config,
        sample_ids=["a", "b"],
        test_sample_ids=["t1"],
        logix_module=_LegacyInstrumentationLogIX,
    )

    assert result["status"] == "legacy-ok"
    assert "influence_scores" in result
    assert result["influence_scores"]["t1"]["a"] == 1.0
    assert "selected_api_path=logix.finalize(legacy_module)" in caplog.text


def test_execute_logix_legacy_instrumentation_marker_context_path(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")
    _LegacyInstrumentationMarkerSetupLogIX.analyses = []
    _LegacyInstrumentationMarkerSetupLogIX.logs = {}
    context = setup_logix(config, tmp_path / "out", _LegacyInstrumentationMarkerSetupLogIX)

    assert context == {"__logix_mode__": "legacy_instrumentation", "marker": "explicit"}

    result = execute_logix(
        context=context,
        config=config,
        sample_ids=["a", "b"],
        test_sample_ids=["t1"],
        logix_module=_LegacyInstrumentationMarkerSetupLogIX,
    )

    assert result["status"] == "legacy-ok"
    assert "influence_scores" in result
    assert result["influence_scores"]["t1"]["a"] == 1.0
    assert "selected_api_path=logix.finalize(legacy_instrumentation)" in caplog.text


def test_execute_logix_legacy_log_split_rejection_fallback(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")
    _LegacyInstrumentationStrictSplitLogIX.analyses = []
    _LegacyInstrumentationStrictSplitLogIX.logs = []
    _LegacyInstrumentationStrictSplitLogIX.log_calls = []
    _LegacyInstrumentationStrictSplitLogIX.get_log_calls = []

    result = execute_logix(
        context={"__logix_mode__": "legacy_instrumentation"},
        config=config,
        sample_ids=["a", "b"],
        test_sample_ids=["t1"],
        logix_module=_LegacyInstrumentationStrictSplitLogIX,
    )

    assert result["status"] == "legacy-ok"
    assert _LegacyInstrumentationStrictSplitLogIX.log_calls[0]["split"] == "train"
    assert _LegacyInstrumentationStrictSplitLogIX.log_calls[1] == {"sample_ids": ["a"]}
    assert _LegacyInstrumentationStrictSplitLogIX.log_calls[2]["split"] == "test"
    assert _LegacyInstrumentationStrictSplitLogIX.log_calls[3] == {"sample_ids": ["t1"]}
    assert "callable=logix.log removed_kwargs=['split'] retained_kwargs=['sample_ids']" in caplog.text


def test_execute_logix_legacy_get_log_split_rejection_fallback(tmp_path: Path, caplog) -> None:
    config = _make_engine_config(tmp_path, {})
    caplog.set_level("INFO")
    _LegacyInstrumentationStrictSplitLogIX.analyses = []
    _LegacyInstrumentationStrictSplitLogIX.logs = []
    _LegacyInstrumentationStrictSplitLogIX.log_calls = []
    _LegacyInstrumentationStrictSplitLogIX.get_log_calls = []

    result = execute_logix(
        context={"__logix_mode__": "legacy_instrumentation"},
        config=config,
        sample_ids=["a", "b"],
        test_sample_ids=["t1"],
        logix_module=_LegacyInstrumentationStrictSplitLogIX,
    )

    analysis_kwargs = _LegacyInstrumentationStrictSplitLogIX.analyses[-1]["kwargs"]
    assert result["status"] == "legacy-ok"
    assert _LegacyInstrumentationStrictSplitLogIX.get_log_calls[0]["split"] == "train"
    assert _LegacyInstrumentationStrictSplitLogIX.get_log_calls[1] == {}
    assert _LegacyInstrumentationStrictSplitLogIX.get_log_calls[2]["split"] == "test"
    assert _LegacyInstrumentationStrictSplitLogIX.get_log_calls[3] == {}
    assert analysis_kwargs["train_log"]["sample_ids"] == ["a"]
    assert analysis_kwargs["test_log"]["sample_ids"] == ["t1"]
    assert "callable=logix.get_log removed_kwargs=['split'] retained_kwargs=[]" in caplog.text


def test_execute_logix_error_lists_discovered_callables(tmp_path: Path) -> None:
    class _UnsupportedContext:
        def noop(self):
            return None

    class _UnsupportedModule:
        def noop(self):
            return None

    config = _make_engine_config(tmp_path, {})
    try:
        execute_logix(
            context=_UnsupportedContext(),
            config=config,
            sample_ids=["a"],
            test_sample_ids=["t1"],
            logix_module=_UnsupportedModule(),
        )
    except AttributeError as exc:
        message = str(exc)
        assert "Attempted paths=[]" in message
        assert "Discovered callable names=[]" in message
    else:
        raise AssertionError("Expected detailed AttributeError for unsupported execute API")


def _write_manifest(path: Path, count: int, start: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source", "license"])
        writer.writeheader()
        for idx in range(start, start + count):
            writer.writerow(
                {
                    "sample_id": f"sample-{idx:05d}",
                    "source": "source",
                    "license": "cc-by",
                }
            )


def test_logix_engine_writes_results_and_metadata(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-test"
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=4, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "seed": 2026,
                "top_k": 2,
                "train_subset_size": 4,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {
                    "setup": {"device": "cpu"},
                    "run": {},
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)

    assert artifacts.output_dir.exists()
    assert artifacts.influence_scores_path.exists()
    assert artifacts.topk_path.exists()
    assert artifacts.metadata_path.exists()

    influence_lines = artifacts.influence_scores_path.read_text(encoding="utf-8").strip().splitlines()
    assert influence_lines[0] == "test_id,train_id,influence_score,rank"
    assert len(influence_lines) == 1 + (2 * 4)

    topk = json.loads(artifacts.topk_path.read_text(encoding="utf-8"))
    assert len(topk) == 2
    first_test = sorted(topk.keys())[0]
    assert len(topk[first_test]) == 2
    assert topk[first_test][0]["rank"] == 1

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "logix-test"
    assert metadata["model_name_or_path"] == "fake-model"
    assert metadata["seed"] == 2026
    assert metadata["top_k"] == 2
    assert metadata["setup_kwargs"]["device"] == "cpu"
    assert metadata["versions"]["python"]
    assert "logix" in metadata["versions"]
    assert "transformers" in metadata["versions"]
    assert "torch" in metadata["versions"]
    assert metadata["timing"]["setup_seconds"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["setup"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["extraction"] >= 0.0
    assert metadata["timing"]["phase_timings_seconds"]["influence_scoring"] >= 0.0
    assert "platform" in metadata["hardware"]
    assert "cuda_available" in metadata["hardware"]
    assert "git_commit_hash" in metadata["git"]
    assert "git_dirty" in metadata["git"]
    assert metadata["timing"]["total_seconds"] >= 0.0
    assert metadata["artifacts"]["execute_logix_result"]["status"] == "ok"


def test_logix_engine_legacy_instrumentation_flow(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-legacy"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    tokenizer_dir = run_root / "train" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)
    _LegacyInstrumentationLogIX.analyses = []
    _LegacyInstrumentationLogIX.logs = {}

    config_path = tmp_path / "attribution_logix_legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-legacy",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "seed": 99,
                "top_k": 2,
                "train_subset_size": 3,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_LegacyInstrumentationLogIX)
    topk = json.loads(artifacts.topk_path.read_text(encoding="utf-8"))
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert len(topk) == 2
    assert metadata["artifacts"]["execute_logix_result"]["status"] == "legacy-ok"
    assert "influence_scores" in metadata["artifacts"]["execute_logix_result"]


def test_logix_engine_without_manifest_still_runs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-with-default-splits"
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_nomani.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-with-default-splits",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "model_name_or_path": "fake-model",
                "top_k": 2,
                "train_subset_size": 8,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["artifacts"]["num_samples"] == 3
    assert metadata["artifacts"]["execute_logix_result"]["top_sample_id"] is not None


def test_logix_engine_rejects_invalid_run_id(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_bad_id.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "bad/run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": False},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Invalid run_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid run_id")


def test_logix_engine_rejects_unsupported_influence_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "attribution_logix_bad_mode.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {"mode": "dense_hessian", "ihvp": {}},
                "lora": {"lora_only": False},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Unsupported influence mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported influence mode")


def test_logix_engine_rejects_missing_train_artifacts_for_lora_only(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "logix-test"
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_missing_adapter.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "logix-test",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except FileNotFoundError as exc:
        assert "LoRA adapter artifacts" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing adapter path")


def test_logix_engine_preflight_reports_run_id_mismatch_with_fix_hint(tmp_path: Path) -> None:
    valid_run_root = tmp_path / "outputs" / "runs" / "aligned-run"
    (valid_run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (valid_run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (valid_run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(valid_run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(valid_run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_bad_run_linkage.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "wrong-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 1,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except FileNotFoundError as exc:
        message = str(exc)
        assert "selected run_id='wrong-run'" in message
        assert "train split manifest" in message
        assert "test split manifest" in message
        assert "LoRA adapter artifacts" in message
        assert "same run_id for splits, training artifacts, and attribution" in message
        assert "train_manifest_path/test_manifest_path overrides" in message
    else:
        raise AssertionError("Expected preflight mismatch error for run_id with missing inputs")


def test_logix_engine_supports_split_path_overrides(tmp_path: Path) -> None:
    split_run_root = tmp_path / "outputs" / "runs" / "split-source"
    _write_manifest(split_run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(split_run_root / "splits" / "test.csv", count=2, start=100)

    artifact_run_root = tmp_path / "outputs" / "runs" / "artifact-source"
    (artifact_run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (artifact_run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (artifact_run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    config_path = tmp_path / "attribution_logix_override.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "artifact-source",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "train_manifest_path": str(split_run_root / "splits" / "train.csv"),
                "test_manifest_path": str(split_run_root / "splits" / "test.csv"),
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_manifest_path"].endswith("split-source/splits/train.csv")
    assert metadata["test_manifest_path"].endswith("split-source/splits/test.csv")
    assert metadata["train_manifest_path_override"].endswith("split-source/splits/train.csv")
    assert metadata["test_manifest_path_override"].endswith("split-source/splits/test.csv")


def test_logix_engine_mixed_run_layout_uses_explicit_manifest_paths(tmp_path: Path) -> None:
    final_report_root = tmp_path / "outputs" / "runs" / "final_report_run"
    (final_report_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (final_report_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (final_report_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")

    default_split_root = tmp_path / "outputs" / "runs" / "default_run"
    _write_manifest(default_split_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(default_split_root / "splits" / "test.csv", count=2, start=100)
    _write_manifest(default_split_root / "splits" / "shadow.csv", count=1, start=200)

    config_path = tmp_path / "attribution_logix_mixed_run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "final_report_run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 2,
                "train_subset_size": 3,
                "train_manifest_path": "outputs/runs/default_run/splits/train.csv",
                "test_manifest_path": "outputs/runs/default_run/splits/test.csv",
                "shadow_manifest_path": "outputs/runs/default_run/splits/shadow.csv",
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {
                    "lora_only": True,
                    "adapter_path": "outputs/runs/final_report_run/train/adapter",
                },
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        # Config paths in this regression use repo-relative paths.
        import os

        os.chdir(tmp_path)
        artifacts = run_logix_engine(config_path, logix_module=_FakeLogIX)
    finally:
        os.chdir(cwd)

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_manifest_path"] == "outputs/runs/default_run/splits/train.csv"
    assert metadata["test_manifest_path"] == "outputs/runs/default_run/splits/test.csv"
    assert metadata["train_manifest_path_override"] == "outputs/runs/default_run/splits/train.csv"
    assert metadata["test_manifest_path_override"] == "outputs/runs/default_run/splits/test.csv"


def test_logix_engine_tiny_smoke_flow_is_deterministic(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    run_id = "smoke-flow"
    run_root = output_root / run_id
    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    # Tiny sample manifests to keep this as a smoke flow.
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_payload = {
        "run_id": run_id,
        "output_root": str(output_root),
        "model_name_or_path": "fake-model",
        "seed": 1337,
        "top_k": 2,
        "train_subset_size": 3,
        "influence": {
            "mode": "ihvp",
            "ihvp": {
                "damping": 0.01,
                "scale": 10.0,
                "recursion_depth": 4,
                "num_samples": 1,
            },
        },
        "lora": {"lora_only": True},
        "logix": {
            "setup": {"device": "cpu"},
            "run": {},
            "test_subset_size": 2,
        },
    }
    config_path = tmp_path / "attribution_logix_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

    first = run_logix_engine(config_path, logix_module=_FakeLogIX)
    second = run_logix_engine(config_path, logix_module=_FakeLogIX)

    expected_output_dir = output_root / run_id / "attribution" / "logix"
    assert first.output_dir == expected_output_dir
    assert first.metadata_path == expected_output_dir / "metadata.json"
    assert first.influence_scores_path == expected_output_dir / "influence_scores.csv"
    assert first.topk_path == expected_output_dir / "topk.json"

    assert first.metadata_path.exists()
    assert first.influence_scores_path.exists()
    assert first.topk_path.exists()

    csv_lines = first.influence_scores_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines
    assert csv_lines[0] == "test_id,train_id,influence_score,rank"
    assert len(csv_lines) > 1

    topk = json.loads(first.topk_path.read_text(encoding="utf-8"))
    assert topk
    for test_id, ranking in topk.items():
        assert test_id
        assert ranking
        assert all("train_id" in item and "influence_score" in item and "rank" in item for item in ranking)

    # Fixed seed/config should always produce the same ordering for train subset and rankings.
    metadata_first = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata_second = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert metadata_first["artifacts"]["train_subset_ids"] == metadata_second["artifacts"]["train_subset_ids"]
    assert first.influence_scores_path.read_text(encoding="utf-8") == second.influence_scores_path.read_text(
        encoding="utf-8"
    )
    assert first.topk_path.read_text(encoding="utf-8") == second.topk_path.read_text(encoding="utf-8")


def test_logix_engine_smoke_setup_compat_fallback_allows_run_to_proceed(tmp_path: Path, caplog) -> None:
    _PermissiveButStrictKwargLogIX.init_payload = {}
    _PermissiveButStrictKwargLogIX.run_payloads = []
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    output_root = tmp_path / "outputs" / "runs"
    run_id = "compat-smoke"
    run_root = output_root / run_id
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_compat_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": run_id,
                "output_root": str(output_root),
                "model_name_or_path": "fake-model",
                "seed": 2026,
                "top_k": 1,
                "train_subset_size": 3,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 4,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {
                    "setup": {"device": "cpu", "compat_mode": "legacy"},
                    "run": {},
                    "test_subset_size": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    caplog.set_level("INFO")
    artifacts = run_logix_engine(config_path, logix_module=_PermissiveButStrictKwargLogIX)

    assert artifacts.influence_scores_path.exists()
    assert _PermissiveButStrictKwargLogIX.init_payload["project"] == "dataset_attribution"
    assert _PermissiveButStrictKwargLogIX.run_payloads
    assert "attempting legacy compatibility path" in caplog.text
    assert "LogIX setup API path=legacy_0_1_1" in caplog.text


def test_logix_engine_initializes_logix_before_downstream_calls(tmp_path: Path) -> None:
    _OrderedLogIX.events = []
    _OrderedLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "ordered-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=3, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=2, start=100)

    config_path = tmp_path / "attribution_logix_ordered.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "ordered-run",
                "project_name": "ordered-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 3,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    run_logix_engine(config_path, logix_module=_OrderedLogIX)
    assert "init" in _OrderedLogIX.events
    assert _OrderedLogIX.init_payload["project"] == "ordered-project"
    init_index = _OrderedLogIX.events.index("init")
    assert init_index < _OrderedLogIX.events.index("setup")
    assert init_index < _OrderedLogIX.events.index("extract_log")
    assert init_index < _OrderedLogIX.events.index("score_influence")
    assert init_index < _OrderedLogIX.events.index("run")


def test_logix_engine_fails_with_explicit_error_when_logix_init_is_unavailable(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "no-init-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_no_init.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "no-init-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_NoInitLogIX)
    except RuntimeError as exc:
        assert "logix.init" in str(exc)
        assert "incompatible LogIX version" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when logix.init is unavailable")


def test_logix_engine_resolves_project_from_nested_logix_project(tmp_path: Path) -> None:
    _OrderedLogIX.events = []
    _OrderedLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "nested-project-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_nested_project.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "nested-project-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"project": "nested-project", "setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    run_logix_engine(config_path, logix_module=_OrderedLogIX)
    assert _OrderedLogIX.init_payload["project"] == "nested-project"


def test_logix_engine_rejects_project_from_legacy_logix_init_project_mapping(tmp_path: Path) -> None:
    _OrderedLogIX.events = []
    _OrderedLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "init-project-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_init_project.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "init-project-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"init": {"project": "init-project"}, "setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_OrderedLogIX)
    except ValueError as exc:
        assert "Deprecated config key `logix.init`" in str(exc)
    else:
        raise AssertionError("Expected ValueError for deprecated logix.init mapping")


def test_logix_engine_uses_default_project_when_project_is_absent(tmp_path: Path) -> None:
    _OrderedLogIX.events = []
    _OrderedLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "default-project-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_default_project.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "default-project-run",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    run_logix_engine(config_path, logix_module=_OrderedLogIX)
    assert _OrderedLogIX.init_payload["project"] == "dataset_attribution"


def test_logix_engine_rejects_empty_project_after_resolution(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "invalid-project-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_invalid_project.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "invalid-project-run",
                "project_name": "   ",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"project": "nested-project", "setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "resolved LogIX project is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty resolved project")


def test_logix_engine_does_not_depend_on_cwd_config_yaml(tmp_path: Path) -> None:
    _PathConfigLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "no-cwd-config-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_no_cwd_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "no-cwd-config-run",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}, "init_config_path": str(tmp_path / "missing-logix-config.yaml")},
            }
        ),
        encoding="utf-8",
    )

    isolated = tmp_path / "isolated"
    isolated.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        import os

        os.chdir(isolated)
        run_logix_engine(config_path, logix_module=_PathConfigLogIX)
    except FileNotFoundError as exc:
        assert "logix.init_config_path was provided but file does not exist" in str(exc)
        assert "logix.config_path" in str(exc)
    finally:
        os.chdir(cwd)
    assert _PathConfigLogIX.init_payload == {}


def test_logix_engine_rejects_legacy_logix_init_mapping(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs" / "legacy-init-mapping-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_legacy_init_mapping.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "legacy-init-mapping-run",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}, "init": {"config": "configs/logix.yaml"}},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Deprecated config key `logix.init`" in str(exc)
        assert "project_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for deprecated logix.init mapping")


def test_logix_engine_uses_explicit_init_config_path_when_supported(tmp_path: Path) -> None:
    _PathConfigLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "path-config-default-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)
    logix_init_cfg = tmp_path / "configs" / "logix_init.yaml"
    logix_init_cfg.parent.mkdir(parents=True, exist_ok=True)
    logix_init_cfg.write_text("project: path-config-project\n", encoding="utf-8")

    config_path = tmp_path / "attribution_logix_path_default.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "path-config-default-run",
                "project_name": "path-config-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}, "init_config_path": "configs/logix_init.yaml"},
            }
        ),
        encoding="utf-8",
    )

    run_logix_engine(config_path, logix_module=_PathConfigLogIX)
    assert isinstance(_PathConfigLogIX.init_payload["config"], str)
    assert _PathConfigLogIX.init_payload["config"] == str(logix_init_cfg)
    assert _PathConfigLogIX.init_payload["project"] == "path-config-project"


def test_logix_engine_init_fallback_remains_compatible_when_config_kwarg_unsupported(tmp_path: Path) -> None:
    _ProjectOnlyInitLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "project-only-config-path-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)
    logix_init_cfg = tmp_path / "configs" / "legacy_init.yaml"
    logix_init_cfg.parent.mkdir(parents=True, exist_ok=True)
    logix_init_cfg.write_text("project: explicit-project\n", encoding="utf-8")

    config_path = tmp_path / "attribution_logix_path_default.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "project-only-config-path-run",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}, "init_config_path": "configs/legacy_init.yaml"},
            }
        ),
        encoding="utf-8",
    )

    artifacts = run_logix_engine(config_path, logix_module=_ProjectOnlyInitLogIX)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert _ProjectOnlyInitLogIX.init_payload == {"project": "explicit-project"}
    assert metadata["logix_init_config_path"] == str(logix_init_cfg)


def test_logix_engine_fails_fast_on_dict_config_for_path_based_logix(tmp_path: Path) -> None:
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "path-config-bad-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_legacy_init_scalar.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "legacy-init-scalar-run",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}, "init": 123},
            }
        ),
        encoding="utf-8",
    )

    try:
        run_logix_engine(config_path, logix_module=_FakeLogIX)
    except ValueError as exc:
        assert "Invalid config for `logix.init`" in str(exc)
        assert "project_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError when logix.init has malformed type")


def test_logix_engine_fails_fast_when_path_based_init_config_is_implicit_and_missing(tmp_path: Path) -> None:
    _PathConfigLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "implicit-missing-init-config"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_missing_implicit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "implicit-missing-init-config",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    isolated = tmp_path / "isolated_cwd_no_config"
    isolated.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        import os

        os.chdir(isolated)
        run_logix_engine(config_path, logix_module=_PathConfigLogIX)
    except FileNotFoundError as exc:
        assert "Installed logix-ai appears to expect an implicit `./config.yaml`" in str(exc)
        assert "logix.init_config_path" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing implicit ./config.yaml")
    finally:
        os.chdir(cwd)


def test_logix_engine_runs_without_local_config_yaml_when_init_accepts_project_only(tmp_path: Path) -> None:
    _ProjectOnlyInitLogIX.init_payload = {}
    _INITIALIZED_LOGIX_MODULE_IDS.clear()
    run_root = tmp_path / "outputs" / "runs" / "project-only-run"
    (run_root / "train" / "adapter").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_root / "splits" / "train.csv", count=2, start=0)
    _write_manifest(run_root / "splits" / "test.csv", count=1, start=100)

    config_path = tmp_path / "attribution_logix_project_only.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "project-only-run",
                "project_name": "explicit-project",
                "output_root": str(tmp_path / "outputs" / "runs"),
                "top_k": 1,
                "train_subset_size": 2,
                "influence": {
                    "mode": "ihvp",
                    "ihvp": {
                        "damping": 0.01,
                        "scale": 10.0,
                        "recursion_depth": 8,
                        "num_samples": 1,
                    },
                },
                "lora": {"lora_only": True},
                "logix": {"setup": {}, "run": {}},
            }
        ),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    isolated = tmp_path / "isolated_cwd"
    isolated.mkdir(parents=True, exist_ok=True)
    try:
        import os

        os.chdir(isolated)
        run_logix_engine(config_path, logix_module=_ProjectOnlyInitLogIX)
    finally:
        os.chdir(cwd)
    assert _ProjectOnlyInitLogIX.init_payload == {"project": "explicit-project"}
