"""Smoke integration test for logix-ai==0.1.1 compatibility."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not bool(os.environ.get("RUN_LOGIX_SMOKE_011")),
    reason="Set RUN_LOGIX_SMOKE_011=1 to run logix-ai==0.1.1 smoke integration test.",
)
def test_logix_011_step2_smoke_flow(tmp_path: Path) -> None:
    """Install + generate smoke config + run attribution with tiny sample sizes."""
    repo_root = Path(__file__).resolve().parents[1]

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(repo_root),
        "logix-ai==0.1.1",
    ]
    install_proc = subprocess.run(
        install_cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if install_proc.returncode != 0:
        combined_install_output = f"{install_proc.stdout}\n{install_proc.stderr}".lower()
        if any(token in combined_install_output for token in ("proxyerror", "no matching distribution", "cannot connect to proxy")):
            pytest.skip(
                "Skipping logix-ai==0.1.1 smoke test because package install failed in this environment "
                "(network/package index unavailable)."
            )
        raise AssertionError(
            f"Failed dependency install: {' '.join(install_cmd)}\n"
            f"stdout:\n{install_proc.stdout}\n"
            f"stderr:\n{install_proc.stderr}"
        )

    output_root = tmp_path / "outputs" / "runs"
    run_id = "smoke-logix-011"
    run_root = output_root / run_id

    adapter_dir = run_root / "train" / "adapter"
    tokenizer_dir = run_root / "train" / "tokenizer"
    splits_dir = run_root / "splits"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    train_manifest = splits_dir / "train.csv"
    test_manifest = splits_dir / "test.csv"
    train_manifest.write_text(
        "sample_id,source,license\ntrain-0001,smoke,cc-by\ntrain-0002,smoke,cc-by\n",
        encoding="utf-8",
    )
    test_manifest.write_text(
        "sample_id,source,license\ntest-0001,smoke,cc-by\n",
        encoding="utf-8",
    )

    smoke_cfg_path = tmp_path / "smoke_attribution_logix.yaml"
    generate_cmd = [
        "bash",
        "scripts/run_step2_logix.sh",
        "--generate-smoke-config",
        str(smoke_cfg_path),
        "--smoke-run-id",
        run_id,
        "--smoke-output-root",
        str(output_root),
    ]
    generate_proc = subprocess.run(
        generate_cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert generate_proc.returncode == 0, (
        f"Smoke config generation failed: {' '.join(generate_cmd)}\n"
        f"stdout:\n{generate_proc.stdout}\n"
        f"stderr:\n{generate_proc.stderr}"
    )

    generated_cfg = yaml.safe_load(smoke_cfg_path.read_text(encoding="utf-8"))
    generated_cfg["top_k"] = 1
    generated_cfg["train_subset_size"] = 2
    generated_cfg["test_manifest_path"] = str(test_manifest)
    generated_cfg["train_manifest_path"] = str(train_manifest)
    generated_cfg["model_name_or_path"] = "smoke-model"
    generated_cfg["logix"] = generated_cfg.get("logix", {}) | {"test_subset_size": 1}

    smoke_tiny_cfg_path = tmp_path / "smoke_attribution_logix_tiny.yaml"
    smoke_tiny_cfg_path.write_text(
        yaml.safe_dump(generated_cfg, sort_keys=False),
        encoding="utf-8",
    )

    run_cmd = [
        sys.executable,
        "-m",
        "src.attribution.logix_engine",
        "--config",
        str(smoke_tiny_cfg_path),
    ]
    run_proc = subprocess.run(
        run_cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = f"{run_proc.stdout}\n{run_proc.stderr}"
    assert run_proc.returncode == 0, (
        f"Step2 attribution run failed: {' '.join(run_cmd)}\n"
        f"stdout:\n{run_proc.stdout}\n"
        f"stderr:\n{run_proc.stderr}"
    )
    assert "setup API AttributeError" not in combined_output
    assert "execute API AttributeError" not in combined_output

    attribution_dir = run_root / "attribution" / "logix"
    topk_path = attribution_dir / "topk.json"
    metadata_path = attribution_dir / "metadata.json"
    assert topk_path.is_file()
    assert metadata_path.is_file()

    topk_payload = json.loads(topk_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert topk_payload
    assert metadata_payload["run_id"] == run_id
