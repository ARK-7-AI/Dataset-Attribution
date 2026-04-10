#!/usr/bin/env bash
set -euo pipefail

PROFILE_CONFIG="${1:-configs/profiles/colab_train_lora.yaml}"
WORKDIR="$(mktemp -d)"
TMP_CONFIG="${WORKDIR}/repro_config.yaml"

echo "[repro] Using profile: ${PROFILE_CONFIG}"
echo "[repro] Working directory: ${WORKDIR}"

python - <<'PY' "${PROFILE_CONFIG}" "${TMP_CONFIG}" "${WORKDIR}"
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
workdir = Path(sys.argv[3])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["run_id"] = None
cfg["auto_run_id"] = True
cfg["output_root"] = str(workdir / "runs")
cfg.setdefault("training", {})
cfg["training"]["max_steps"] = int(cfg["training"].get("max_steps") or 4)
cfg["training"]["logging_steps"] = 1
cfg["training"]["save_steps"] = int(cfg["training"].get("save_steps") or 2)
cfg["training"]["eval_strategy"] = "no"
yaml.safe_dump(cfg, dst.open("w", encoding="utf-8"), sort_keys=False)
print(f"[repro] Wrote temp config: {dst}")
PY

python -m src.training.lora_train --config "${TMP_CONFIG}"
python -m src.training.lora_train --config "${TMP_CONFIG}"

python - <<'PY' "${WORKDIR}/runs"
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys

runs_root = Path(sys.argv[1])
run_dirs = sorted(path for path in runs_root.glob("*/train") if path.is_dir())
if len(run_dirs) < 2:
    raise SystemExit("Need two training runs to compare reproducibility.")

run_a, run_b = run_dirs[-2], run_dirs[-1]
print(f"[repro] comparing:\n  A={run_a}\n  B={run_b}")

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

params_a = read_json(run_a / "params.json")
params_b = read_json(run_b / "params.json")
metrics_a = read_json(run_a / "metrics.json")
metrics_b = read_json(run_b / "metrics.json")

required_files = [
    "params.json",
    "metrics.json",
    "trainer_state.json",
    "resolved_config.yaml",
    "adapter/adapter_model.bin",
    "tokenizer/tokenizer.json",
]
for rel in required_files:
    if not (run_a / rel).exists() or not (run_b / rel).exists():
        raise SystemExit(f"Missing required artifact in one run: {rel}")

schema_a = sorted(params_a.keys())
schema_b = sorted(params_b.keys())
if schema_a != schema_b:
    raise SystemExit("params.json schema mismatch across runs")

metric_keys = {"train_loss", "steps", "train_steps_per_second", "epochs_completed"}
for key in metric_keys:
    if key not in metrics_a or key not in metrics_b:
        raise SystemExit(f"Missing metric key '{key}'")

loss_delta = abs(float(metrics_a["train_loss"]) - float(metrics_b["train_loss"]))
steps_equal = int(metrics_a["steps"]) == int(metrics_b["steps"])
sps_delta = abs(float(metrics_a["train_steps_per_second"]) - float(metrics_b["train_steps_per_second"]))

hashes_a = {rel: sha256(run_a / rel) for rel in required_files}
hashes_b = {rel: sha256(run_b / rel) for rel in required_files}
schema_consistent = schema_a == schema_b

print(f"[repro] schema_consistent={schema_consistent}")
print(f"[repro] train_loss_delta={loss_delta:.6f}")
print(f"[repro] train_steps_per_second_delta={sps_delta:.6f}")
print(f"[repro] steps_equal={steps_equal}")
for rel in required_files:
    print(f"[repro] hash {rel}: {'MATCH' if hashes_a[rel] == hashes_b[rel] else 'DIFF'}")

if not steps_equal:
    raise SystemExit("Global training steps differed across runs.")
if loss_delta > 1e-4:
    raise SystemExit(f"Train loss drift too high: {loss_delta:.6f}")
if sps_delta > 0.5:
    raise SystemExit(f"Train throughput drift too high: {sps_delta:.6f}")
PY

echo "[repro] verification complete."
