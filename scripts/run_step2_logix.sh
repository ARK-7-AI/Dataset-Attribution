#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/attribution_logix.yaml"
SMOKE_CONFIG_OUT=""
SMOKE_RUN_ID="smoke-logix"
SMOKE_OUTPUT_ROOT="outputs/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --generate-smoke-config)
      SMOKE_CONFIG_OUT="$2"
      shift 2
      ;;
    --smoke-run-id)
      SMOKE_RUN_ID="$2"
      shift 2
      ;;
    --smoke-output-root)
      SMOKE_OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_step2_logix.sh [--config <path>] [--generate-smoke-config <path>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/run_step2_logix.sh [--config <path>] [--generate-smoke-config <path>]"
      exit 1
      ;;
  esac
done

if [[ -n "${SMOKE_CONFIG_OUT}" ]]; then
  # No source patching required; engine initializes with logix.init(project=project_name).
  # For mixed-run layouts, set run_id and explicit *_manifest_path overrides in the generated config.
  python - <<'PY' "${SMOKE_CONFIG_OUT}" "${SMOKE_RUN_ID}" "${SMOKE_OUTPUT_ROOT}"
from pathlib import Path
import sys
import yaml

dst = Path(sys.argv[1])
run_id = str(sys.argv[2])
output_root = str(sys.argv[3])
payload = {
    "run_id": run_id,
    "project_name": "dataset_attribution",
    "output_root": output_root,
    "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
    "seed": 2026,
    "top_k": 10,
    "train_subset_size": 16,
    "influence": {
        "mode": "ihvp",
        "ihvp": {"damping": 0.01, "scale": 25.0, "recursion_depth": 16, "num_samples": 1},
    },
    "lora": {"lora_only": True},
    "logix": {"project": "dataset_attribution", "setup": {}, "run": {}},
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(f"[step2] Generated smoke config: {dst}")
PY
  CONFIG_PATH="${SMOKE_CONFIG_OUT}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}"
  echo "Fix: pass a valid attribution config with --config."
  exit 1
fi

if ! python -c "import logix" >/dev/null 2>&1; then
  echo "Python package 'logix' is not importable in this environment."
  echo "Fix: install project dependencies that include logix-ai, then retry."
  exit 1
fi

echo "[step2] Running LogIX attribution"
echo "[step2] config=${CONFIG_PATH}"

python -m src.attribution.logix_engine --config "${CONFIG_PATH}"

echo "[step2] LogIX attribution finished"
