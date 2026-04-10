#!/usr/bin/env bash
set -euo pipefail

PROFILE="dev"
FINAL_REPORT_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --final-report)
      FINAL_REPORT_FLAG="--final-report"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/run_lora.sh [--profile dev|final] [--final-report]"
      exit 1
      ;;
  esac
done

if [[ "${PROFILE}" == "dev" ]]; then
  CONFIG_PATH="configs/train_lora.dev.yaml"
elif [[ "${PROFILE}" == "final" ]]; then
  CONFIG_PATH="configs/train_lora.final.yaml"
else
  echo "Unsupported profile: ${PROFILE}. Use 'dev' or 'final'."
  exit 1
fi

echo "Starting LoRA training..."
echo "Config: ${CONFIG_PATH}"

echo "Running preflight data-path checks..."
python scripts/preflight_data_paths.py --config "${CONFIG_PATH}"

python -m src.training.lora_train --config "${CONFIG_PATH}" ${FINAL_REPORT_FLAG}

LATEST_RUN_DIR="$(ls -dt outputs/runs/*/train 2>/dev/null | head -n 1 || true)"

if [[ -n "${LATEST_RUN_DIR}" ]]; then
  echo "Training outputs: ${LATEST_RUN_DIR}"
  echo "Run metadata: ${LATEST_RUN_DIR}/params.json"
  echo "Run metrics: ${LATEST_RUN_DIR}/metrics.json"
else
  echo "No run directory found under outputs/runs/."
fi
