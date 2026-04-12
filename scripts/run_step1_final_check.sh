#!/usr/bin/env bash
set -euo pipefail

RUN_ID="default_run"
OUTPUT_ROOT="outputs/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_step1_final_check.sh [--run-id <run_id>] [--output-root <path>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/run_step1_final_check.sh [--run-id <run_id>] [--output-root <path>]"
      exit 1
      ;;
  esac
done

TRAIN_DIR="${OUTPUT_ROOT}/${RUN_ID}/train"

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Training directory not found: ${TRAIN_DIR}"
  echo "Fix: run training first (e.g., bash scripts/run_lora.sh --profile final --final-report)."
  exit 1
fi

echo "[step1] Running final-run gate"
echo "[step1] train_dir=${TRAIN_DIR}"

python scripts/final_run_check.py --train-dir "${TRAIN_DIR}"

echo "[step1] Final-run gate complete"
echo "[step1] Report: ${TRAIN_DIR}/final_run_check.json"
