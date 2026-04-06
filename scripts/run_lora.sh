#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/train_lora.yaml"

echo "Starting LoRA training..."
echo "Config: ${CONFIG_PATH}"

python -m src.training.lora_train --config "${CONFIG_PATH}"

LATEST_RUN_DIR="$(ls -dt outputs/runs/*/train 2>/dev/null | head -n 1 || true)"

if [[ -n "${LATEST_RUN_DIR}" ]]; then
  echo "Training outputs: ${LATEST_RUN_DIR}"
  echo "Run metadata: ${LATEST_RUN_DIR}/params.json"
  echo "Run metrics: ${LATEST_RUN_DIR}/metrics.json"
else
  echo "No run directory found under outputs/runs/."
fi
