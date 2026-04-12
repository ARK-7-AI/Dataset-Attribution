#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/attribution_logix.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_step2_logix.sh [--config <path>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash scripts/run_step2_logix.sh [--config <path>]"
      exit 1
      ;;
  esac
done

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
