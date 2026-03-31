# Dataset-Attribution

Dataset attribution for LoRA fine-tuned LLMs: accuracy, efficiency, and privacy evaluation using influence functions, gradient similarity, and retrieval methods.

## Project layout

- `src/` — source modules for data, training, attribution, privacy, and evaluation.
- `configs/` — YAML configuration files for experiments.
- `scripts/` — helper scripts for automation and orchestration.
- `tests/` — test suite.
- `outputs/` — generated artifacts (checkpoints, logs, reports).

## Entrypoint convention

Use Python module execution for all runnable jobs:

```bash
python -m src.training.lora_train --config configs/train_lora.yaml
```

This keeps execution consistent across local runs, CI, and future script wrappers.
