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

## Dataset split input formats

`src.data.split.read_dataset()` auto-detects the dataset format by file extension:

- `.csv`: standard CSV table with headers.
- `.json`: either
  - a list of objects, e.g. `[{"sample_id": "...", "source": "...", "license": "..."}]`, or
  - a wrapped object with `data`, e.g. `{"data": [{"sample_id": "...", "source": "...", "license": "..."}]}`.

If `sample_id`, `source`, or `license` are missing, defaults are auto-filled as:
- `sample_id`: `sample-{row_index}`
- `source`: `unknown`
- `license`: `unknown`
