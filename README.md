# Dataset-Attribution

Dataset attribution for LoRA fine-tuned LLMs: accuracy, efficiency, and privacy evaluation using influence functions, gradient similarity, and retrieval methods.

## Project layout

- `src/` — source modules for data, training, attribution, privacy, and evaluation.
- `configs/` — YAML configuration files for experiments.
- `scripts/` — helper scripts for automation and orchestration.
- `tests/` — test suite.
- `outputs/` — generated artifacts (checkpoints, logs, reports).

## Code vs Data locations

Use this repository path convention to keep code and datasets separate:

- `src/data/` -> Python modules only (splitters, loaders, schemas). Do **not** store datasets here.
- `data/raw/` -> canonical source datasets (for example: `data/raw/alpaca_data.json`).
- `data/processed/` -> optional transformed artifacts only when actively used by a pipeline.
- `outputs/runs/<run_id>/splits/` -> generated split manifests (`train.csv`, `test.csv`, `shadow.csv`).

Examples:

- Split config dataset source: `dataset.path: data/raw/alpaca_data.json`
- Train config source dataset: `data.dataset_json_path: data/raw/alpaca_data.json`
- Train config manifests: `data.train_manifest_path: outputs/runs/<run_id>/splits/train.csv`

## Entrypoint convention

Use Python module execution for all runnable jobs:

```bash
python -m src.training.lora_train --config configs/train_lora.dev.yaml
```

This keeps execution consistent across local runs, CI, and future script wrappers.

## Dataset split input formats

`src.data.split.read_dataset()` auto-detects the dataset format by file extension:

- `.csv`: standard CSV table with headers.
- `.json`: either
  - a list of objects, e.g. `[{"sample_id": "...", "source": "...", "license": "..."}]`, or
  - a wrapped object with `data`, e.g. `{"data": [{"sample_id": "...", "source": "...", "license": "..."}]}`.

If `sample_id`, `source`, or `license` are missing, normalization auto-fills:
- `sample_id`: `{dataset_name}-{row_index:06d}` (example: `alpaca-000123`)
- `source`: `{dataset_name}` (example: `alpaca`)
- `license`: `unknown`

During training ingestion, JSON rows are validated strictly:
- each row must be a JSON object (dict),
- each row must include at least one non-empty text payload field used for instruction tuning,
- errors include the failing row index for fast debugging.

Optional: set `data.normalized_snapshot_path` (for example `data/processed/alpaca_normalized.json`) in training config to persist the normalized records used for split-manifest joins.

### Training ingestion schema mapping (including Alpaca)

The LoRA training loader supports configurable field mapping under `data` in the training profiles:

- `configs/train_lora.dev.yaml`
- `configs/train_lora.final.yaml`

- `prompt_field` (required): primary instruction/prompt key.
- `input_field` (optional): additional context key (for Alpaca's `input`), appended into prompt text when present.
- `response_field` (required): primary target/answer key.
- `response_fallback_fields` (optional list): fallback keys tested in order if `response_field` is missing/empty.
- `normalize_response_key` (optional bool): when enabled, preprocessing writes canonical `response` for every row using `response_field` + fallbacks.

Supported JSON schema examples:

- Existing prompt/response style:
  - `{"prompt": "...", "response": "..."}`
- Alpaca default style:
  - `{"instruction": "...", "input": "...", "output": "..."}`

Recommended Alpaca mapping:

- `prompt_field: instruction`
- `input_field: input`
- `response_field: response`
- `response_fallback_fields: [output, answer, completion]`

If no non-empty target is found across `response_field` + `response_fallback_fields`, training fails fast with a `ValueError` that includes the row `sample_id` and all tested field names.

## LoRA data pipeline run order

Run data splitting **before** training so manifests exist at `outputs/runs/<run_id>/splits/*.csv` and training can resolve `data.train_manifest_path` / `data.test_manifest_path`.

1. Split stage (first): load `configs/data.yaml`, select/subsample rows if configured, then write `train.csv`, `test.csv`, and `shadow.csv` into `outputs/runs/<run_id>/splits/`.

- `dataset.subset_size` must be less than or equal to the source dataset row count; otherwise split generation fails fast with a clear error.
- Use `configs/data.yaml` for the bundled sample dataset, or `configs/data_3000_template.yaml` when running the 3000-row architecture (2400/300/300).
2. Training stage (second): load either `configs/train_lora.dev.yaml` (smoke) or `configs/train_lora.final.yaml` (report), resolve `<run_id>` inside manifest paths, and train against the generated split manifests.

3. Artifact validation (automatic final step): `src.training.lora_train` now runs output validation and fails fast if required train artifacts/schema are incomplete. You can also run it manually with `python scripts/validate_train_outputs.py --train-dir outputs/runs/<run_id>/train`.


## Baseline model selection (ungated)

- Primary baseline model: `Qwen/Qwen2.5-3B-Instruct` (decoder-only causal LM, ungated).
- Fallback baseline model: `microsoft/Phi-3.5-mini-instruct` (also ungated, ~3.8B params).
- Reason for swap: avoid gated-access failures from `meta-llama/Llama-3.2-3B-Instruct` while keeping similar parameter scale for apples-to-apples LoRA experiments.

### Expected hardware footprint

For ~3B class instruct models with LoRA adapters:

- bf16/fp16 training (default config): typically ~16-24 GB VRAM depending on sequence length/checkpointing.
- optional 8-bit loading: often ~10-14 GB VRAM.
- optional 4-bit loading: often ~8-12 GB VRAM with possible throughput/quality trade-offs.

If memory is constrained, reduce `training.batch_size`, raise `training.gradient_accumulation_steps`, and consider `load_in_8bit`/`load_in_4bit`.

## Exact commands to start LoRA training

Dev smoke profile (low-cost sanity check):

```bash
python -m src.training.lora_train --config configs/train_lora.dev.yaml
```

Final report-grade profile (required for report references):

```bash
python -m src.training.lora_train --config configs/train_lora.final.yaml --final-report
```

Equivalent wrapper script:

```bash
bash scripts/run_lora.sh --profile dev
bash scripts/run_lora.sh --profile final --final-report
```

## Training dependency compatibility (pinned)

To reduce breakage from upstream API shifts in Hugging Face Trainer, this project pins:

- `transformers>=4.46,<4.49`
- `accelerate>=1.0,<1.3`

These ranges are validated with the LoRA training pipeline and regression tests in `tests/test_lora_train.py`. At startup, `src.training.lora_train` logs the effective runtime versions for `transformers`, `accelerate`, and `peft` to make debugging easier.

## Reproducibility layer

Each training run now persists reproducibility metadata under `outputs/runs/<run_id>/train/params.json` (and mirrors it in `resolved_config.yaml`) in a `reproducibility` block:

- Python version used to execute training.
- Library versions: `transformers`, `peft`, `accelerate`, and `torch`.
- CUDA runtime details (availability, device count, device names, CUDA version reported by torch).
- Deterministic controls and seed state (`PYTHONHASHSEED`, NumPy/Torch seeding flags, cuDNN deterministic/benchmark mode).
- Git checkout metadata (`git_commit_hash`, `git_dirty`).

Training enforces deterministic defaults by:

- Applying a single seed to Python, NumPy (if installed), and Torch/CUDA.
- Enabling deterministic Torch algorithms when available.
- Forcing `dataloader_num_workers=0` and setting both `seed` and `data_seed` in `TrainingArguments`.
- Setting `CUBLAS_WORKSPACE_CONFIG` when absent.

### Verify reproducibility across reruns

Use the verification script to run a short profile twice and compare artifact schema + training deltas:

```bash
bash scripts/verify_reproducibility.sh configs/profiles/colab_train_lora.yaml
```

What it checks:

- Required artifact presence in both runs (`params.json`, `metrics.json`, `trainer_state.json`, `resolved_config.yaml`, adapter/tokenizer outputs).
- Stable `params.json` schema across reruns.
- Near-identical metrics (`steps` equal, low `train_loss` drift, bounded throughput drift).
- SHA256 hash comparison for key artifacts to quickly spot differences.

## Expected output artifacts and where to find `run_id`

After training finishes, artifacts are written under:

- `outputs/runs/<run_id>/train/params.json` — resolved run parameters (includes the canonical `run_id`).
- `outputs/runs/<run_id>/train/metrics.json` — summarized training metrics.
- `outputs/runs/<run_id>/train/trainer_state.json` — Trainer state/log history.
- `outputs/runs/<run_id>/train/resolved_config.yaml` — frozen config used by this run.
- `outputs/runs/<run_id>/train/adapter/` — LoRA adapter weights.
- `outputs/runs/<run_id>/train/tokenizer/` — tokenizer files used/saved during training.
- `outputs/runs/<run_id>/train/checkpoints/` — Hugging Face Trainer checkpoints.

How to find `run_id`:

- If `run_id` is explicitly set in the selected profile config (`configs/train_lora.dev.yaml` or `configs/train_lora.final.yaml`), that value is used.
- If `auto_run_id: true` and `run_id: null`, a UTC timestamp+suffix is generated.
- Read `outputs/runs/<run_id>/train/params.json` and inspect its `run_id` field (or use the path printed at the end of training).

## Final-run gate (required before attribution)

Before starting the attribution phase, run the final gate to mark the training run as pass/fail against objective criteria:

```bash
python scripts/final_run_check.py --train-dir outputs/runs/<run_id>/train
```

The gate checks:

- minimum training progress (steps + epochs),
- acceptable loss behavior,
- required train artifacts,
- required runtime metrics capture,
- downstream output validation pass.

Detailed checklist and thresholds are documented in `docs/final_run_checklist.md`.

## GPU/CPU notes and troubleshooting

### Device selection

- `device_map: auto` (default in both training profile configs) lets Transformers place model weights automatically, typically preferring GPU when available.
- In single-GPU training without quantization/offload, `src.training.lora_train` now omits `device_map` and keeps the model fully on CUDA.
- A Colab-ready profile is available at `configs/profiles/colab_train_lora.yaml`; it sets `device_map: null` to avoid `device_map: auto` during training runs.
- For quick, stable LR comparisons, use `configs/profiles/colab_lr_sweep_stable.yaml` (2e-4, 1e-4, 5e-5) with fixed `seed: 42` and `max_steps: 120`.
- Keep the default Colab LR at `2e-4` unless run metrics show clear evidence that a higher LR is better.
- For CPU-only runs, set `device_map: cpu` in the training config.
- Startup logs print the final placement decision (`resolved_device_map`, placement flags, and parameter-device summary).

### OOM (Out Of Memory)

If you hit CUDA OOM or host RAM pressure:

- Reduce `training.batch_size`.
- Increase `training.gradient_accumulation_steps` to preserve effective batch size.
- Reduce `training.max_seq_len`.
- Prefer a smaller base model.
- Use lower precision (`torch_dtype: float16` or `bfloat16` when hardware supports it).

### Tokenizer pad token

If your tokenizer has no pad token, this training entrypoint auto-falls back to EOS by setting:

- `tokenizer.pad_token = tokenizer.eos_token`

This avoids common collation/padding failures for decoder-only models.

### Dtype settings (`torch_dtype`, `fp16`, `bf16`)

- `torch_dtype` controls model load dtype and supports: `auto`, `float16`, `bfloat16`, `float32`.
- In `training`, enable **exactly one** of:
  - `fp16: true` (with `bf16: false`), or
  - `bf16: true` (with `fp16: false`).
- If your GPU does not support bf16, use fp16 or float32.
- If you see instability/NaNs, try `torch_dtype: float32`, disable mixed precision, and lower the learning rate.
