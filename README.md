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

For LogIX attribution integration, use:

```bash
python -m src.attribution.logix_engine --config configs/attribution.yaml
```

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

Final report runs enforce repository traceability policy:

- Default (`reporting.final_repo_state_policy: fail_dirty`): abort the run if the git checkout is dirty.
- Optional (`reporting.final_repo_state_policy: capture_diff`): allow dirty checkout, but persist
  `outputs/runs/<run_id>/train/repo_state.diff.txt` containing full `git status` + `git diff HEAD`.

This guarantees that every report-grade run is either pinned to a clean commit or carries a
replayable checkout diff artifact for auditability.

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
- For final report runs: selected repo-state policy and optional `repo_state.diff.txt` artifact path.

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

## One-command wrappers for attribution steps

Use the wrapper scripts below to run the pre-attribution gate and LogIX attribution in one command.

### Step 1: final-run gate wrapper

- Script: `scripts/run_step1_final_check.sh`
- Prerequisites:
  - A completed training run exists under `outputs/runs/<run_id>/train`.
  - Python environment dependencies from `pyproject.toml` are installed.
- Exact command:

```bash
bash scripts/run_step1_final_check.sh --run-id <run_id>
```

- Expected runtime characteristics:
  - Usually CPU-only and fast (typically a few seconds to ~1 minute), because it validates artifacts and metrics JSON files.
  - Runtime scales mainly with artifact/log file size in the train directory.
- Common failure modes and fixes:
  - `Training directory not found`: confirm `run_id` and `output_root`, or run training first.
  - Missing required artifacts/metrics in gate output: rerun training and ensure it completes cleanly.
  - Validation failures from strict thresholds: inspect `final_run_check.json` and, if intentional, rerun `scripts/final_run_check.py` with explicit thresholds.

### Step 2: LogIX attribution wrapper

- Script: `scripts/run_step2_logix.sh`
- Prerequisites:
  - Step 1 gate has passed for the same run.
  - `logix` Python package is installed (via project dependencies including `logix-ai`), with required contract `logix>=0.1.1`.
  - Compatibility note (tested in this repository): the engine supports modern module-level `logix.setup(...)` + `logix.run(...)`/`logix.execute(...)` paths and legacy `0.1.1`-style setup wrappers requiring `log_option_kwargs`; execute-phase dispatch probes context/module entry points in deterministic order and retries strict-kwargs rejections after dropping unsupported keys.
  - Attribution config exists (default: `configs/attribution_logix.yaml`) and points to valid run artifacts/manifests.
  - LogIX runtime initialization is required before extraction/influence calls; the engine performs explicit `logix.init(project=...)` startup using deterministic project precedence: top-level `project_name`, then `logix.project`, then fallback `dataset_attribution`.
  - Standard execution does **not** require a root-level `config.yaml` in the repository.
  - Canonical initialization keys are `project_name` (preferred) or `logix.project`. Legacy `logix.init` usage is deprecated and rejected by the engine.
  - The engine fails fast when the resolved LogIX project is empty/invalid and logs effective `project` + `run_id` at startup.
  - `shadow_manifest_path` is currently informational-only in this LogIX engine path (validated for existence and logged), and is not threaded into scoring payloads.
  - Default behavior uses one `run_id` for splits and training artifacts, but mixed-run layouts are supported when explicit manifest paths are set (for example: training artifacts from `final_report_run` with split CSVs from `default_run`).
- Exact command:

```bash
bash scripts/run_step2_logix.sh --config configs/attribution_logix.yaml
```

- Smoke config workflow (no in-place edits to `configs/attribution_logix.yaml`):

```bash
bash scripts/run_step2_logix.sh --generate-smoke-config /tmp/attribution_logix_smoke.yaml --smoke-run-id final_report_run --smoke-output-root outputs/runs
```

Then patch only `/tmp/attribution_logix_smoke.yaml` for mixed-run manifests (set `train_manifest_path`, `test_manifest_path`, and `shadow_manifest_path` to `outputs/runs/default_run/splits/*.csv`) and run:

```bash
bash scripts/run_step2_logix.sh --config /tmp/attribution_logix_smoke.yaml
```

- Expected runtime characteristics:
  - Typically longer than the gate check (often several minutes to tens of minutes) depending on model size, hardware, and `train_subset_size`.
  - Runtime is sensitive to IHVP controls (`recursion_depth`, `num_samples`) and available accelerator resources.
- Common failure modes and fixes:
  - `Python package 'logix' is not importable`: install dependencies that include `logix-ai`.
  - `LogIX is not initialized` / init failure: verify config includes a valid `run_id`, set `project_name` (or rely on default), and confirm the installed LogIX package version supports `logix.init(...)`.
  - Config not found / invalid config keys: verify `--config` path and required keys in YAML (`run_id`, `output_root`, `top_k`, `train_subset_size`, IHVP controls).
  - Missing train artifacts or manifest paths: ensure training outputs and split manifests exist for the configured `run_id`.
  - OOM or slow execution: lower `train_subset_size`, reduce IHVP recursion/sample settings, or use a more capable GPU.

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
