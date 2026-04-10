# Final-run checklist (pre-attribution gate)

Use this checklist after LoRA training and before starting attribution (`src.attribution.*`).
A run is considered **eligible for attribution** only when all checks pass.

## Objective checklist criteria

1. **Minimum training progress reached**
   - `metrics.json.steps >= 100`
   - `metrics.json.epochs_completed >= 1.0`

2. **Acceptable loss behavior**
   - `metrics.json.train_loss` is finite and `<= 3.0`
   - If `trainer_state.json.log_history` contains loss points, the final logged loss must be less than or equal to the first logged loss (non-divergent trend).

3. **Required artifacts present**
   - `adapter/` (with LoRA weights)
   - `tokenizer/` (with tokenizer files)
   - `metrics.json`
   - `params.json`
   - `trainer_state.json`

4. **Required runtime metrics captured**
   - `train_runtime`
   - `train_loss`
   - `steps`
   - `epochs_completed`
   - `train_steps_per_second`
   - `train_tokens_per_second`
   - `timing_breakdown_s`
   - `throughput.steps_per_second`
   - `throughput.tokens_per_second`

5. **Required downstream validation pass**
   - `validate_training_outputs` must pass for the run, confirming artifact/schema compatibility expected by downstream stages.

## Automated gate script

Run:

```bash
python scripts/final_run_check.py --train-dir outputs/runs/<run_id>/train
```

The script writes `final_run_check.json` into the same `train` directory by default and exits:

- `0` when all checks pass
- `1` when any check fails

### Optional strict thresholds

```bash
python scripts/final_run_check.py \
  --train-dir outputs/runs/<run_id>/train \
  --min-steps 200 \
  --min-epochs 1.5 \
  --max-final-loss 2.5
```

Use stricter values for report-grade runs when needed.
