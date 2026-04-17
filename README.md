# Dataset-Attribution

A focused workflow for preparing data and running the **final LoRA training** job used in attribution experiments.

## 1) Configs and data

This repo uses three core config files:

- `configs/data.yaml`  
  Controls dataset input and split generation.
- `configs/data_3000_template.yaml`  
  Template for a 3000-row split setup (2400 train / 300 test / 300 shadow).
- `configs/train_lora.final.yaml`  
  Final report-grade LoRA training configuration.

### Data locations

- Raw dataset: `data/raw/` (example: `data/raw/alpaca_data.json`)
- Optional processed snapshots: `data/processed/`
- Generated split manifests: `outputs/runs/<run_id>/splits/`

Keep datasets in `data/` and keep Python code in `src/`.

## 2) Data processing (split generation)

Run split generation first so training manifests exist:

```bash
python -m src.data.split --config configs/data.yaml
```

Expected output:

- `outputs/runs/<run_id>/splits/train.csv`
- `outputs/runs/<run_id>/splits/test.csv`
- `outputs/runs/<run_id>/splits/shadow.csv`

## 3) Final LoRA training

After splits are ready, start final training:

```bash
python -m src.training.lora_train --config configs/train_lora.final.yaml --final-report
```

Equivalent wrapper:

```bash
bash scripts/run_lora.sh --profile final --final-report
```

## 4) Minimal run order

1. Prepare/confirm dataset in `data/raw/`.
2. Generate splits from `configs/data.yaml`.
3. Run final LoRA training with `configs/train_lora.final.yaml`.

That’s the full required path for final training and downstream attribution work.
