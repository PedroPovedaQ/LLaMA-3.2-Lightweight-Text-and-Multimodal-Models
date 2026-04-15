# Add New Model Checklist

Use this checklist to add a new model and connect it to the baseline benchmark pipeline in this repo.

## 1) Add model registry entry

Edit `scripts/experiment_utils.py` and add a new `MODEL_SPECS` entry:

```python
"smollm2-1.7b": ModelSpec(
    hf_model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    download_path="models/others/smollm2-1.7b",
    description="SmolLM2 1.7B Instruct model",
),
```

## 2) Expose model in download CLI choices

Edit `scripts/download_models.py` and add the same key to the `--models` `choices` list.

## 3) Download the model

```bash
python scripts/download_models.py --provider hf --models smollm2-1.7b
```

## 4) Run baseline accuracy benchmarks

```bash
python scripts/run_benchmarks.py \
  --model-key smollm2-1.7b \
  --benchmarks hellaswag arc gsm8k \
  --limit 20 \
  --precision fp16 \
  --device mps
```

Notes:
- Use `--device mps` on Apple Silicon.
- Use `--device cuda` on NVIDIA GPUs.
- Start with `--limit 20` for a smoke run, then scale up.

## 5) Run efficiency benchmarks

```bash
python scripts/run_efficiency.py \
  --model-key smollm2-1.7b \
  --precision fp16 \
  --device mps \
  --num-prompts 5 \
  --warmup-runs 2 \
  --timed-runs 3 \
  --max-new-tokens 64
```

## 6) Aggregate and regenerate report artifacts

```bash
python scripts/aggregate_results.py
```

Outputs:
- Raw JSON: `results/raw/`
- Aggregated CSV: `results/processed/metrics.csv`
- PDF report: `results/reports/project_snapshot_report.pdf`

## 7) Include model in snapshot PDF comparison

Edit `scripts/generate_report.py`:
- Add model ID to `TARGET_MODELS`
- Add label to `DISPLAY_NAME`
- Add color to `MODEL_COLORS`

Without this step, the model still benchmarks correctly, but it will not appear on the current fixed 3-model PDF pages.

## 8) Verification checklist

- [ ] `python scripts/download_models.py --help` shows your model key under `--models`.
- [ ] `python scripts/run_benchmarks.py --help` shows your model key under `--model-key`.
- [ ] `python scripts/run_efficiency.py --help` shows your model key under `--model-key`.
- [ ] One benchmark JSON and one efficiency JSON exist for your model in `results/raw/`.
- [ ] `results/processed/metrics.csv` contains rows for your model.

