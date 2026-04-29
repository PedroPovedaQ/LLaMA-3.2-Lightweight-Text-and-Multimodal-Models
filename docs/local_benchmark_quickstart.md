# Local Benchmark Quickstart

Use this quickstart if you want to run the repository locally with the Hugging Face-backed pipeline.

This guide assumes:

- you are using Hugging Face models and datasets
- you want to run `scripts/run_benchmarks.py`
- you want the default per-run outputs: raw JSON, PNG charts, and a PDF summary

## 1) Clone and install

```bash
git clone https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models.git
cd LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
python scripts/setup.py
pip install -r requirements.txt
```

## 2) Authenticate with Hugging Face

Some models and datasets require authentication.

```bash
hf auth login
```

Notes:

- `gpqa` is a gated Hugging Face dataset, so it will not run unless your HF account has access
- some LLaMA models may also require Hugging Face access approval

## 3) Download a model in HF format

The simplest path is to use the repo's download script with the Hugging Face provider.

Example:

```bash
python scripts/download_models.py --provider hf --models llama-3.2-1b
```

This writes a local HF-format checkpoint under `models/`.

Common examples:

```bash
python scripts/download_models.py --provider hf --models llama-3.2-1b
python scripts/download_models.py --provider hf --models llama-3.2-3b
python scripts/download_models.py --provider hf --models phi-3-mini
```

## 4) Run a smoke benchmark

Start small first.

```bash
python scripts/run_benchmarks.py \
  --model-id ./models/llama-3.2-1b \
  --benchmarks hellaswag arc \
  --limit 5 \
  --precision fp16 \
  --device auto
```

Why `--model-id ./models/...` instead of a remote HF ID:

- it uses the local checkpoint you already downloaded
- it avoids unnecessary model lookups against the Hugging Face Hub
- it is the most reliable local path for repeatable runs

## 5) Find the outputs

Each run now writes:

- raw JSON: `results/raw/`
- accuracy PNG: `results/reports/`
- runtime PNG: `results/reports/`
- per-run PDF summary: `results/reports/`

Typical files look like:

- `results/raw/benchmark_<run_id>.json`
- `results/reports/benchmark_<run_id>_accuracy.png`
- `results/reports/benchmark_<run_id>_runtime.png`
- `results/reports/benchmark_<run_id>_report.pdf`

## 6) Run the baseline reasoning set

Once the smoke test works, run the broader reasoning baseline:

```bash
python scripts/run_benchmarks.py \
  --model-id ./models/llama-3.2-1b \
  --benchmarks hellaswag arc gpqa gsm8k \
  --limit 20 \
  --precision fp16 \
  --device auto
```

Important:

- `gpqa` requires Hugging Face access because the dataset is gated
- on CPU-only machines, start with a smaller `--limit`

## 7) Compare multiple local models

Example:

```bash
python scripts/run_benchmarks.py \
  --model-id ./models/llama-3.2-3b \
  --benchmarks hellaswag arc \
  --limit 20 \
  --precision fp16 \
  --device auto

python scripts/run_benchmarks.py \
  --model-id ./models/phi-3-mini \
  --benchmarks hellaswag arc \
  --limit 20 \
  --precision fp16 \
  --device auto
```

If you also want the project-level PDF snapshot across models, use:

```bash
python scripts/generate_report.py
```

That snapshot report expects the baseline benchmark set and matching efficiency runs described in [benchmarks.md](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/docs/benchmarks.md).

## 8) Common issues

`DatasetNotFoundError` or gated dataset failures:

- run `hf auth login`
- confirm your HF account has access to the dataset or model

Model download worked, but benchmark run still touches the network:

- use `--model-id ./models/<local-dir>` instead of a remote HF model ID

Runs are too slow:

- reduce `--limit`
- use `--device cuda` on NVIDIA GPUs or `--device mps` on Apple Silicon when available

You only want raw JSON and no images/PDF:

- pass `--no-report` to `scripts/run_benchmarks.py`
