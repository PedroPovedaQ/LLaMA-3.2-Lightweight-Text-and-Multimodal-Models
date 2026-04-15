# Llama CLI Skill

Use this skill when you want to download Meta Llama checkpoints with the `llama-model` CLI and run this project's benchmark scripts against local checkpoint paths.

## Goal

- Use `llama-model` as an alternative model acquisition path.
- Keep benchmark execution in this repo unchanged (`scripts/run_benchmarks.py`, `scripts/run_efficiency.py`).

## Prerequisites

1. Install CLI:
```bash
python -m pip install -U llama-models
```
2. Verify CLI:
```bash
llama-model --help
```
3. Accept the Llama license terms.

## Discover model IDs

```bash
llama-model list --show-all
llama-model describe -m Llama3.2-1B-Instruct
```

## Download with llama-model directly

### Source: Hugging Face
```bash
llama-model download \
  --source huggingface \
  --model-id Llama3.2-1B-Instruct \
  --hf-token "$HF_TOKEN"
```

### Source: Meta signed URL
```bash
llama-model download \
  --source meta \
  --model-id Llama3.2-1B-Instruct \
  --meta-url 'https://...llamameta.net/*?...'
```

## Download through this repo script

Default command (now `llama-cli` provider):
```bash
python scripts/download_models.py
```

This pulls `llama-3.2-1b` and `llama-3.2-3b` by default.

```bash
python scripts/download_models.py \
  --provider llama-cli \
  --models llama-3.2-1b llama-3.2-3b \
  --llama-source huggingface \
  --hf-token "$HF_TOKEN"
```

For Meta source:
```bash
python scripts/download_models.py \
  --provider llama-cli \
  --models llama-3.2-1b \
  --llama-source meta \
  --meta-url 'https://...llamameta.net/*?...'
```

The script writes pointer manifests to:
- `models/llama-3.2-1b/LLAMA_CLI_MODEL.json`
- `models/llama-3.2-3b/LLAMA_CLI_MODEL.json`

## Run benchmarks with downloaded local checkpoints

If the directory contains HF-formatted weights (`model.safetensors` or `pytorch_model.bin`), use the `local_checkpoint_dir` from the manifest as `--model-id`.

```bash
python scripts/run_benchmarks.py \
  --model-id ~/.llama/checkpoints/Llama3.2-1B-Instruct \
  --benchmarks hellaswag arc gsm8k \
  --limit 20 --device mps --precision fp16
```

```bash
python scripts/run_efficiency.py \
  --model-id ~/.llama/checkpoints/Llama3.2-1B-Instruct \
  --device mps --precision fp16 \
  --num-prompts 5 --warmup-runs 1 --timed-runs 2 --max-new-tokens 64
```

## Known caveat

On some Python environments, `llama-model download` can fail with:
- `No module named 'llama_models.cli.model'`

This repository includes `scripts/llama_cli_download_wrapper.py` and routes `--provider llama-cli` through it to patch that import issue automatically.

Also note:
- `llama-model` commonly downloads `original/consolidated.00.pth` checkpoints.
- The current benchmark runner uses Transformers and expects HF-formatted weights (`model.safetensors` or `pytorch_model.bin`).
- If you pass an original-only checkpoint directory, the runner now exits with a clear format mismatch message.

If you need a benchmark-ready path immediately, use:
- `python scripts/download_models.py --provider hf ...`
- or `python scripts/download_models.py --provider ollama ...`

while keeping benchmark commands unchanged.
