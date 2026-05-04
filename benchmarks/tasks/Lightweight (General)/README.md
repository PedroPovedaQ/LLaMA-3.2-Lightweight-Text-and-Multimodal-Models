# Llama 3.2 Lightweight Model Benchmark Replication

Replication of the **lightweight instruction-tuned benchmarks** from Meta's [Llama 3.2 release](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/).

## Models Evaluated

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Llama 3.2 1B Instruct | 1.2B | `meta-llama/Llama-3.2-1B-Instruct` |
| Llama 3.2 3B Instruct | 3.2B | `meta-llama/Llama-3.2-3B-Instruct` |
| Phi-3.5-mini Instruct | 3.8B | `microsoft/Phi-3.5-mini-instruct` |
| Gemma 2 2B IT | 2.6B | `google/gemma-2-2b-it` |

## Benchmarks

| Benchmark | Protocol | Metric |
|-----------|----------|--------|
| MMLU | 5-shot | Accuracy (%) |
| Open-rewrite eval | 0-shot | ROUGE-L |
| TLDR9+ | 1-shot | ROUGE-L |
| IFEval | 0-shot | Prompt-level strict accuracy (%) |

## Results

| Benchmark | Llama 1B (Ours/Paper) | Llama 3B (Ours/Paper) | Phi-3.5 (Ours/Paper) | Gemma 2B (Ours/Paper) |
|---|---|---|---|---|
| MMLU (5-shot) | 45.6 / 49.3 | 60.9 / 63.4 | 69.2 / 69.0 | 57.1 / 57.8 |
| Open-rewrite | 34.6 / 41.6 | 34.6 / 40.1 | 32.8 / 34.5 | 15.6 / 31.2 |
| TLDR9+ | 8.8 / 16.8 | 7.6 / 19.0 | 7.2 / 12.8 | 8.6 / 13.9 |
| IFEval | 64.0 / 59.5 | 76.0 / 77.4 | 55.5 / 59.2 | 63.8 / 61.9 |

## Project Structure

```
├── config.py              # Model registry, benchmark configs, prompt templates
├── model_loader.py        # Unified model/tokenizer loading and text generation
├── utils.py               # Answer extraction, ROUGE scoring, result I/O
├── eval_mmlu.py           # MMLU 5-shot evaluation
├── eval_open_rewrite.py   # Open-rewrite eval (ROUGE-L)
├── eval_tldr.py           # TLDR9+ summarization eval (ROUGE-L)
├── eval_ifeval.py         # IFEval instruction-following eval
├── run_all.py             # Runs all 4 benchmarks for one model
├── compare_results.py     # Generates comparison tables and charts
├── slurm_job.sh           # SLURM script for UCF Newton cluster
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Prerequisites

- Accept model licenses on HuggingFace:
  - [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
  - [Gemma 2](https://huggingface.co/google/gemma-2-2b-it)
- Create a HuggingFace [access token](https://huggingface.co/settings/tokens)

### 2. Environment Setup

```bash
conda create -n llama_eval python=3.10 -y
conda activate llama_eval
pip install -r requirements.txt

# IMPORTANT: Install PyTorch with V100 support (if using V100 GPUs)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Download NLTK data for ROUGE scoring
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Login to HuggingFace
hf auth login
```

### 3. Running Evaluations

```bash
# Smoke test (3 examples, 1 model, 1 benchmark)
python run_all.py --model llama-3.2-1b --max_samples 3 --benchmarks mmlu

# Single model, all benchmarks
python run_all.py --model llama-3.2-1b

# Single benchmark, single model
python eval_mmlu.py --model llama-3.2-3b

# All 4 models via SLURM (UCF Newton)
sbatch slurm_job.sh

# Generate comparison after all models are evaluated
python compare_results.py \
    --results results/llama-3.2-1b results/llama-3.2-3b \
              results/phi-3.5-mini results/gemma-2-2b
```

## Notes

- MMLU and IFEval scores are within 2-4% of the paper, which is expected for a replication with different prompt formatting and answer extraction.
- Open-rewrite and TLDR9+ scores are lower than the paper because: (1) the exact Open-rewrite benchmark may be Meta-internal, and (2) TLDR uses chat-formatted generation which produces longer responses that score lower on ROUGE-L.
- Phi-3.5-mini uses `float16` precision to fit on 16GB V100 GPUs.
- The `slurm_job.sh` script is configured for UCF Newton cluster (partition=normal, conda path at /apps/anaconda/).

## Hardware

Evaluated on UCF Newton cluster:
- Tesla V100-PCIE-16GB / V100-PCIE-32GB
- All models fit on a single GPU
- Total evaluation time: ~10 hours for all 4 models across all 4 benchmarks
