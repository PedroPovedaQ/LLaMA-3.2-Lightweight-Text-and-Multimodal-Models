# LLaMA 3.2: Lightweight Text and Multimodal Models


**Evaluating the accuracy-latency Pareto frontier for lightweight language models on consumer hardware**
### Team 9: Pedro Poveda |	Joel Gonzalez |	Jose Gabriel Gonzalez Nunez |	Matthew Horvath

## Paper Information

- **Title**: LLaMA 3.2: Lightweight Text and Multimodal Models
- **Authors**: Meta AI
- **Venue**: Meta Release, September 2024
- **Description**: Releases 1B and 3B parameter language models that achieve strong performance for their size class, specifically designed for on-device deployment. The 3B model approaches the quality of the original LLaMA-2-7B while being more than 2x smaller. Also includes 11B and 90B vision-language models with efficient visual token processing.
- **Project angle**: Benchmark LLaMA-3.2-1B and 3B against Phi-3-mini and other small models on standard benchmarks, deploy on consumer hardware with quantization (4-bit, 2-bit), and evaluate the accuracy-latency Pareto frontier for on-device use cases.

## 🎯 Project Overview

This project benchmarks LLaMA 3.2's lightweight models (1B & 3B parameters) against other compact language models, focusing on their suitability for on-device deployment. We evaluate the trade-offs between accuracy and inference speed to identify optimal models for resource-constrained environments.

### Key Research Questions
- How do LLaMA 3.2 variants compare to Phi-3-mini and other small models?
- What is the accuracy-latency Pareto frontier for on-device deployment?
- How does quantization (4-bit, 2-bit) affect performance across models?

## 📊 Models Under Evaluation

| Model | Parameters | Focus |
|-------|------------|-------|
| **LLaMA 3.2-1B** | 1B | Ultra-lightweight baseline |
| **LLaMA 3.2-3B** | 3B | Balanced mobile deployment |
| **Phi-3-mini** | 3.8B | Microsoft's efficient alternative |
| **Gemma 2B** | 2B | Google's compact model |
| **TinyLlama** | 1.1B | Community ultra-small model |

## 🧪 Evaluation Framework

### Performance Benchmarks
- **MMLU** - Multitask language understanding
- **GSM8K** - Mathematical reasoning
- **HellaSwag** - Commonsense reasoning
- **ARC-Challenge** - Reading comprehension

### Efficiency Metrics
- **Latency** - Time per token (ms/token)
- **Throughput** - Tokens per second
- **Memory Usage** - Peak RAM consumption
- **Quantization Impact** - FP16 vs INT4 vs INT2 performance

## 🚀 Quick Start

### Option A: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/llama-3.2-comparison/blob/main/notebooks/colab_setup.ipynb)

1. Open the Colab setup notebook
2. Run all cells to clone repo and install dependencies  
3. Start benchmarking with free GPU access

### Option B: Local Setup
```bash
git clone https://github.com/yourusername/llama-3.2-comparison.git
cd llama-3.2-comparison
python scripts/setup.py
pip install -r requirements.txt
python scripts/download_models.py
```

### Model Source Options

```bash
# Official Hugging Face checkpoints (requires access for Meta Llama models)
python scripts/download_models.py --provider hf --models llama-3.2-1b llama-3.2-3b phi-3-mini

# Ollama mirror pull path (easy team onboarding, no HF gate required)
python scripts/download_models.py --provider ollama --models llama-3.2-1b llama-3.2-3b phi-3-mini
```

## 📁 Repository Structure

```
├── models/                 # Model configurations
│   ├── llama-3.2-1b/      # LLaMA 3.2 1B
│   ├── llama-3.2-3b/      # LLaMA 3.2 3B  
│   ├── phi-3-mini/        # Phi-3 mini
│   └── others/            # Additional models
├── benchmarks/            # Evaluation datasets and scripts
├── notebooks/             # Analysis and visualization
│   └── colab_setup.ipynb  # Google Colab setup
├── scripts/               # Automation scripts
└── results/               # Experimental results
```

## 📈 Expected Results

We anticipate finding clear trade-offs in the accuracy-latency space:
- **1B models**: Fastest inference, moderate accuracy
- **3B models**: Balanced performance-efficiency 
- **Quantization**: Significant speedup with manageable accuracy loss

## 🔧 Technical Details

### Hardware Compatibility
- **GPU**: CUDA-compatible (recommended)
- **CPU**: Fallback for smaller models
- **Memory**: 8GB+ RAM for 3B models
- **Colab**: Free T4/V100 GPU access

### Quantization Support
- **FP16**: Baseline half-precision
- **INT4**: 4-bit quantization for strong efficiency gains
- **INT2**: 2-bit quantization for maximum compression

## 🏃 Experiment Pipeline

Use these scripts for a reproducible baseline workflow.

### 1. Run Accuracy Benchmarks

```bash
python scripts/run_benchmarks.py \
  --model-key llama-3.2-1b \
  --benchmarks hellaswag arc gsm8k \
  --limit 100 \
  --precision fp16 \
  --device auto
```

No-HF-gate (local Ollama) benchmark path:

```bash
python scripts/run_benchmarks_ollama.py \
  --model-key llama-3.2-1b \
  --benchmarks hellaswag arc gsm8k \
  --limit 20
```

Notes:
- Raw output is saved to `results/raw/benchmark_<run_id>.json`
- Start with low `--limit` (for example, `20`) to validate the pipeline before full runs
- `int4` is supported in this baseline via bitsandbytes on CUDA
- `int2` is not supported in this transformers baseline (use GGUF/llama.cpp path for 2-bit runs)

### 2. Run Efficiency Measurements

```bash
python scripts/run_efficiency.py \
  --model-key llama-3.2-1b \
  --precision fp16 \
  --device auto \
  --num-prompts 5 \
  --warmup-runs 2 \
  --timed-runs 3 \
  --max-new-tokens 64
```

Notes:
- Raw output is saved to `results/raw/efficiency_<run_id>.json`
- Metrics include TTFT, latency/token, throughput, and memory

### 3. Aggregate Raw Results to CSV

```bash
python scripts/aggregate_results.py
```

This writes normalized metrics to `results/processed/metrics.csv` for plotting and Pareto analysis.

### 4. Generate PDF Snapshot Report

```bash
python scripts/generate_report.py
```

This writes `results/reports/project_snapshot_report.pdf` using the latest matching benchmark and efficiency runs for:
- LLaMA-3.2-1B
- LLaMA-3.2-3B
- Phi-3-mini

## 📖 References

- [LLaMA 3.2: Lightweight Text and Multimodal Models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [Efficient Model Deployment Best Practices](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)

## 📋 Progress

- [x] Repository setup and documentation
- [x] Model download automation
- [x] Colab integration
- [x] Baseline benchmark implementation (HellaSwag, ARC, GSM8K)
- [x] Baseline efficiency pipeline (TTFT, latency/token, throughput, memory)
- [ ] Results analysis and visualization
- [ ] Pareto frontier analysis

---

**Course**: CAP6614 - Efficient Machine Learning  
**Focus**: On-device deployment and edge computing optimization
