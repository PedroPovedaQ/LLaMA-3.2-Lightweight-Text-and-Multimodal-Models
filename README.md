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

See [docs/benchmarks.md](docs/benchmarks.md) for the full benchmark catalog (task keys, metrics, and source links).

### Text Benchmarks

| Category | Benchmark | Setup | Metric |
|----------|-----------|-------|--------|
| General | **MMLU** | 5-shot | Accuracy |
| General | **Open-rewrite eval** | 0-shot, rougeL | rougeL |
| General | **TLDR9+** | 1-shot, rougeL | rougeL |
| General | **IFEval** | — | Accuracy |
| Tool Use | **BFCL V2** | — | Accuracy |
| Tool Use | **Nexus** | — | Accuracy |
| Math | **GSM8K** | 8-shot, CoT | Exact match |
| Math | **MATH** | 0-shot, CoT | Exact match |
| Reasoning | **ARC-Challenge** | 0-shot | Accuracy |
| Reasoning | **GPQA** | 0-shot | Accuracy |
| Reasoning | **HellaSwag** | 0-shot | Accuracy |
| Long Context | **InfiniteBench/En.MC** | 128k | Accuracy |
| Long Context | **InfiniteBench/En.QA** | 128k | Accuracy |
| Long Context | **NIH/Multi-needle** | — | Accuracy |
| Multilingual | **MGSM** | 0-shot, CoT | Exact match |

### Vision Benchmarks (Instruction Tuned)

| Category | Benchmark | Setup | Metric |
|----------|-----------|-------|--------|
| College-level | **MMMU** | 0-shot CoT, micro avg accuracy | Accuracy |
| College-level | **MMMU-Pro, Standard** | 10-opts, text | Accuracy |
| College-level | **MMMU-Pro, Vision** | text | Accuracy |
| Math | **MathVista** | testmini | Accuracy |
| Charts & Diagrams | **ChartQA** | 0-shot CoT, relaxed accuracy | Accuracy |
| Charts & Diagrams | **AI2 Diagram** | test | Accuracy |
| Document | **DocVQA** | ANLS | ANLS |
| Visual QA | **VQAv2** | test | Accuracy |

### Efficiency Metrics
- **Latency** - Time per token (ms/token)
- **Throughput** - Tokens per second
- **Memory Usage** - Peak RAM consumption
- **Quantization Impact** - FP16 vs INT4 vs INT2 performance

## 🚀 Getting Started

### Option A: Google Colab (Recommended)

> **Note:** This repo is private. The Colab badge won't work directly. To open the notebook:
> 1. Go to [colab.research.google.com](https://colab.research.google.com)
> 2. Click **File → Open notebook → GitHub**
> 3. Sign in with your GitHub account and select this repo
> 4. Pick `notebooks/colab_setup.ipynb`

1. Open the notebook using the steps above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run the setup cells to clone the repo and install dependencies:
   ```python
   !git clone https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models.git
   %cd LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
   !pip install -r requirements-colab.txt
   ```
4. Download models (default is `llama-cli` provider for LLaMA models):
   ```python
   !huggingface-cli login --token YOUR_TOKEN
   !python scripts/download_models.py

   # Optional: pull Phi-3-mini too
   !python scripts/download_models.py --provider hf --models phi-3-mini
   ```
5. Run a smoke test to verify everything works:
   ```python
   !python -m benchmarks.runner --model tinyllama --quant fp16 --max-samples 10
   ```
6. Run benchmarks:
   ```python
   # Single model + benchmark
   !python -m benchmarks.runner --model llama-3.2-1b --quant int4 --benchmark mmlu

   # All text benchmarks for one model
   !python -m benchmarks.runner --model llama-3.2-1b --quant fp16

   # Vision benchmarks (requires vision model)
   !python -m benchmarks.runner --model llama-3.2-11b-vision --quant fp16 --benchmark mmmu --vision

   # Full matrix
   !python -m benchmarks.runner --all
   ```

### Option B: Local Setup
```bash
git clone https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models.git
cd LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
python scripts/setup.py
pip install -r requirements.txt
huggingface-cli login
python scripts/download_models.py

# Optional: include Phi-3-mini for 3-way comparison
python scripts/download_models.py --provider hf --models phi-3-mini

# Smoke test
python -m benchmarks.runner --model tinyllama --quant fp16 --max-samples 10
```
> **Note:** Local setup requires a CUDA GPU for quantized inference (INT8/INT4). FP16 can run on CPU but will be slow.

### Model Source Options

```bash
# Default path (llama-cli provider; downloads LLaMA 3.2 1B + 3B)
python scripts/download_models.py

# Add Phi-3-mini (HF provider)
python scripts/download_models.py --provider hf --models phi-3-mini

# Official Hugging Face checkpoints (all core models)
python scripts/download_models.py --provider hf --models llama-3.2-1b llama-3.2-3b phi-3-mini

# Meta Llama CLI path (downloads with llama-model)
python scripts/download_models.py --provider llama-cli --models llama-3.2-1b llama-3.2-3b --llama-source huggingface

# Meta signed URL path (no Hugging Face dependency for model download)
python scripts/download_models.py --provider llama-cli --models llama-3.2-1b --llama-source meta --meta-url 'https://...llamameta.net/*?...'
```

Llama CLI usage skill:
- `skills/llama-cli/SKILL.md`

Notes:
- `llama-cli` downloads may produce original `.pth` checkpoints; the current Transformers benchmark scripts require HF-formatted weights (`model.safetensors` / `pytorch_model.bin`).
- Existing benchmark reruns in this repo were executed with HF-backed model loading (`scripts/run_benchmarks.py`).

## 🔌 Extending Entry Points

To add a new model key:
1. Add one entry to `MODEL_SPECS` in `scripts/experiment_utils.py`.
2. The new key will automatically appear in:
   - `scripts/run_benchmarks.py --model-key ...`
   - `scripts/run_efficiency.py --model-key ...`
   - `scripts/download_models.py --models ...` (when using model keys)

To add a new benchmark:
1. Add an evaluator function to `scripts/benchmark_eval.py`.
2. Register it in `BENCHMARK_EVALUATORS` in the same file.
3. It will automatically appear in:
   - `scripts/run_benchmarks.py --benchmarks ...`
4. For the full workflow and reporting implications, see [docs/add_new_benchmark.md](docs/add_new_benchmark.md).

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
  --benchmarks hellaswag arc gpqa gsm8k \
  --limit 100 \
  --precision fp16 \
  --device auto
```

Notes:
- Raw output is saved to `results/raw/benchmark_<run_id>.json`
- Visual artifacts are also saved by default to `results/reports/` as PNG charts plus a per-run PDF summary
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

This writes normalized metrics to `results/processed/metrics.csv` for plotting and Pareto analysis, and now also regenerates the PDF report by default.
Use `--skip-report` to disable report generation.

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

## 📅 Milestones (Target: April 13)

| Milestone | Deadline | Deliverables |
|-----------|----------|--------------|
| **M1** — Environment & Setup | Mar 28 | `pip install`, download models, confirm GPU access (Colab or local) |
| **M2** — Benchmark Runner + Quantization | Apr 2 | Finalize eval scripts (15 text + 8 vision), quantization wrapper (FP16/INT8/INT4), smoke test on TinyLlama |
| **M3** — Full Experiment Matrix | Apr 6 | Run all model × quant × benchmark combos, collect latency/memory, commit raw results |
| **M4** — Analysis & Visualization | Apr 9 | Pareto frontier plot, comparison tables, charts, statistical significance tests |
| **M5** — Presentation & Report | Apr 13 | Slides finalized, NeurIPS report draft, Colab demo ready |

## 📋 Progress

- [x] Repository setup and documentation
- [x] Model download automation
- [x] Colab integration
- [x] Baseline benchmark implementation (HellaSwag, ARC, GPQA, GSM8K)
- [x] Baseline efficiency pipeline (TTFT, latency/token, throughput, memory)
- [x] Benchmark framework scaffold (15 text + 8 vision)
- [x] Quantization wrapper (FP16 / INT8 / INT4 via bitsandbytes)
- [ ] Finalize benchmark implementations
- [ ] Latency & memory profiling
- [ ] Full experiment matrix
- [ ] Results analysis and visualization
- [ ] Pareto frontier analysis
- [ ] Final presentation and NeurIPS report

---

**Course**: CAP6614 - Efficient Machine Learning
**Focus**: On-device deployment and edge computing optimization
