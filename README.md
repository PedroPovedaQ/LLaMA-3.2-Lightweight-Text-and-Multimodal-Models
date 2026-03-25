# LLaMA 3.2 vs Small Model Comparison


**Evaluating the accuracy-latency Pareto frontier for lightweight language models on consumer hardware**
### Team 9: Pedro Poveda |	Joel Gonzalez |	Jose Gabriel Gonzalez Nunez |	Matthew Horvath

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
- **Quantization Impact** - FP16 vs INT8 vs INT4 performance

## 🚀 Getting Started

### Option A: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/blob/main/notebooks/colab_setup.ipynb)

1. Click the badge above to open the Colab notebook
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run the setup cells to clone the repo and install dependencies:
   ```python
   !git clone https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models.git
   %cd LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
   !pip install -r requirements-colab.txt
   ```
4. Download models (requires [Hugging Face token](https://huggingface.co/settings/tokens) for gated models like LLaMA):
   ```python
   !huggingface-cli login --token YOUR_TOKEN
   !python scripts/download_models.py
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

# Smoke test
python -m benchmarks.runner --model tinyllama --quant fp16 --max-samples 10
```
> **Note:** Local setup requires a CUDA GPU for quantized inference (INT8/INT4). FP16 can run on CPU but will be slow.

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
- **INT8**: 8-bit quantization via bitsandbytes
- **INT4**: 4-bit quantization for extreme efficiency

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
