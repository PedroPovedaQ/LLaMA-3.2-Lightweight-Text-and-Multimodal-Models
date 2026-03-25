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

### Performance Benchmarks
- **MMLU** - Multitask language understanding
- **GSM8K** - Mathematical reasoning
- **HellaSwag** - Commonsense reasoning
- **ARC-Challenge** - Reading comprehension

### Efficiency Metrics
- **Latency** - Time per token (ms/token)
- **Throughput** - Tokens per second
- **Memory Usage** - Peak RAM consumption
- **Quantization Impact** - FP16 vs INT8 vs INT4 performance

## 🚀 Quick Start

### Option A: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/blob/main/notebooks/colab_setup.ipynb)

1. Open the Colab setup notebook
2. Run all cells to clone repo and install dependencies  
3. Start benchmarking with free GPU access

### Option B: Local Setup
```bash
git clone https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models.git
cd LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
python scripts/setup.py
pip install -r requirements.txt
python scripts/download_models.py
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
- **INT8**: 8-bit quantization via bitsandbytes
- **INT4**: 4-bit quantization for extreme efficiency

## 📖 References

- [LLaMA 3.2: Lightweight Text and Multimodal Models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [Efficient Model Deployment Best Practices](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)

## 📋 Progress

- [x] Repository setup and documentation
- [x] Model download automation
- [x] Colab integration
- [ ] Benchmark implementation
- [ ] Quantization pipeline
- [ ] Results analysis and visualization
- [ ] Pareto frontier analysis

---

**Course**: CAP6614 - Efficient Machine Learning  
**Focus**: On-device deployment and edge computing optimization# LLaMA-3.2-Lightweight-Text-and-Multimodal-Models
