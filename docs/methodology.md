# Methodology: Small Language Model Comparison

## Overview

This document outlines the experimental methodology for comparing small language models, with focus on LLaMA 3.2 variants against Phi-3-mini and other compact models suitable for edge/mobile deployment.

## Research Questions

### Primary Questions
1. **Performance vs. Size**: How do 1B vs 3B parameter models compare in task accuracy?
2. **Efficiency Analysis**: Which models offer the best performance-per-parameter and performance-per-watt ratios?
3. **Mobile Readiness**: Which models are most suitable for resource-constrained deployment?

### Secondary Questions
- Do different models excel at different task types?
- How does quantization affect relative performance rankings?
- What are the memory and latency trade-offs across models?

## Model Selection

### Core Models (Required)
- **LLaMA 3.2-1B** - Ultra-compact baseline
- **LLaMA 3.2-3B** - Balanced mobile model
- **Phi-3-mini** - Microsoft's efficient alternative (~3.8B params)

### Additional Models (Optional)
- **Gemma 2B** - Google's compact model
- **TinyLlama 1.1B** - Community ultra-small model
- **Qwen2 1.5B** - Alibaba's efficient model

## Evaluation Framework

### Performance Benchmarks

#### 1. Language Understanding
- **MMLU** (Massive Multitask Language Understanding)
  - 5-shot evaluation
  - Focus on key subjects: STEM, humanities, social sciences
  - Metric: Accuracy (%)

- **HellaSwag** (Commonsense Reasoning)
  - 10-shot evaluation  
  - Metric: Accuracy (%)

#### 2. Mathematical Reasoning
- **GSM8K** (Grade School Math)
  - Chain-of-thought prompting
  - 5-shot evaluation
  - Metric: Exact match accuracy (%)

#### 3. Reading Comprehension
- **ARC-Challenge** (AI2 Reasoning Challenge)
  - 25-shot evaluation
  - Metric: Accuracy (%)

#### 4. Text Generation Quality
- **WikiText-2** Perplexity
  - Measure: Perplexity score (lower is better)

#### 5. Instruction Following
- **MT-Bench Style** evaluation
  - Custom instruction-following tasks
  - Human evaluation on 1-10 scale
  - Categories: Writing, reasoning, math, coding, extraction

### Efficiency Metrics

#### 1. Inference Performance
- **Latency**: Time per token generation (ms/token)
- **Throughput**: Tokens per second  
- **Time to First Token**: Cold start latency

#### 2. Resource Utilization
- **Memory Usage**: Peak RAM consumption (GB)
- **GPU Memory**: VRAM usage during inference
- **CPU Utilization**: Average CPU load during inference

#### 3. Energy Consumption (if measurable)
- **Power Draw**: Watts consumed during inference
- **Energy per Token**: Joules per generated token

## Experimental Setup

### Hardware Configuration
- **GPU**: [Specify your GPU - e.g., RTX 4090, A100, etc.]
- **CPU**: [Specify CPU model]
- **RAM**: [Amount of system RAM]
- **Storage**: SSD for model storage

### Software Environment
- **OS**: [Operating system]
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.30+
- **CUDA**: [Version if using GPU]

### Model Configurations

#### Standard Settings
- **Precision**: FP16 for consistency
- **Max Context**: 2048 tokens (or model maximum if lower)
- **Temperature**: 0.0 for deterministic evaluation
- **Top-p**: 1.0 (no nucleus sampling for benchmarks)

#### Quantization Tests (Optional)
- **INT8**: Using bitsandbytes
- **INT4**: Using bitsandbytes/GPTQ
- Compare: FP16 vs INT8 vs INT4 performance

### Evaluation Protocol

#### 1. Benchmark Execution
- **Batch Size**: Optimize per model for fair comparison
- **Repetitions**: 3 runs per benchmark per model
- **Random Seeds**: Fixed seeds for reproducibility
- **Warm-up**: 5 inference calls before timing

#### 2. Data Collection
- **Performance Metrics**: Accuracy, latency, throughput
- **Resource Metrics**: Memory usage, GPU utilization  
- **Error Analysis**: Failed cases for qualitative review

#### 3. Statistical Analysis
- **Significance Testing**: Paired t-tests for performance differences
- **Effect Sizes**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all metrics

## Expected Outcomes

### Performance Predictions
1. **LLaMA 3.2-3B** should outperform 1B variant on accuracy
2. **Phi-3-mini** may show competitive performance despite larger size
3. **Efficiency rankings** may differ from pure accuracy rankings

### Key Comparisons
- **1B models**: TinyLlama vs LLaMA 3.2-1B
- **~3B models**: LLaMA 3.2-3B vs Phi-3-mini vs Gemma 2B  
- **Size scaling**: 1B vs 3B performance gains within LLaMA family

## Limitations and Considerations

### Methodological Limitations
- **Hardware dependency**: Results specific to evaluation hardware
- **Benchmark bias**: Limited to selected evaluation tasks
- **Context length**: Restricted to shorter context evaluation

### Fairness Considerations  
- **Training data**: Models trained on different datasets
- **Optimization**: Some models may be more optimized for specific hardware
- **Instruction tuning**: Varying levels of instruction following training

## Deliverables

### Analysis Reports
1. **Performance Summary**: Benchmark results across all models
2. **Efficiency Analysis**: Resource utilization comparison
3. **Trade-off Analysis**: Performance vs efficiency curves
4. **Recommendations**: Best model for different use cases

### Code and Data
1. **Evaluation Scripts**: Reproducible benchmark code
2. **Results Database**: Raw and processed results
3. **Visualization Notebooks**: Charts and analysis plots
4. **Documentation**: Complete setup and reproduction guide

---
*Last updated: March 2026*