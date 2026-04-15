# Benchmark Catalog

This document lists the benchmarks used in this repository, what they measure, and where their data comes from.

## 1) Benchmarks used by the current report pipeline

These are the benchmarks executed by:
- `scripts/run_benchmarks.py`
- `scripts/run_benchmarks_ollama.py`

| CLI key | Benchmark | What it measures | Metric in repo | Dataset/source link |
|---|---|---|---|---|
| `hellaswag` | HellaSwag | Commonsense completion (multiple choice) | Accuracy | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| `arc` | ARC-Challenge | Grade-school science QA (multiple choice) | Accuracy | [ai2_arc (ARC-Challenge)](https://huggingface.co/datasets/ai2_arc) |
| `gsm8k` | GSM8K | Grade-school math word problems | Exact match (numeric final answer) | [gsm8k](https://huggingface.co/datasets/gsm8k) |

## 2) Full benchmark framework in `benchmarks/`

These are registered in:
- `benchmarks/tasks/__init__.py` (text)
- `benchmarks/tasks/vision/__init__.py` (vision)

### Text benchmarks (15)

| Task key | Benchmark | Metric in repo | Dataset/source link |
|---|---|---|---|
| `mmlu` | MMLU | Accuracy | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| `open_rewrite` | Open-rewrite eval | ROUGE-L | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| `tldr9` | TLDR9+ style summarization | ROUGE-L | [webis/tldr-17](https://huggingface.co/datasets/webis/tldr-17) |
| `ifeval` | IFEval (instruction following) | Accuracy | [google/IFEval](https://huggingface.co/datasets/google/IFEval) |
| `bfcl_v2` | Berkeley Function Calling Leaderboard (BFCL) | Accuracy | [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) |
| `nexus` | Nexus (tool-use style eval) | Accuracy | [Nexusflow/NexusRaven_API_evaluation](https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation) |
| `gsm8k` | GSM8K | Exact match | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| `math` | MATH | Exact match | [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) |
| `arc_challenge` | ARC-Challenge | Accuracy | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| `gpqa` | GPQA | Accuracy | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) |
| `hellaswag` | HellaSwag | Accuracy | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| `infinitebench_mc` | InfiniteBench (MC variant) | Accuracy | [xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) |
| `infinitebench_qa` | InfiniteBench (QA variant) | Accuracy | [xinrongzhang2022/InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) |
| `nih_multi_needle` | NIH / Multi-needle long-context | Accuracy | Synthetic-style benchmark (no HF dataset path set in code) |
| `mgsm` | MGSM (multilingual GSM8K) | Exact match | [juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm) |

### Vision benchmarks (8)

| Task key | Benchmark | Metric in repo | Dataset/source link |
|---|---|---|---|
| `mmmu` | MMMU | Accuracy | [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU) |
| `mmmu_pro_standard` | MMMU-Pro (standard) | Accuracy | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| `mmmu_pro_vision` | MMMU-Pro (vision) | Accuracy | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| `mathvista` | MathVista | Accuracy | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) |
| `chartqa` | ChartQA | Relaxed accuracy | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| `ai2_diagram` | AI2 Diagram-style visual reasoning | Accuracy | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| `docvqa` | DocVQA | ANLS | [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) |
| `vqav2` | VQAv2 | Accuracy | [HuggingFaceM4/VQAv2](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) |

## 3) Notes on benchmark fidelity

- Some task modules include `TODO` comments for dataset/task alignment (`open_rewrite`, `tldr9`, `nexus`, `ai2_diagram`, `nih_multi_needle`).
- For strict paper replication, verify each task prompt format, split, and scoring against the original benchmark protocol before final reporting.
- The PDF snapshot report currently uses the three-benchmark baseline set (`hellaswag`, `arc`, `gsm8k`).
