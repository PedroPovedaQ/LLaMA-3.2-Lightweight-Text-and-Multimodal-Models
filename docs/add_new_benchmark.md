# Add New Benchmark Guide

Use this guide when you want to add a new benchmark to this repository's Hugging Face-backed evaluation pipeline.

This repo has two benchmark entry points:

- `scripts/run_benchmarks.py`: the baseline runner used for the project's raw benchmark JSON, per-run PNG/PDF artifacts, and the snapshot report inputs
- `python -m benchmarks.runner`: the broader task framework under `benchmarks/tasks/`

If your goal is "make a new benchmark available from `scripts/run_benchmarks.py`", follow Sections 1-6 below. If you also want it available in the unified framework, follow Section 7 as well.

## 1) Decide the benchmark key and source

Before writing code, define:

- CLI key, for example `gpqa`
- Dataset source, ideally a Hugging Face dataset
- Split to evaluate, for example `validation` or `test`
- Metric, for example `accuracy` or `exact_match`
- Output format expected from the model

Notes:

- Prefer Hugging Face datasets so runs stay consistent with the rest of the repo
- If the dataset is gated, document that clearly; the benchmark may require Hugging Face authentication at runtime
- Keep the CLI key short and stable because it becomes part of raw result JSON and report logic

## 2) Add the evaluator to `scripts/benchmark_eval.py`

Create a new evaluator function in [benchmark_eval.py](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/scripts/benchmark_eval.py).

The function signature should match the existing evaluators:

```python
def evaluate_my_benchmark(
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
    ...
```

The evaluator should:

1. Load the dataset with `datasets.load_dataset(...)`
2. Limit examples using `safe_take(dataset, limit)`
3. Build prompts in the same style as the rest of the baseline runner
4. Call `generate_fn(prompt, max_new_tokens)`
5. Parse the model output
6. Compute the metric
7. Return a result dictionary with the same fields used elsewhere

Minimum expected result fields:

```python
{
    "name": "my_benchmark",
    "num_examples": total,
    "correct": correct,
    "accuracy": round(accuracy, 6),
    "duration_sec": round(elapsed, 3),
    "examples_per_sec": ...,
}
```

If the benchmark is not accuracy-based, keep the output shape consistent but use the metric field names that match the rest of the script's conventions.

## 3) Register it in `BENCHMARK_EVALUATORS`

At the bottom of [benchmark_eval.py](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/scripts/benchmark_eval.py), add the new key to `BENCHMARK_EVALUATORS`.

Example:

```python
BENCHMARK_EVALUATORS: Dict[str, BenchmarkEvaluator] = {
    "hellaswag": evaluate_hellaswag,
    "arc": evaluate_arc,
    "my_benchmark": evaluate_my_benchmark,
}
```

Once registered, it automatically appears in:

- `python scripts/run_benchmarks.py --help`
- `python scripts/run_benchmarks.py --benchmarks ...`

## 4) Update docs and benchmark catalog

Update [benchmarks.md](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/docs/benchmarks.md):

- Add the benchmark to the appropriate table
- Document the dataset/source link
- Note any caveats such as gating, custom splits, or partial protocol fidelity

If the benchmark becomes part of the baseline set used in project reporting, also update:

- [README.md](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/README.md) example benchmark commands
- [generate_report.py](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/scripts/generate_report.py) baseline assumptions such as `BASELINE_BENCHMARKS`

Important:

- Do not add a benchmark to the snapshot report path unless the report code explicitly knows how to summarize it
- If the report expects a fixed benchmark set, update that set and any chart labels together

## 5) Run a smoke test

Start with a very small run:

```bash
python scripts/run_benchmarks.py \
  --model-id ./models/llama-3.2-1b \
  --benchmarks my_benchmark \
  --limit 5 \
  --precision fp16 \
  --device auto
```

Expected outputs:

- Raw JSON in `results/raw/`
- Accuracy/runtime PNG charts in `results/reports/`
- A per-run PDF summary in `results/reports/`

Use `--no-report` only if you explicitly want to skip the visual artifacts.

## 6) Verify behavior and edge cases

Check:

- The CLI accepts the new benchmark key
- The dataset loads in your target environment
- Output parsing is robust to extra model text
- The metric values look sane on a small sample
- The raw JSON contains the expected benchmark name and fields

Common issues:

- Gated datasets on Hugging Face require authentication
- Offline environments only work if the dataset is already cached
- CPU-only runs can be slow; use low `--limit` for validation
- Some model/tokenizer combinations may try to touch the network unless you use local checkpoint paths

## 7) Optional: add it to the unified `benchmarks/` framework

If you also want the benchmark available via `python -m benchmarks.runner`, add a task module under `benchmarks/tasks/` and register it in [benchmarks/tasks/__init__.py](/Users/pedro.poveda/Documents/learning/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/benchmarks/tasks/__init__.py).

That path is separate from `scripts/benchmark_eval.py`.

In practice:

- `scripts/benchmark_eval.py` controls `scripts/run_benchmarks.py`
- `benchmarks/tasks/__init__.py` controls `python -m benchmarks.runner`

If you want both entry points to expose the new benchmark, you must update both registries.

## 8) Suggested checklist

- [ ] Added evaluator to `scripts/benchmark_eval.py`
- [ ] Registered benchmark in `BENCHMARK_EVALUATORS`
- [ ] Updated `docs/benchmarks.md`
- [ ] Updated report assumptions if the benchmark is part of the baseline report set
- [ ] Smoke-tested with `--limit 5`
- [ ] Confirmed raw JSON and per-run PNG/PDF artifacts were generated
- [ ] Added unified-framework task registration if needed
