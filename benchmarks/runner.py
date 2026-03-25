#!/usr/bin/env python3
"""Unified benchmark runner.

Usage:
    # Single model + benchmark
    python -m benchmarks.runner --model llama-3.2-1b --quant int4 --benchmark mmlu

    # Single model, all benchmarks
    python -m benchmarks.runner --model llama-3.2-1b --quant fp16

    # Full matrix (all models x all quant levels x all benchmarks)
    python -m benchmarks.runner --all

    # Quick smoke test (10 samples per benchmark)
    python -m benchmarks.runner --model tinyllama --quant fp16 --max-samples 10

See: https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/issues/3
"""

import argparse
import itertools
import sys

from benchmarks.tasks import TASKS
from benchmarks.utils.quantization import MODEL_REGISTRY, QUANT_LEVELS, load_model, get_memory_footprint_mb
from benchmarks.utils.results import build_result_dict, save_result


def run_single(model_name, quant, benchmark_name, max_samples=None):
    """Run a single (model, quant, benchmark) combination."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} | Quant: {quant} | Benchmark: {benchmark_name}")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name} ({quant})...")
    model, tokenizer = load_model(model_name, quant=quant)
    mem_mb = get_memory_footprint_mb(model)
    print(f"Model loaded — {mem_mb:.0f} MB")

    # Run benchmark
    task_cls = TASKS[benchmark_name]
    task = task_cls(max_samples=max_samples)
    hf_name = MODEL_REGISTRY.get(model_name, model_name)

    print(f"Running {benchmark_name} ({task.n_shot}-shot, {task.metric_name})...")
    score, metrics = task.run(model, tokenizer)

    print(f"Score: {score:.4f} ({task.metric_name})")
    print(f"Latency: {metrics.latency_ms_per_token:.1f} ms/token")
    print(f"Throughput: {metrics.throughput_tokens_per_sec:.1f} tokens/sec")
    print(f"Peak memory: {metrics.peak_memory_mb:.0f} MB")

    # Save result
    result = build_result_dict(
        model_name=hf_name,
        model_short=model_name,
        quant=quant,
        benchmark=benchmark_name,
        score=round(score, 4),
        metric=task.metric_name,
        metrics=metrics,
        config={
            "n_shot": task.n_shot,
            "max_samples": max_samples or "all",
        },
    )
    save_result(result)

    # Free GPU memory before next run
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_matrix(max_samples=None):
    """Run all models x quant levels x benchmarks."""
    models = list(MODEL_REGISTRY.keys())
    benchmarks = list(TASKS.keys())

    total = len(models) * len(QUANT_LEVELS) * len(benchmarks)
    print(f"Running full matrix: {len(models)} models x {len(QUANT_LEVELS)} quant x {len(benchmarks)} benchmarks = {total} experiments")

    results = []
    for i, (m, q, b) in enumerate(itertools.product(models, QUANT_LEVELS, benchmarks), 1):
        print(f"\n[{i}/{total}]")
        try:
            result = run_single(m, q, b, max_samples=max_samples)
            results.append(result)
        except Exception as e:
            print(f"FAILED: {m} {q} {b} — {e}")
            results.append({"model_short": m, "quant": q, "benchmark": b, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark runner")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model to evaluate")
    parser.add_argument("--quant", choices=QUANT_LEVELS, default="fp16", help="Quantization level")
    parser.add_argument("--benchmark", choices=list(TASKS.keys()), help="Benchmark to run (omit for all)")
    parser.add_argument("--all", action="store_true", help="Run full matrix (all models x quant x benchmarks)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark (for testing)")

    args = parser.parse_args()

    if args.all:
        run_matrix(max_samples=args.max_samples)
    elif args.model:
        benchmarks = [args.benchmark] if args.benchmark else list(TASKS.keys())
        for b in benchmarks:
            run_single(args.model, args.quant, b, max_samples=args.max_samples)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
