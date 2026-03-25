#!/usr/bin/env python3
"""Unified benchmark runner.

Usage:
    # Single model + benchmark
    python -m benchmarks.runner --model llama-3.2-1b --quant int4 --benchmark mmlu

    # Single model, all text benchmarks
    python -m benchmarks.runner --model llama-3.2-1b --quant fp16

    # Vision benchmarks (requires vision model)
    python -m benchmarks.runner --model llama-3.2-11b-vision --quant fp16 --benchmark mmmu --vision

    # Full text matrix (all models x all quant levels x all text benchmarks)
    python -m benchmarks.runner --all

    # Quick smoke test (10 samples per benchmark)
    python -m benchmarks.runner --model tinyllama --quant fp16 --max-samples 10

See: https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/issues/3
"""

import argparse
import itertools
import sys

import torch

from benchmarks.tasks import TEXT_TASKS, TASKS
from benchmarks.tasks.vision import VISION_TASKS
from benchmarks.utils.quantization import MODEL_REGISTRY, QUANT_LEVELS, load_model, get_memory_footprint_mb
from benchmarks.utils.results import build_result_dict, save_result


def load_vision_model(model_name, quant="fp16", cache_dir="./models"):
    """Load a vision model and processor.

    Vision models use a processor (handles images + text) instead of a tokenizer.
    """
    from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig

    hf_name = MODEL_REGISTRY.get(model_name, model_name)
    kwargs = {}
    quantization_config = None

    if quant == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quant == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    if quantization_config:
        kwargs["quantization_config"] = quantization_config

    model = MllamaForConditionalGeneration.from_pretrained(
        hf_name, device_map="auto", cache_dir=cache_dir, **kwargs
    )
    processor = AutoProcessor.from_pretrained(hf_name, cache_dir=cache_dir)
    return model, processor


def run_single(model_name, quant, benchmark_name, max_samples=None, vision=False):
    """Run a single (model, quant, benchmark) combination."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} | Quant: {quant} | Benchmark: {benchmark_name}")
    print(f"{'='*60}")

    task_cls = TASKS[benchmark_name]
    task = task_cls(max_samples=max_samples)
    hf_name = MODEL_REGISTRY.get(model_name, model_name)

    if vision or benchmark_name in VISION_TASKS:
        print(f"Loading vision model {model_name} ({quant})...")
        model, processor = load_vision_model(model_name, quant=quant)
        mem_mb = get_memory_footprint_mb(model)
        print(f"Model loaded — {mem_mb:.0f} MB")

        print(f"Running {benchmark_name} ({task.n_shot}-shot, {task.metric_name})...")
        score, metrics = task.run(model, processor)
    else:
        print(f"Loading {model_name} ({quant})...")
        model, tokenizer = load_model(model_name, quant=quant)
        mem_mb = get_memory_footprint_mb(model)
        print(f"Model loaded — {mem_mb:.0f} MB")

        print(f"Running {benchmark_name} ({task.n_shot}-shot, {task.metric_name})...")
        score, metrics = task.run(model, tokenizer)

    print(f"Score: {score:.4f} ({task.metric_name})")
    print(f"Latency: {metrics.latency_ms_per_token:.1f} ms/token")
    print(f"Throughput: {metrics.throughput_tokens_per_sec:.1f} tokens/sec")
    print(f"Peak memory: {metrics.peak_memory_mb:.0f} MB")

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
            "vision": vision or benchmark_name in VISION_TASKS,
        },
    )
    save_result(result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_matrix(max_samples=None, vision=False):
    """Run all models x quant levels x benchmarks."""
    models = list(MODEL_REGISTRY.keys())
    benchmarks = list(VISION_TASKS.keys()) if vision else list(TEXT_TASKS.keys())
    mode = "vision" if vision else "text"

    total = len(models) * len(QUANT_LEVELS) * len(benchmarks)
    print(f"Running full {mode} matrix: {len(models)} models x {len(QUANT_LEVELS)} quant x {len(benchmarks)} benchmarks = {total} experiments")

    results = []
    for i, (m, q, b) in enumerate(itertools.product(models, QUANT_LEVELS, benchmarks), 1):
        print(f"\n[{i}/{total}]")
        try:
            result = run_single(m, q, b, max_samples=max_samples, vision=vision)
            results.append(result)
        except Exception as e:
            print(f"FAILED: {m} {q} {b} — {e}")
            results.append({"model_short": m, "quant": q, "benchmark": b, "error": str(e)})

    return results


def main():
    all_tasks = list(TASKS.keys())
    parser = argparse.ArgumentParser(description="Unified benchmark runner")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model to evaluate")
    parser.add_argument("--quant", choices=QUANT_LEVELS, default="fp16", help="Quantization level")
    parser.add_argument("--benchmark", choices=all_tasks, help="Benchmark to run (omit for all)")
    parser.add_argument("--all", action="store_true", help="Run full matrix (all models x quant x benchmarks)")
    parser.add_argument("--vision", action="store_true", help="Use vision model loading and vision benchmarks")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark (for testing)")

    args = parser.parse_args()

    if args.all:
        run_matrix(max_samples=args.max_samples, vision=args.vision)
    elif args.model:
        if args.benchmark:
            benchmarks = [args.benchmark]
        elif args.vision:
            benchmarks = list(VISION_TASKS.keys())
        else:
            benchmarks = list(TEXT_TASKS.keys())
        for b in benchmarks:
            run_single(args.model, args.quant, b, max_samples=args.max_samples, vision=args.vision)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
