#!/usr/bin/env python3
"""Run baseline accuracy benchmarks for small language models."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

from benchmark_eval import benchmark_names, run_benchmark
from experiment_utils import (
    MODEL_KEYS,
    RAW_RESULTS_DIR,
    REPORTS_DIR,
    capture_hardware_info,
    ensure_results_dirs,
    format_user_prompt,
    load_model_and_tokenizer,
    make_run_id,
    model_input_device,
    resolve_model_id,
    save_json,
)
from generate_benchmark_report import generate_benchmark_report

def parse_args() -> argparse.Namespace:
    available_benchmarks = benchmark_names()
    parser = argparse.ArgumentParser(description="Run baseline benchmark evaluations")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model id")
    parser.add_argument(
        "--model-key",
        type=str,
        choices=sorted(MODEL_KEYS.keys()),
        default="llama-3.2-1b",
        help="Shorthand for a predefined model id",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        # Pulled from benchmark_eval.BENCHMARK_EVALUATORS registry.
        choices=available_benchmarks,
        default=available_benchmarks,
        help="Benchmarks to run",
    )
    parser.add_argument("--limit", type=int, default=100, help="Examples per benchmark")
    parser.add_argument("--precision", type=str, default="fp16", help="fp16 | bf16 | int4 | int2")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu | mps")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include per-example predictions in output json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for raw benchmark JSON",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip PNG/PDF report generation for this benchmark run",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_completion(
    model,
    tokenizer,
    runtime,
    prompt: str,
    max_new_tokens: int,
) -> str:
    rendered_prompt = format_user_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 2048), 4096),
    )
    target_device = model_input_device(model, runtime.device)
    encoded = {k: v.to(target_device) for k, v in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    set_seed(args.seed)

    model_id = resolve_model_id(args)
    model, tokenizer, runtime = load_model_and_tokenizer(
        model_id=model_id,
        precision=args.precision,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    run_id = make_run_id("benchmark")
    started_at = datetime.now(timezone.utc)

    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        return generate_completion(model, tokenizer, runtime, prompt, max_new_tokens=max_new_tokens)

    benchmark_results = []
    for bench in args.benchmarks:
        print(f"[benchmark] Running {bench}...")
        benchmark_results.append(
            run_benchmark(bench, generate_fn, args.limit, args.save_predictions)
        )

    ended_at = datetime.now(timezone.utc)
    summary = {
        "run_type": "benchmark",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "runtime": asdict(runtime),
        "config": {
            "benchmarks": args.benchmarks,
            "limit": args.limit,
            "seed": args.seed,
            "save_predictions": args.save_predictions,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
            },
        },
        "hardware": capture_hardware_info(),
        "benchmarks": benchmark_results,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RAW_RESULTS_DIR / f"{run_id}.json"

    save_json(summary, output_path)
    print(f"[benchmark] Saved results to: {output_path}")

    if not args.no_report:
        output_prefix = REPORTS_DIR / run_id
        artifacts = generate_benchmark_report(summary, output_prefix)
        print(f"[benchmark] Saved accuracy chart to: {artifacts['accuracy_png']}")
        print(f"[benchmark] Saved runtime chart to: {artifacts['runtime_png']}")
        print(f"[benchmark] Saved PDF report to: {artifacts['report_pdf']}")

    for item in benchmark_results:
        print(
            "[benchmark] "
            f"{item['name']}: accuracy={item['accuracy']:.4f} "
            f"({item['correct']}/{item['num_examples']})"
        )


if __name__ == "__main__":
    main()
