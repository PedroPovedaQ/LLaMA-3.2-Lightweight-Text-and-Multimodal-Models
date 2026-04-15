#!/usr/bin/env python3
"""Measure inference efficiency metrics for small language models."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency at runtime
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

from experiment_utils import (
    MODEL_KEYS,
    RAW_RESULTS_DIR,
    capture_hardware_info,
    ensure_results_dirs,
    format_user_prompt,
    load_model_and_tokenizer,
    make_run_id,
    model_input_device,
    resolve_model_id,
    save_json,
)

DEFAULT_PROMPTS = [
    "Summarize the causes of the French Revolution in four bullet points.",
    "Write pseudocode for binary search and explain time complexity in one sentence.",
    "A store sells 3 notebooks for $7.50. How much for 14 notebooks?",
    "Give a concise explanation of what overfitting means in machine learning.",
    "Draft a short email asking a professor for office-hours clarification.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline efficiency measurements")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model id")
    parser.add_argument(
        "--model-key",
        type=str,
        choices=sorted(MODEL_KEYS.keys()),
        default="llama-3.2-1b",
        help="Shorthand for a predefined model id",
    )
    parser.add_argument("--precision", type=str, default="fp16", help="fp16 | bf16 | int4 | int2")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu | mps")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional text file with one prompt per non-empty line",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to evaluate (taken from prompt-file or defaults)",
    )
    parser.add_argument("--warmup-runs", type=int, default=2, help="Warmup runs per prompt")
    parser.add_argument("--timed-runs", type=int, default=3, help="Timed runs per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generated tokens per timed run")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for raw efficiency JSON",
    )
    return parser.parse_args()


def load_prompts(prompt_file: Optional[str], num_prompts: int) -> List[str]:
    if num_prompts <= 0:
        return []

    prompts: List[str] = []
    if prompt_file:
        with Path(prompt_file).open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()

    if len(prompts) >= num_prompts:
        return prompts[:num_prompts]

    expanded = prompts.copy()
    idx = 0
    while len(expanded) < num_prompts:
        expanded.append(prompts[idx % len(prompts)])
        idx += 1
    return expanded


def _synchronize_if_needed(runtime_device: str) -> None:
    if torch is not None and runtime_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch is not None and runtime_device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_generate(model, tokenizer, runtime, prompt: str, max_new_tokens: int) -> Dict[str, float]:
    rendered_prompt = format_user_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 2048), 4096),
    )
    target_device = model_input_device(model, runtime.device)
    encoded = {k: v.to(target_device) for k, v in encoded.items()}

    _synchronize_if_needed(runtime.device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    _synchronize_if_needed(runtime.device)
    elapsed = time.perf_counter() - start

    generated_tokens = int(output.shape[1] - encoded["input_ids"].shape[1])
    return {
        "elapsed_sec": elapsed,
        "generated_tokens": float(generated_tokens),
    }


def mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main() -> None:
    args = parse_args()

    if psutil is None:
        raise ImportError("psutil is required. Install dependencies: pip install -r requirements.txt")
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    ensure_results_dirs()

    model_id = resolve_model_id(args)
    prompts = load_prompts(args.prompt_file, args.num_prompts)
    if not prompts:
        raise ValueError("No prompts available for efficiency run")

    process = psutil.Process()
    ram_before_load_gb = process.memory_info().rss / (1024**3)

    model, tokenizer, runtime = load_model_and_tokenizer(
        model_id=model_id,
        precision=args.precision,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    ram_after_load_gb = process.memory_info().rss / (1024**3)

    if runtime.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_id = make_run_id("efficiency")
    started_at = datetime.now(timezone.utc)

    per_prompt = []
    ttft_ms_all = []
    latency_ms_per_token_all = []
    throughput_tok_s_all = []

    for idx, prompt in enumerate(prompts):
        print(f"[efficiency] Prompt {idx + 1}/{len(prompts)}")

        for _ in range(max(args.warmup_runs, 0)):
            _ = timed_generate(
                model,
                tokenizer,
                runtime,
                prompt,
                max_new_tokens=max(8, min(args.max_new_tokens, 16)),
            )

        ttft_ms = []
        latency_ms_per_token = []
        throughput_tok_s = []

        for _ in range(max(args.timed_runs, 1)):
            ttft = timed_generate(model, tokenizer, runtime, prompt, max_new_tokens=1)
            ttft_ms_value = ttft["elapsed_sec"] * 1000.0
            ttft_ms.append(ttft_ms_value)
            ttft_ms_all.append(ttft_ms_value)

            run = timed_generate(
                model,
                tokenizer,
                runtime,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            generated = max(run["generated_tokens"], 1.0)
            latency_value = (run["elapsed_sec"] * 1000.0) / generated
            throughput_value = generated / run["elapsed_sec"]
            latency_ms_per_token.append(latency_value)
            throughput_tok_s.append(throughput_value)
            latency_ms_per_token_all.append(latency_value)
            throughput_tok_s_all.append(throughput_value)

        per_prompt.append(
            {
                "prompt_index": idx,
                "prompt": prompt,
                "ttft_ms": mean_std(ttft_ms),
                "latency_ms_per_token": mean_std(latency_ms_per_token),
                "throughput_tok_s": mean_std(throughput_tok_s),
            }
        )

    ended_at = datetime.now(timezone.utc)
    ram_after_run_gb = process.memory_info().rss / (1024**3)

    gpu_peak_gb = None
    if runtime.device == "cuda" and torch.cuda.is_available():
        gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    summary = {
        "run_type": "efficiency",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "runtime": asdict(runtime),
        "config": {
            "num_prompts": len(prompts),
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "max_new_tokens": args.max_new_tokens,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
            },
        },
        "hardware": capture_hardware_info(),
        "memory": {
            "ram_before_load_gb": round(ram_before_load_gb, 4),
            "ram_after_load_gb": round(ram_after_load_gb, 4),
            "ram_after_run_gb": round(ram_after_run_gb, 4),
            "gpu_peak_gb": round(gpu_peak_gb, 4) if gpu_peak_gb is not None else None,
        },
        "metrics": {
            "ttft_ms": mean_std(ttft_ms_all),
            "latency_ms_per_token": mean_std(latency_ms_per_token_all),
            "throughput_tok_s": mean_std(throughput_tok_s_all),
        },
        "per_prompt": per_prompt,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RAW_RESULTS_DIR / f"{run_id}.json"

    save_json(summary, output_path)
    print(f"[efficiency] Saved results to: {output_path}")

    print(
        "[efficiency] "
        f"ttft_ms={summary['metrics']['ttft_ms']['mean']:.2f} | "
        f"latency_ms_per_token={summary['metrics']['latency_ms_per_token']['mean']:.2f} | "
        f"throughput_tok_s={summary['metrics']['throughput_tok_s']['mean']:.2f}"
    )


if __name__ == "__main__":
    main()
