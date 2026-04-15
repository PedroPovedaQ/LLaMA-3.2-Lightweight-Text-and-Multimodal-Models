#!/usr/bin/env python3
"""Run benchmark evaluations against locally available Ollama models."""

from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_eval import evaluate_arc, evaluate_gsm8k, evaluate_hellaswag
from experiment_utils import (
    MODEL_SPECS,
    RAW_RESULTS_DIR,
    capture_hardware_info,
    ensure_results_dirs,
    make_run_id,
    save_json,
)

OLLAMA_MODEL_KEYS = sorted(k for k, s in MODEL_SPECS.items() if s.ollama_tag is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ollama-backed benchmark evaluations")
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Ollama model tag (e.g. llama3.2:1b)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        choices=OLLAMA_MODEL_KEYS,
        default="llama-3.2-1b",
        help="Shorthand key for an Ollama model tag",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["hellaswag", "arc", "gsm8k"],
        default=["hellaswag", "arc", "gsm8k"],
        help="Benchmarks to run",
    )
    parser.add_argument("--limit", type=int, default=100, help="Examples per benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--host",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama host URL",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="HTTP timeout per model generation call",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include per-example predictions in output JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for raw benchmark JSON",
    )
    return parser.parse_args()


def resolve_model_tag(args: argparse.Namespace) -> str:
    if args.model_tag:
        return args.model_tag
    tag = MODEL_SPECS[args.model_key].ollama_tag
    if tag is None:
        raise ValueError(f"No Ollama tag configured for model key {args.model_key!r}")
    return tag


def set_seed(seed: int) -> None:
    random.seed(seed)


def _post_json(url: str, payload: Dict[str, object], timeout_sec: int) -> Dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def assert_ollama_available(host: str, timeout_sec: int) -> None:
    tags_url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout_sec) as response:
            _ = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Start Ollama first (e.g. `ollama serve`)."
        ) from exc


def ollama_generate(
    host: str,
    model_tag: str,
    prompt: str,
    max_new_tokens: int,
    timeout_sec: int,
) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": max_new_tokens,
        },
    }
    response = _post_json(url, payload, timeout_sec)
    return str(response.get("response", "")).strip()


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    set_seed(args.seed)

    model_tag = resolve_model_tag(args)
    assert_ollama_available(args.host, args.timeout_sec)

    run_id = make_run_id("benchmark_ollama")
    started_at = datetime.now(timezone.utc)

    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        return ollama_generate(
            args.host,
            model_tag,
            prompt,
            max_new_tokens=max_new_tokens,
            timeout_sec=args.timeout_sec,
        )

    benchmark_results = []
    for bench in args.benchmarks:
        print(f"[benchmark-ollama] Running {bench}...")
        if bench == "hellaswag":
            benchmark_results.append(
                evaluate_hellaswag(generate_fn, args.limit, args.save_predictions)
            )
        elif bench == "arc":
            benchmark_results.append(
                evaluate_arc(generate_fn, args.limit, args.save_predictions)
            )
        elif bench == "gsm8k":
            benchmark_results.append(
                evaluate_gsm8k(generate_fn, args.limit, args.save_predictions)
            )
        else:
            raise ValueError(f"Unknown benchmark: {bench}")

    ended_at = datetime.now(timezone.utc)
    summary = {
        "run_type": "benchmark",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "runtime": {
            "model_id": f"ollama:{model_tag}",
            "precision": "provider_default",
            "device": "ollama_local",
            "cache_dir": None,
        },
        "config": {
            "provider": "ollama",
            "host": args.host,
            "benchmarks": args.benchmarks,
            "limit": args.limit,
            "seed": args.seed,
            "save_predictions": args.save_predictions,
            "decoding": {
                "temperature": 0,
                "top_p": 1,
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
    print(f"[benchmark-ollama] Saved results to: {output_path}")

    for item in benchmark_results:
        print(
            "[benchmark-ollama] "
            f"{item['name']}: accuracy={item['accuracy']:.4f} "
            f"({item['correct']}/{item['num_examples']})"
        )


if __name__ == "__main__":
    main()
