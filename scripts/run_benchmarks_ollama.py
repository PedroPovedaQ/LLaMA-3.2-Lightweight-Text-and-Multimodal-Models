#!/usr/bin/env python3
"""Run benchmark evaluations against locally available Ollama models."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

from experiment_utils import (
    LETTERS,
    RAW_RESULTS_DIR,
    capture_hardware_info,
    ensure_results_dirs,
    extract_last_number,
    make_run_id,
    normalize_numeric,
    parse_mc_answer,
    safe_take,
    save_json,
)

MODEL_TAGS = {
    "llama-3.2-1b": "llama3.2:1b",
    "llama-3.2-3b": "llama3.2",
    "phi-3-mini": "phi3:mini",
    "tinyllama": "tinyllama",
}

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
        choices=sorted(MODEL_TAGS.keys()),
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
    return MODEL_TAGS[args.model_key]


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


def evaluate_hellaswag(
    host: str,
    model_tag: str,
    limit: int,
    timeout_sec: int,
    save_predictions: bool,
) -> Dict[str, object]:
    dataset = load_dataset("hellaswag", split="validation")
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()
    for i, item in enumerate(items):
        options = [str(x).strip() for x in item["endings"]]
        answer_letter = LETTERS[int(item["label"])]
        ctx = str(item["ctx"]).strip()

        option_block = "\n".join(f"{LETTERS[idx]}. {text}" for idx, text in enumerate(options))
        prompt = (
            "Choose the best ending for the context.\n"
            f"Context: {ctx}\n"
            f"Options:\n{option_block}\n"
            "Answer with a single letter only."
        )

        output = ollama_generate(host, model_tag, prompt, max_new_tokens=8, timeout_sec=timeout_sec)
        pred = parse_mc_answer(output, options)
        is_correct = pred == answer_letter
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": pred,
                    "target": answer_letter,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    total = len(items)
    accuracy = (correct / total) if total else 0.0
    result: Dict[str, object] = {
        "name": "hellaswag",
        "num_examples": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "duration_sec": round(elapsed, 3),
        "examples_per_sec": round(total / elapsed, 4) if elapsed > 0 else None,
    }
    if save_predictions:
        result["predictions"] = predictions
    return result


def evaluate_arc(
    host: str,
    model_tag: str,
    limit: int,
    timeout_sec: int,
    save_predictions: bool,
) -> Dict[str, object]:
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    items = safe_take(dataset, limit)

    correct = 0
    evaluated = 0
    predictions = []
    start = time.perf_counter()
    for i, item in enumerate(items):
        question = str(item["question"]).strip()
        labels = [str(x).strip() for x in item["choices"]["label"]]
        texts = [str(x).strip() for x in item["choices"]["text"]]

        if not texts:
            continue

        label_map = {}
        for idx, label in enumerate(labels):
            target_letter = LETTERS[idx]
            label_map[label.upper()] = target_letter
            label_map[str(idx + 1)] = target_letter
            label_map[target_letter] = target_letter

        answer_key = str(item["answerKey"]).strip().upper()
        target = label_map.get(answer_key)
        if target is None:
            continue

        evaluated += 1
        options = texts
        option_block = "\n".join(f"{LETTERS[idx]}. {text}" for idx, text in enumerate(options))
        prompt = (
            "Answer the multiple-choice question.\n"
            f"Question: {question}\n"
            f"Options:\n{option_block}\n"
            "Answer with a single letter only."
        )

        output = ollama_generate(host, model_tag, prompt, max_new_tokens=8, timeout_sec=timeout_sec)
        pred = parse_mc_answer(output, options)
        is_correct = pred == target
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": pred,
                    "target": target,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    total = evaluated
    accuracy = (correct / total) if total else 0.0
    result: Dict[str, object] = {
        "name": "arc",
        "num_examples": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "duration_sec": round(elapsed, 3),
        "examples_per_sec": round(total / elapsed, 4) if elapsed > 0 else None,
    }
    if save_predictions:
        result["predictions"] = predictions
    return result


def evaluate_gsm8k(
    host: str,
    model_tag: str,
    limit: int,
    timeout_sec: int,
    save_predictions: bool,
) -> Dict[str, object]:
    dataset = load_dataset("gsm8k", "main", split="test")
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()
    for i, item in enumerate(items):
        question = str(item["question"]).strip()
        answer = str(item["answer"])

        target = None
        if "####" in answer:
            target = normalize_numeric(answer.split("####")[-1])
        if target is None:
            target = extract_last_number(answer)

        prompt = (
            "Solve the math word problem.\n"
            f"Question: {question}\n"
            "Return the final numeric answer at the end."
        )
        output = ollama_generate(host, model_tag, prompt, max_new_tokens=128, timeout_sec=timeout_sec)
        pred = extract_last_number(output)

        is_correct = pred is not None and target is not None and pred == target
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": pred,
                    "target": target,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    total = len(items)
    accuracy = (correct / total) if total else 0.0
    result: Dict[str, object] = {
        "name": "gsm8k",
        "num_examples": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "duration_sec": round(elapsed, 3),
        "examples_per_sec": round(total / elapsed, 4) if elapsed > 0 else None,
    }
    if save_predictions:
        result["predictions"] = predictions
    return result


def main() -> None:
    args = parse_args()
    ensure_results_dirs()
    set_seed(args.seed)

    model_tag = resolve_model_tag(args)
    assert_ollama_available(args.host, args.timeout_sec)

    run_id = make_run_id("benchmark_ollama")
    started_at = datetime.now(timezone.utc)

    benchmark_results = []
    for bench in args.benchmarks:
        print(f"[benchmark-ollama] Running {bench}...")
        if bench == "hellaswag":
            benchmark_results.append(
                evaluate_hellaswag(
                    args.host,
                    model_tag,
                    args.limit,
                    args.timeout_sec,
                    args.save_predictions,
                )
            )
        elif bench == "arc":
            benchmark_results.append(
                evaluate_arc(
                    args.host,
                    model_tag,
                    args.limit,
                    args.timeout_sec,
                    args.save_predictions,
                )
            )
        elif bench == "gsm8k":
            benchmark_results.append(
                evaluate_gsm8k(
                    args.host,
                    model_tag,
                    args.limit,
                    args.timeout_sec,
                    args.save_predictions,
                )
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
        output_path = RAW_RESULTS_DIR / f"benchmark_ollama_{run_id}.json"

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
