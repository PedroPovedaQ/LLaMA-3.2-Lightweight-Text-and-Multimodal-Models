#!/usr/bin/env python3
"""Run baseline accuracy benchmarks for small language models."""

from __future__ import annotations

import argparse
import random
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

from experiment_utils import (
    RAW_RESULTS_DIR,
    capture_hardware_info,
    ensure_results_dirs,
    format_user_prompt,
    load_model_and_tokenizer,
    make_run_id,
    model_input_device,
    save_json,
)

MODEL_KEYS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
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
        choices=["hellaswag", "arc", "gsm8k"],
        default=["hellaswag", "arc", "gsm8k"],
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return MODEL_KEYS[args.model_key]


def safe_take(dataset: Iterable[dict], limit: int) -> List[dict]:
    if limit <= 0:
        return []
    return [dataset[i] for i in range(min(limit, len(dataset)))]


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


def parse_mc_answer(text: str, choices: List[str]) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    # Priority 1: direct single-letter prediction.
    letter_matches = re.findall(r"\b([A-Z])\b", cleaned.upper())
    for candidate in letter_matches:
        if candidate in LETTERS[: len(choices)]:
            return candidate

    # Priority 2: text overlap with one option.
    lowered = cleaned.lower()
    matched = []
    for idx, choice in enumerate(choices):
        choice_text = choice.strip().lower()
        if choice_text and choice_text in lowered:
            matched.append(LETTERS[idx])

    if len(matched) == 1:
        return matched[0]
    return None


def normalize_numeric(value: str) -> Optional[str]:
    raw = value.strip().replace(",", "")
    if not raw:
        return None

    try:
        dec = Decimal(raw)
    except InvalidOperation:
        return None

    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal("1")))
    return format(dec.normalize(), "f").rstrip("0").rstrip(".")


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    return normalize_numeric(matches[-1])


def evaluate_hellaswag(model, tokenizer, runtime, limit: int, save_predictions: bool) -> Dict[str, object]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets is required for benchmark runs. Install dependencies: pip install -r requirements.txt"
        ) from exc

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

        output = generate_completion(model, tokenizer, runtime, prompt, max_new_tokens=8)
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


def evaluate_arc(model, tokenizer, runtime, limit: int, save_predictions: bool) -> Dict[str, object]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets is required for benchmark runs. Install dependencies: pip install -r requirements.txt"
        ) from exc

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

        output = generate_completion(model, tokenizer, runtime, prompt, max_new_tokens=8)
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


def evaluate_gsm8k(model, tokenizer, runtime, limit: int, save_predictions: bool) -> Dict[str, object]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets is required for benchmark runs. Install dependencies: pip install -r requirements.txt"
        ) from exc

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
        output = generate_completion(model, tokenizer, runtime, prompt, max_new_tokens=128)
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

    model_id = resolve_model_id(args)
    model, tokenizer, runtime = load_model_and_tokenizer(
        model_id=model_id,
        precision=args.precision,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    run_id = make_run_id("benchmark")
    started_at = datetime.now(timezone.utc)

    benchmark_results = []
    for bench in args.benchmarks:
        print(f"[benchmark] Running {bench}...")
        if bench == "hellaswag":
            benchmark_results.append(
                evaluate_hellaswag(model, tokenizer, runtime, args.limit, args.save_predictions)
            )
        elif bench == "arc":
            benchmark_results.append(
                evaluate_arc(model, tokenizer, runtime, args.limit, args.save_predictions)
            )
        elif bench == "gsm8k":
            benchmark_results.append(
                evaluate_gsm8k(model, tokenizer, runtime, args.limit, args.save_predictions)
            )
        else:
            raise ValueError(f"Unknown benchmark: {bench}")

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
        output_path = RAW_RESULTS_DIR / f"benchmark_{run_id}.json"

    save_json(summary, output_path)
    print(f"[benchmark] Saved results to: {output_path}")

    for item in benchmark_results:
        print(
            "[benchmark] "
            f"{item['name']}: accuracy={item['accuracy']:.4f} "
            f"({item['correct']}/{item['num_examples']})"
        )


if __name__ == "__main__":
    main()
