#!/usr/bin/env python3
"""Shared benchmark evaluation logic (prompts, parsing, metrics)."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from experiment_utils import (
    LETTERS,
    extract_last_number,
    normalize_numeric,
    parse_mc_answer,
    safe_take,
)

GenerateFn = Callable[[str, int], str]


def evaluate_hellaswag(
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
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

        output = generate_fn(prompt, 8)
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
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
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

        label_map: Dict[str, str] = {}
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

        output = generate_fn(prompt, 8)
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
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
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

        target: Optional[object] = None
        if "####" in answer:
            target = normalize_numeric(answer.split("####")[-1])
        if target is None:
            target = extract_last_number(answer)

        prompt = (
            "Solve the math word problem.\n"
            f"Question: {question}\n"
            "Return the final numeric answer at the end."
        )
        output = generate_fn(prompt, 128)
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
