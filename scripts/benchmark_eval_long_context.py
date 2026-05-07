#!/usr/bin/env python3
"""Shared benchmark evaluation logic (prompts, parsing, metrics)."""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional, Sequence as TypingSequence

from experiment_utils import (
    LETTERS,
    extract_last_number,
    normalize_numeric,
    parse_mc_answer,
    safe_take,
)

GenerateFn = Callable[[str, int], str]
BenchmarkEvaluator = Callable[[GenerateFn, int, bool], Dict[str, object]]

# Keeps InfiniteBench usable on small GPUs / small context windows.
# Increase this if you have more VRAM and want a harder long-context test.
MAX_CONTEXT_CHARS = 4000


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Keep the beginning and end of a long context so important clues are less likely to be dropped."""
    context = str(context).strip()
    if len(context) <= max_chars:
        return context

    half = max_chars // 2
    return (
        context[:half]
        + "\n\n[... context truncated for small-model benchmark ...]\n\n"
        + context[-half:]
    )


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def normalize_answer_value(answer: object) -> str:
    """Convert list/string answers into a clean target string."""
    if isinstance(answer, list):
        if not answer:
            return ""
        answer = answer[0]
    return str(answer).strip()


def loose_match(pred: object, target: object) -> bool:
    """Loose text match for QA-style outputs."""
    pred_s = normalize_text(pred)
    target_s = normalize_text(target)

    if not pred_s or not target_s:
        return False

    # Direct containment either way.
    if target_s in pred_s or pred_s in target_s:
        return True

    # Numeric fallback.
    pred_num = extract_last_number(pred_s)
    target_num = extract_last_number(target_s)
    if pred_num is not None and target_num is not None:
        return numeric_match(pred_num, target_num)

    # Token-overlap fallback for short answer strings.
    target_tokens = [t for t in re.findall(r"[a-z0-9]+", target_s) if len(t) > 1]
    if target_tokens:
        overlap = sum(1 for t in target_tokens if t in pred_s)
        return overlap / len(target_tokens) >= 0.75

    return False


def numeric_match(pred: object, target: object, tolerance: float = 1e-3) -> bool:
    """Compare numeric answers exactly after normalization, then approximately."""
    if pred is None or target is None:
        return False

    pred_s = normalize_numeric(str(pred)) or extract_last_number(str(pred))
    target_s = normalize_numeric(str(target)) or extract_last_number(str(target))

    if pred_s is None or target_s is None:
        return False

    if pred_s == target_s:
        return True

    try:
        return abs(float(pred_s) - float(target_s)) <= tolerance
    except Exception:
        return False


def build_mc_prompt(question: str, option_block: str, context: Optional[str] = None) -> str:
    """Few-shot multiple-choice prompt to make small models more likely to output a letter."""
    parts = [
        "Answer the multiple-choice question. Choose exactly one option.",
        "",
        "Example:",
        "Question: What is 2 + 2?",
        "Options:",
        "A. 3",
        "B. 4",
        "C. 5",
        "D. 6",
        "Answer: B",
        "",
    ]

    if context:
        parts.extend(
            [
                "Use the following context when answering.",
                f"Context:\n{context}",
                "",
            ]
        )

    parts.extend(
        [
            f"Question: {question}",
            f"Options:\n{option_block}",
            "Answer with a single letter only.",
            "Answer:",
        ]
    )
    return "\n".join(parts)


def build_math_prompt(question: str) -> str:
    """Few-shot math prompt with a clear final-answer format."""
    return (
        "Solve the math problem. Give brief reasoning, then finish with 'Final answer: <number>'.\n\n"
        "Example:\n"
        "Question: A box has 3 bags with 4 apples each. How many apples are there?\n"
        "Reasoning: 3 bags times 4 apples each is 12.\n"
        "Final answer: 12\n\n"
        f"Question: {question}\n"
        "Reasoning:"
    )


def _result(
    name: str,
    total: int,
    correct: int,
    elapsed: float,
    predictions: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "name": name,
        "num_examples": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "duration_sec": round(elapsed, 3),
        "examples_per_sec": round(total / elapsed, 4) if elapsed > 0 else None,
    }
    if predictions is not None:
        result["predictions"] = predictions
    return result


def infinitebench_features():
    """Features schema used to avoid dataset casting problems with InfiniteBench."""
    from datasets import Features, Sequence, Value

    return Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )


# ---------------------------------------------------------------------
# Standard benchmarks
# ---------------------------------------------------------------------

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
        prompt = build_mc_prompt(
            question=f"Choose the best ending for this context: {ctx}",
            option_block=option_block,
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
    return _result(
        "hellaswag",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


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
        prompt = build_mc_prompt(question=question, option_block=option_block)

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
    return _result(
        "arc",
        evaluated,
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


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

        target: Optional[str] = None
        if "####" in answer:
            target = normalize_numeric(answer.split("####")[-1])
        if target is None:
            target = extract_last_number(answer)

        prompt = build_math_prompt(question)

        output = generate_fn(prompt, 128)
        pred = extract_last_number(output)
        is_correct = numeric_match(pred, target)
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
    return _result(
        "gsm8k",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


def evaluate_mmlu(
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

    # This dataset name works with modern datasets versions.
    dataset = load_dataset("cais/mmlu", "all", split="test")
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()

    for i, item in enumerate(items):
        question = str(item["question"]).strip()
        options = [str(x).strip() for x in item["choices"]]
        target = LETTERS[int(item["answer"])]

        option_block = "\n".join(f"{LETTERS[idx]}. {text}" for idx, text in enumerate(options))
        prompt = build_mc_prompt(question=question, option_block=option_block)

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
    return _result(
        "mmlu",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


# ---------------------------------------------------------------------
# Picture-added benchmarks
# ---------------------------------------------------------------------

def evaluate_mgsm(
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
    """MGSM-style math benchmark.

    The original juletxara/mgsm repo uses a dataset script blocked by recent
    Hugging Face datasets versions, so this uses GSM8K as a stable English
    math proxy while keeping the benchmark key as 'mgsm' for your project table.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "datasets is required for benchmark runs. Install dependencies: pip install -r requirements.txt"
        ) from exc

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()

    for i, item in enumerate(items):
        question = str(item["question"]).strip()
        target = extract_last_number(str(item["answer"]))

        prompt = build_math_prompt(question)

        output = generate_fn(prompt, 128)
        pred = extract_last_number(output)
        is_correct = numeric_match(pred, target)
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
    return _result(
        "mgsm",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


def evaluate_infinitebench_en_mc(
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

    dataset = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        split="longbook_choice_eng",
        features=infinitebench_features(),
    )
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()

    for i, item in enumerate(items):
        context = truncate_context(str(item.get("context", item.get("input", ""))))
        question = str(item.get("input", "") or item.get("question", "")).strip()
        options = [str(x).strip() for x in item.get("options", []) if str(x).strip()]

        # InfiniteBench answers are stored as a list in this dataset.
        target_raw = normalize_answer_value(item.get("answer", ""))
        target = target_raw.strip().upper()

        if not options:
            # Fallback if an item has no options field.
            options = ["A", "B", "C", "D"]

        option_block = "\n".join(f"{LETTERS[idx]}. {text}" for idx, text in enumerate(options))
        prompt = build_mc_prompt(question=question, option_block=option_block, context=context)

        output = generate_fn(prompt, 8)
        pred = parse_mc_answer(output, options)

        # Some InfiniteBench answers may be the option text rather than the letter.
        is_correct = pred == target or loose_match(output, target_raw)
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": pred,
                    "target": target_raw,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    return _result(
        "infinitebench_en_mc",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


def evaluate_infinitebench_en_qa(
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

    dataset = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        split="longbook_qa_eng",
        features=infinitebench_features(),
    )
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()

    for i, item in enumerate(items):
        context = truncate_context(str(item.get("context", "")))
        question = str(item.get("input", "") or item.get("question", "")).strip()
        target = normalize_answer_value(item.get("answer", ""))

        prompt = (
            "Answer the question using only the context. "
            "Give a concise answer.\n\n"
            "Example:\n"
            "Context: Alice lives in Paris. Bob lives in Rome.\n"
            "Question: Where does Alice live?\n"
            "Answer: Paris\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        output = generate_fn(prompt, 64)
        is_correct = loose_match(output, target)
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": output,
                    "target": target,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    return _result(
        "infinitebench_en_qa",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


def evaluate_nih_multi_needle(
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

    dataset = load_dataset(
        "xinrongzhang2022/InfiniteBench",
        split="passkey",
        features=infinitebench_features(),
    )
    items = safe_take(dataset, limit)

    correct = 0
    predictions = []
    start = time.perf_counter()

    for i, item in enumerate(items):
        context = truncate_context(str(item.get("context", item.get("input", ""))))
        target = normalize_answer_value(item.get("answer", ""))

        prompt = (
            "Find the hidden passkey in the context. "
            "Return only the hidden value.\n\n"
            "Example:\n"
            "Context: The pass key is 12345. Remember it.\n"
            "Answer: 12345\n\n"
            f"Context:\n{context}\n"
            "Answer:"
        )

        output = generate_fn(prompt, 32)
        is_correct = loose_match(output, target)
        correct += int(is_correct)

        if save_predictions:
            predictions.append(
                {
                    "index": i,
                    "prediction": output,
                    "target": target,
                    "correct": is_correct,
                    "raw_output": output,
                }
            )

    elapsed = time.perf_counter() - start
    return _result(
        "nih_multi_needle",
        len(items),
        correct,
        elapsed,
        predictions if save_predictions else None,
    )


# Benchmark registry entry point:
# Add new benchmark evaluators here so runner scripts pick them up automatically.
BENCHMARK_EVALUATORS: Dict[str, BenchmarkEvaluator] = {
    "hellaswag": evaluate_hellaswag,
    "arc": evaluate_arc,
    "gsm8k": evaluate_gsm8k,
    "mmlu": evaluate_mmlu,
    "infinitebench_en_mc": evaluate_infinitebench_en_mc,
    "infinitebench_en_qa": evaluate_infinitebench_en_qa,
    "nih_multi_needle": evaluate_nih_multi_needle,
    "mgsm": evaluate_mgsm,
}


def benchmark_names() -> List[str]:
    """Ordered benchmark keys available to CLI users."""
    return list(BENCHMARK_EVALUATORS.keys())


def run_benchmark(
    name: str,
    generate_fn: GenerateFn,
    limit: int,
    save_predictions: bool,
) -> Dict[str, object]:
    """Dispatch helper used by benchmark runner scripts."""
    evaluator = BENCHMARK_EVALUATORS.get(name)
    if evaluator is None:
        valid = ", ".join(benchmark_names())
        raise ValueError(f"Unknown benchmark {name!r}. Valid values: {valid}")
    return evaluator(generate_fn, limit, save_predictions)
