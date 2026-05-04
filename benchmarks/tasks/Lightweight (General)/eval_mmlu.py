"""
eval_mmlu.py — MMLU 5-shot Evaluation
=======================================
Evaluates text models on MMLU (Massive Multitask Language Understanding).

Protocol from the paper:
    - 5-shot (5 solved examples prepended to each test question)
    - Metric: accuracy (percentage of correct answers)
    - 57 subjects, ~14,000 test questions total

Usage:
    python eval_mmlu.py --model llama-3.2-1b
    python eval_mmlu.py --model llama-3.2-3b --max_samples 100
"""

import argparse
import logging
import random
from tqdm import tqdm
from datasets import load_dataset

from config import MMLUConfig, get_model_config, MMLU_EXAMPLE_TEMPLATE, MMLU_QUESTION_TEMPLATE
from model_loader import load_model_and_tokenizer, generate_completion
from utils import extract_answer_letter, save_results, setup_logging

logger = logging.getLogger(__name__)

# MMLU answer index -> letter mapping
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu_data(config: MMLUConfig, max_samples: int = None):
    """Load MMLU test and few-shot (validation) splits."""
    logger.info("Loading MMLU dataset...")

    test_ds = load_dataset(
        config.dataset_name, config.dataset_config,
        split=config.test_split,
    )

    few_shot_ds = load_dataset(
        config.dataset_name, config.dataset_config,
        split=config.few_shot_split,
    )

    logger.info(f"Test: {len(test_ds)} examples, Few-shot pool: {len(few_shot_ds)}")

    if max_samples and max_samples < len(test_ds):
        test_ds = test_ds.shuffle(seed=42).select(range(max_samples))
        logger.info(f"Truncated to {max_samples} test samples")

    return test_ds, few_shot_ds


def format_mmlu_example(example) -> str:
    """Format a single MMLU example as a solved Q&A string."""
    choices = example["choices"]
    answer_idx = example["answer"]
    answer_letter = INDEX_TO_LETTER.get(answer_idx, str(answer_idx))

    return MMLU_EXAMPLE_TEMPLATE.format(
        question=example["question"],
        A=choices[0], B=choices[1], C=choices[2], D=choices[3],
        answer=answer_letter,
    )


def format_mmlu_question(example) -> str:
    """Format a test question (without the answer)."""
    choices = example["choices"]
    return MMLU_QUESTION_TEMPLATE.format(
        question=example["question"],
        A=choices[0], B=choices[1], C=choices[2], D=choices[3],
    )


def build_5shot_prompt(test_example, few_shot_pool, num_shots: int = 5) -> str:
    """
    Build the full 5-shot prompt for a test question.
    Picks examples from the same subject when possible.
    """
    test_subject = test_example.get("subject", "")

    same_subject = [ex for ex in few_shot_pool
                    if ex.get("subject", "") == test_subject]

    if len(same_subject) >= num_shots:
        selected = random.sample(same_subject, num_shots)
    elif same_subject:
        remaining = num_shots - len(same_subject)
        others = random.sample(
            [ex for ex in few_shot_pool
             if ex.get("subject", "") != test_subject],
            min(remaining, len(few_shot_pool) - len(same_subject))
        )
        selected = same_subject + others
    else:
        selected = random.sample(
            list(few_shot_pool),
            min(num_shots, len(few_shot_pool))
        )

    parts = []
    for ex in selected:
        parts.append(format_mmlu_example(ex))
    parts.append(format_mmlu_question(test_example))

    return "\n\n".join(parts)


def evaluate_mmlu(model_name: str, max_samples: int = None):
    """Run the full MMLU 5-shot evaluation. Returns accuracy (0-100)."""
    config = MMLUConfig()
    model_config = get_model_config(model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)

    test_ds, few_shot_ds = load_mmlu_data(config, max_samples)
    few_shot_list = list(few_shot_ds)

    random.seed(42)

    results = []
    correct = 0

    logger.info(f"Starting MMLU 5-shot evaluation ({len(test_ds)} questions)...")

    for idx, example in enumerate(tqdm(test_ds, desc="MMLU")):
        prompt = build_5shot_prompt(example, few_shot_list, config.num_shots)

        try:
            response = generate_completion(
                model, tokenizer, prompt, max_new_tokens=5
            )
        except Exception as e:
            logger.error(f"Example {idx}: Generation failed: {e}")
            response = ""

        predicted = extract_answer_letter(response)

        gt_idx = example["answer"]
        ground_truth = INDEX_TO_LETTER.get(gt_idx, str(gt_idx))

        is_correct = (predicted == ground_truth)
        if is_correct:
            correct += 1

        results.append({
            "idx": idx,
            "subject": example.get("subject", "unknown"),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:100],
        })

        if (idx + 1) % 200 == 0:
            acc = 100.0 * correct / (idx + 1)
            logger.info(f"  Progress: {idx+1}/{len(test_ds)}, accuracy: {acc:.1f}%")

    accuracy = 100.0 * correct / len(results) if results else 0.0

    paper_score = config.paper_scores.get(model_name, None)
    logger.info(f"\n{'='*60}")
    logger.info(f"MMLU Results for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:               {accuracy:.1f}%")
    if paper_score:
        logger.info(f"Paper's reported score: {paper_score}%")
        logger.info(f"Difference:             {accuracy - paper_score:+.1f}%")

    save_results(results, model_name, "mmlu",
                 metadata={"paper_score": paper_score, "num_shots": 5})

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU 5-shot evaluation")
    parser.add_argument("--model", type=str, default="llama-3.2-1b")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    accuracy = evaluate_mmlu(args.model, args.max_samples)
    print(f"\nFinal MMLU accuracy: {accuracy:.1f}%")
