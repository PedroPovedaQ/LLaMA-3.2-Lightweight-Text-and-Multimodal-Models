"""
eval_tldr.py — TLDR9+ Summarization Evaluation
=================================================
Evaluates models on Reddit post summarization.

Protocol from the paper:
    - 1-shot (one example summary provided)
    - Metric: ROUGE-L against reference TL;DR summaries
    - Split: test

Dataset: CarperAI/openai_summarize_tldr
Columns: 'prompt' (formatted Reddit post) and 'label' (reference summary)

Usage:
    python eval_tldr.py --model llama-3.2-1b
"""

import argparse
import logging
import random
from tqdm import tqdm
from datasets import load_dataset

from config import TLDR9Config, get_model_config
from model_loader import load_model_and_tokenizer, generate_response
from utils import compute_rouge_l, save_results, setup_logging

logger = logging.getLogger(__name__)


def load_tldr_data(config: TLDR9Config, max_samples: int = None):
    """Load the TLDR summarization dataset."""
    logger.info("Loading TLDR dataset...")

    try:
        dataset = load_dataset(config.dataset_name, split=config.test_split)
    except Exception as e:
        logger.warning(f"Could not load {config.dataset_name}: {e}")
        logger.info("Falling back to 'webis/tldr-17'")
        dataset = load_dataset("webis/tldr-17", split="test")

    logger.info(f"Loaded {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    if len(dataset) > 0:
        logger.info(f"First example keys: {list(dataset[0].keys())}")
        for col in dataset.column_names:
            val = dataset[0][col]
            if isinstance(val, str):
                logger.info(f"  {col}: '{val[:80]}...'")

    cap = max_samples or config.max_samples
    if cap and cap < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(cap))
        logger.info(f"Capped to {cap} examples")

    return dataset


def get_post_and_summary(example):
    """
    Extract the post text and reference summary from a dataset example.

    The CarperAI/openai_summarize_tldr dataset has these columns:
    - prompt: the formatted Reddit post (SUBREDDIT + TITLE + POST)
    - label: the reference TL;DR summary
    """
    post = None
    summary = None

    # Try to get the post content — "prompt" is what CarperAI uses
    if "prompt" in example and example["prompt"]:
        post = str(example["prompt"])
    elif "post" in example and example["post"]:
        post = str(example["post"])
    elif "query" in example and example["query"]:
        post = str(example["query"])
    elif "content" in example and example["content"]:
        post = str(example["content"])
    elif "document" in example and example["document"]:
        post = str(example["document"])
    elif "text" in example and example["text"]:
        post = str(example["text"])

    # Try to get the summary — "label" is what CarperAI uses
    if "label" in example and example["label"]:
        summary = str(example["label"])
    elif "summary" in example and example["summary"]:
        summary = str(example["summary"])
    elif "reference_response" in example and example["reference_response"]:
        summary = str(example["reference_response"])
    elif "tldr" in example and example["tldr"]:
        summary = str(example["tldr"])

    return post, summary


def build_1shot_prompt(test_post: str, shot_example: dict) -> str:
    """
    Build the 1-shot prompt: one solved example, then the test post.

    The CarperAI dataset's "prompt" field is already formatted as:
    SUBREDDIT: r/xxx
    TITLE: xxx
    POST: xxx
    TL;DR:

    So we use it directly and just prepend one solved example.
    """
    shot_post, shot_summary = get_post_and_summary(shot_example)

    prompt_parts = []

    # 1-shot example
    if shot_post and shot_summary:
        if "TL;DR:" in shot_post:
            prompt_parts.append(f"{shot_post[:800]} {shot_summary}")
        else:
            prompt_parts.append(
                f"Post: {shot_post[:500]}\n\nTL;DR: {shot_summary}"
            )
        prompt_parts.append("")

    # Test question
    if "TL;DR:" in test_post:
        prompt_parts.append(test_post[:1000])
    else:
        prompt_parts.append(
            f"Summarize this post in one or two sentences.\n\n"
            f"Post: {test_post[:800]}\n\nTL;DR:"
        )

    return "\n".join(prompt_parts)


def evaluate_tldr(model_name: str, max_samples: int = None):
    """Run the TLDR9+ evaluation. Returns avg ROUGE-L (0-100)."""
    config = TLDR9Config()
    model_config = get_model_config(model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)

    dataset = load_tldr_data(config, max_samples)

    # Pick a fixed 1-shot example
    random.seed(42)
    shot_idx = random.randint(0, min(50, len(dataset) - 1))
    shot_example = dataset[shot_idx]

    results = []
    total_rouge = 0.0
    evaluated = 0

    logger.info(f"Starting TLDR9+ eval ({len(dataset)} examples)...")

    for idx, example in enumerate(tqdm(dataset, desc="TLDR9+")):
        if idx == shot_idx:
            continue

        post, reference_summary = get_post_and_summary(example)
        if not post or not reference_summary:
            continue

        prompt = build_1shot_prompt(post, shot_example)

        try:
            response = generate_response(
                model, tokenizer, prompt,
                model_family=model_config.model_family,
                max_new_tokens=150,
            )
        except Exception as e:
            logger.error(f"Example {idx}: Generation failed: {e}")
            response = ""

        rouge_l = compute_rouge_l(response, reference_summary)
        total_rouge += rouge_l
        evaluated += 1

        results.append({
            "idx": idx,
            "post": post[:200],
            "reference": reference_summary[:200],
            "response": response[:200],
            "rouge_l": round(rouge_l, 2),
        })

        if (evaluated) % 100 == 0:
            avg = total_rouge / evaluated
            logger.info(f"  Progress: {evaluated}/{len(dataset)}, "
                        f"avg ROUGE-L: {avg:.1f}")

    avg_rouge = total_rouge / evaluated if evaluated else 0.0
    paper_score = config.paper_scores.get(model_name, None)

    logger.info(f"\n{'='*60}")
    logger.info(f"TLDR9+ Results for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Avg ROUGE-L:            {avg_rouge:.1f}")
    if paper_score:
        logger.info(f"Paper's reported score: {paper_score}")
        logger.info(f"Difference:             {avg_rouge - paper_score:+.1f}")

    save_results(results, model_name, "tldr9",
                 metadata={"paper_score": paper_score, "metric": "rougeL",
                           "num_shots": 1})

    return avg_rouge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TLDR9+ eval")
    parser.add_argument("--model", type=str, default="llama-3.2-1b")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    score = evaluate_tldr(args.model, args.max_samples)
    print(f"\nFinal TLDR9+ ROUGE-L: {score:.1f}")
