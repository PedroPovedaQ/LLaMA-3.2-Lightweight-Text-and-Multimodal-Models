"""
eval_open_rewrite.py — Open-Rewrite Evaluation
================================================
Evaluates models on open-ended text rewriting ability.

Protocol from the paper:
    - 0-shot (no examples provided)
    - Metric: ROUGE-L (measures overlap with reference rewrite)

Usage:
    python eval_open_rewrite.py --model llama-3.2-1b
"""

import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset

from config import OpenRewriteConfig, get_model_config, REWRITE_PROMPT
from model_loader import load_model_and_tokenizer, generate_response
from utils import compute_rouge_l, save_results, setup_logging

logger = logging.getLogger(__name__)


def load_rewrite_data(config: OpenRewriteConfig, max_samples: int = None):
    """Load a rewriting/paraphrase dataset."""
    logger.info("Loading rewrite evaluation dataset...")

    try:
        dataset = load_dataset(config.dataset_name, split=config.test_split)
    except Exception as e:
        logger.warning(f"Could not load {config.dataset_name}: {e}")
        logger.info("Falling back to 'sentence-transformers/parallel-sentences'")
        dataset = load_dataset(
            "sentence-transformers/parallel-sentences", "tatoeba",
            split="train",
        )

    cap = max_samples or config.max_samples
    if cap and cap < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(cap))

    logger.info(f"Loaded {len(dataset)} examples for rewrite eval")
    return dataset


def get_source_and_reference(example):
    """Extract source text and reference rewrite from a dataset example."""
    source_keys = ["source", "input", "text", "original", "sentence1",
                   "english", "prompt"]
    target_keys = ["target", "output", "rewrite", "paraphrase", "sentence2",
                   "translation", "response"]

    source = None
    reference = None

    for key in source_keys:
        if key in example and example[key]:
            source = str(example[key])
            break

    for key in target_keys:
        if key in example and example[key]:
            reference = str(example[key])
            break

    if source and not reference:
        reference = source

    return source, reference


def evaluate_open_rewrite(model_name: str, max_samples: int = None):
    """Run the open-rewrite evaluation. Returns avg ROUGE-L (0-100)."""
    config = OpenRewriteConfig()
    model_config = get_model_config(model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)

    dataset = load_rewrite_data(config, max_samples)

    results = []
    total_rouge = 0.0

    logger.info(f"Starting Open-rewrite eval ({len(dataset)} examples)...")

    for idx, example in enumerate(tqdm(dataset, desc="Open-rewrite")):
        source, reference = get_source_and_reference(example)
        if not source:
            continue

        prompt = REWRITE_PROMPT.format(text=source[:500])

        try:
            response = generate_response(
                model, tokenizer, prompt,
                model_family=model_config.model_family,
                max_new_tokens=model_config.max_new_tokens,
            )
        except Exception as e:
            logger.error(f"Example {idx}: Generation failed: {e}")
            response = ""

        rouge_l = compute_rouge_l(response, reference) if reference else 0.0
        total_rouge += rouge_l

        results.append({
            "idx": idx,
            "source": source[:200],
            "reference": (reference or "")[:200],
            "response": response[:200],
            "rouge_l": round(rouge_l, 2),
        })

        if (idx + 1) % 100 == 0:
            avg = total_rouge / (idx + 1)
            logger.info(f"  Progress: {idx+1}/{len(dataset)}, avg ROUGE-L: {avg:.1f}")

    avg_rouge = total_rouge / len(results) if results else 0.0
    paper_score = config.paper_scores.get(model_name, None)

    logger.info(f"\n{'='*60}")
    logger.info(f"Open-rewrite Results for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Avg ROUGE-L:            {avg_rouge:.1f}")
    if paper_score:
        logger.info(f"Paper's reported score: {paper_score}")
        logger.info(f"Difference:             {avg_rouge - paper_score:+.1f}")

    save_results(results, model_name, "open_rewrite",
                 metadata={"paper_score": paper_score, "metric": "rougeL"})

    return avg_rouge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-rewrite eval")
    parser.add_argument("--model", type=str, default="llama-3.2-1b")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    score = evaluate_open_rewrite(args.model, args.max_samples)
    print(f"\nFinal Open-rewrite ROUGE-L: {score:.1f}")
