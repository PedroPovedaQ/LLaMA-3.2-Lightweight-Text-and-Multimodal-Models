"""
eval_ifeval.py — IFEval (Instruction Following Evaluation)
============================================================
Tests whether models can follow specific, verifiable instructions.

Protocol from the paper:
    - Metric: prompt-level strict accuracy
    - Each prompt has 1+ verifiable constraints
    - A prompt is "correct" only if ALL constraints are satisfied

Dataset: google/IFEval on HuggingFace (541 prompts)

Usage:
    python eval_ifeval.py --model llama-3.2-1b
"""

import argparse
import logging
import re
from tqdm import tqdm
from datasets import load_dataset

from config import IFEvalConfig, get_model_config
from model_loader import load_model_and_tokenizer, generate_response
from utils import save_results, setup_logging

logger = logging.getLogger(__name__)


def load_ifeval_data(config: IFEvalConfig, max_samples: int = None):
    """Load the IFEval dataset."""
    logger.info("Loading IFEval dataset...")

    dataset = load_dataset(config.dataset_name, split=config.test_split)

    logger.info(f"Loaded {len(dataset)} prompts")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Truncated to {max_samples} prompts")

    return dataset


def check_instruction(response: str, instruction_id: str,
                      kwargs: dict) -> bool:
    """
    Check if a single instruction constraint is satisfied.

    IFEval has many constraint types. We implement the most common ones.
    """
    response_lower = response.lower()

    # ---- Keyword constraints ----
    if "keywords:existence" in instruction_id:
        keywords = kwargs.get("keywords", [])
        return all(kw.lower() in response_lower for kw in keywords)

    elif "keywords:frequency" in instruction_id:
        keyword = kwargs.get("keyword", "").lower()
        frequency = kwargs.get("frequency", 1)
        return response_lower.count(keyword) >= frequency

    elif "keywords:forbidden_words" in instruction_id:
        words = kwargs.get("forbidden_words", [])
        return not any(w.lower() in response_lower for w in words)

    elif "keywords:letter_frequency" in instruction_id:
        letter = kwargs.get("letter", "").lower()
        let_num = kwargs.get("let_count", 0) or kwargs.get("num_letters", 0)
        actual = response_lower.count(letter)
        return actual >= let_num

    # ---- Length constraints ----
    elif "length_constraints:number_words" in instruction_id:
        words = response.split()
        relation = kwargs.get("relation", "at least")
        num = kwargs.get("num_words", 0)
        if "at least" in relation:
            return len(words) >= num
        elif "at most" in relation:
            return len(words) <= num
        return True

    elif "length_constraints:number_sentences" in instruction_id:
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        relation = kwargs.get("relation", "at least")
        num = kwargs.get("num_sentences", 0)
        if "at least" in relation:
            return len(sentences) >= num
        elif "at most" in relation:
            return len(sentences) <= num
        return True

    elif "length_constraints:number_paragraphs" in instruction_id:
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in response.split("\n") if p.strip()]
        num = kwargs.get("num_paragraphs", 0)
        return len(paragraphs) >= num

    elif "length_constraints:nth_paragraph_first_word" in instruction_id:
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        n = kwargs.get("num_paragraphs", 1)
        first_word = kwargs.get("first_word", "").lower()
        if n <= len(paragraphs):
            actual_first = paragraphs[n - 1].split()[0].lower() if paragraphs[n - 1].split() else ""
            return actual_first == first_word
        return False

    # ---- Format constraints ----
    elif "detectable_format:number_bullet_lists" in instruction_id:
        bullets = re.findall(r'^\s*[\*\-\•]', response, re.MULTILINE)
        num = kwargs.get("num_bullets", 0)
        return len(bullets) >= num

    elif "detectable_format:number_highlighted_sections" in instruction_id:
        highlights = re.findall(r'\*[^*]+\*', response)
        num = kwargs.get("num_highlights", 0)
        return len(highlights) >= num

    elif "detectable_format:title" in instruction_id:
        lines = response.strip().split("\n")
        if lines:
            first = lines[0].strip()
            return bool(first) and not first.endswith(".")
        return False

    elif "detectable_format:json_format" in instruction_id:
        try:
            import json
            json.loads(response.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return "{" in response and "}" in response

    # ---- Content constraints ----
    elif "detectable_content:postscript" in instruction_id:
        markers = ["P.S.", "P.S", "PS:", "PS.", "p.s."]
        return any(m in response for m in markers)

    elif "detectable_content:number_placeholders" in instruction_id:
        placeholders = re.findall(r'\[.*?\]', response)
        num = kwargs.get("num_placeholders", 0)
        return len(placeholders) >= num

    elif "startend:end_checker" in instruction_id:
        end_phrase = kwargs.get("end_phrase", "").lower()
        return response_lower.rstrip().endswith(end_phrase.rstrip())

    elif "startend:quotation" in instruction_id:
        return response.strip().startswith('"') and response.strip().endswith('"')

    elif "change_case:english_lowercase" in instruction_id:
        letters = re.findall(r'[a-zA-Z]', response)
        if not letters:
            return True
        return all(c.islower() for c in letters)

    elif "change_case:english_capital" in instruction_id:
        sentences = re.split(r'[.!?]\s+', response)
        return all(s[0].isupper() for s in sentences if s)

    elif "punctuation:no_comma" in instruction_id:
        return "," not in response

    # ---- Default: pass (unknown constraint type) ----
    else:
        logger.debug(f"Unknown instruction type: {instruction_id}")
        return True


def evaluate_ifeval(model_name: str, max_samples: int = None):
    """Run the IFEval evaluation. Returns prompt-level strict accuracy (0-100)."""
    config = IFEvalConfig()
    model_config = get_model_config(model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)

    dataset = load_ifeval_data(config, max_samples)

    results = []
    correct = 0

    logger.info(f"Starting IFEval ({len(dataset)} prompts)...")

    for idx, example in enumerate(tqdm(dataset, desc="IFEval")):
        prompt = example.get("prompt", "")

        try:
            response = generate_response(
                model, tokenizer, prompt,
                model_family=model_config.model_family,
                max_new_tokens=model_config.max_new_tokens,
            )
        except Exception as e:
            logger.error(f"Prompt {idx}: Generation failed: {e}")
            response = ""

        # Check all constraints
        instruction_id_list = example.get("instruction_id_list", [])
        kwargs_list = example.get("kwargs", [])

        if isinstance(kwargs_list, str):
            import json
            try:
                kwargs_list = json.loads(kwargs_list)
            except Exception:
                kwargs_list = [{}] * len(instruction_id_list)

        all_passed = True
        constraint_results = []

        for inst_id, kw in zip(instruction_id_list, kwargs_list):
            if isinstance(kw, str):
                import json
                try:
                    kw = json.loads(kw)
                except Exception:
                    kw = {}

            passed = check_instruction(response, inst_id, kw or {})
            constraint_results.append({
                "instruction_id": inst_id,
                "passed": passed,
            })
            if not passed:
                all_passed = False

        if all_passed:
            correct += 1

        results.append({
            "idx": idx,
            "prompt": prompt[:200],
            "response": response[:300],
            "correct": all_passed,
            "num_constraints": len(instruction_id_list),
            "constraints_passed": sum(1 for c in constraint_results if c["passed"]),
            "constraint_details": constraint_results,
        })

        if (idx + 1) % 100 == 0:
            acc = 100.0 * correct / (idx + 1)
            logger.info(f"  Progress: {idx+1}/{len(dataset)}, accuracy: {acc:.1f}%")

    accuracy = 100.0 * correct / len(results) if results else 0.0
    paper_score = config.paper_scores.get(model_name, None)

    total_constraints = sum(r["num_constraints"] for r in results)
    passed_constraints = sum(r["constraints_passed"] for r in results)
    inst_accuracy = 100.0 * passed_constraints / total_constraints if total_constraints else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"IFEval Results for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Prompt-level strict accuracy: {accuracy:.1f}%")
    logger.info(f"Instruction-level accuracy:   {inst_accuracy:.1f}%")
    if paper_score:
        logger.info(f"Paper's reported score:       {paper_score}%")
        logger.info(f"Difference:                   {accuracy - paper_score:+.1f}%")

    save_results(results, model_name, "ifeval",
                 metadata={"paper_score": paper_score,
                           "prompt_level_accuracy": round(accuracy, 2),
                           "instruction_level_accuracy": round(inst_accuracy, 2)})

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IFEval evaluation")
    parser.add_argument("--model", type=str, default="llama-3.2-1b")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    accuracy = evaluate_ifeval(args.model, args.max_samples)
    print(f"\nFinal IFEval accuracy: {accuracy:.1f}%")
