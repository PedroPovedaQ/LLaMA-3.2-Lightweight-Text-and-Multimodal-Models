"""
utils.py — Shared Utilities
============================
Answer extraction, ROUGE-L scoring, result saving, and logging.
"""

import re
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List


# =============================================================================
# 1. ANSWER EXTRACTION (for MMLU multiple-choice)
# =============================================================================

def extract_answer_letter(response: str) -> Optional[str]:
    """
    Extract a single answer letter (A-D) from a model's response.

    Multiple fallback patterns, in order of confidence:
    1. Explicit "The answer is X" or "Answer: X"
    2. Standalone letter at the start of the response
    3. First A-D letter found in the response
    """
    if not response:
        return None

    text = response.strip()

    # Pattern 1: "the answer is X" or "answer: X"
    match = re.search(r"[Aa]nswer[:\s]+(?:is\s+)?([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    # Pattern 2: response starts with a letter (common for completion-style)
    match = re.match(r"^([A-Da-d])[)\.\s,]", text)
    if match:
        return match.group(1).upper()

    # Pattern 3: just a single letter
    match = re.fullmatch(r"([A-Da-d])", text.split()[0] if text.split() else "")
    if match:
        return match.group(1).upper()

    # Pattern 4: first capital A-D found
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)

    return None


# =============================================================================
# 2. ROUGE-L SCORING
# =============================================================================

_rouge_scorer = None

def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between prediction and reference.
    Returns score on 0-100 scale to match paper's reporting.
    """
    global _rouge_scorer

    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)

    if not prediction or not reference:
        return 0.0

    scores = _rouge_scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure * 100.0


# =============================================================================
# 3. RESULT I/O
# =============================================================================

def save_results(results: List[Dict[str, Any]], model_name: str,
                 benchmark_name: str, results_dir: str = "results",
                 metadata: Optional[Dict] = None):
    """
    Save evaluation results to a JSON file.
    Output: results/<model_name>/<benchmark_name>.json
    """
    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Compute score: use ROUGE-L average for ROUGE benchmarks, accuracy otherwise
    if any("rouge_l" in r for r in results):
        scores = [r.get("rouge_l", 0.0) for r in results]
        score = sum(scores) / len(scores) if scores else 0.0
    else:
        correct_count = sum(1 for r in results if r.get("correct", False))
        total_count = len(results)
        score = 100.0 * correct_count / total_count if total_count > 0 else 0.0

    output = {
        "metadata": {
            "model": model_name,
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(results),
            "score": round(score, 2),
            **(metadata or {}),
        },
        "results": results,
    }

    filepath = os.path.join(out_dir, f"{benchmark_name}.json")
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logging.getLogger(__name__).info(
        f"Results saved: {filepath} (score={score:.1f}%)"
    )
    return filepath


def load_results(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# 4. LOGGING
# =============================================================================

def setup_logging(level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/eval_{timestamp}.log"),
        ],
    )
    return logging.getLogger(__name__)
