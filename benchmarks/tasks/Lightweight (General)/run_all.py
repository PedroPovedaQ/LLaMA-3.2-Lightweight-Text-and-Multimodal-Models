"""
run_all.py — Run All Benchmarks
=================================
Single entry point to evaluate a model on all four benchmarks:
1. MMLU (5-shot, accuracy)
2. Open-rewrite eval (0-shot, ROUGE-L)
3. TLDR9+ (1-shot, ROUGE-L)
4. IFEval (instruction-following accuracy)

Usage:
    python run_all.py --model llama-3.2-1b
    python run_all.py --model llama-3.2-3b --max_samples 20  # smoke test
    python run_all.py --model llama-3.2-1b --benchmarks mmlu ifeval
"""

import argparse
import os
import json
import time
import logging
from datetime import timedelta

from config import (MMLUConfig, OpenRewriteConfig, TLDR9Config,
                    IFEvalConfig, get_model_config)
from eval_mmlu import evaluate_mmlu
from eval_open_rewrite import evaluate_open_rewrite
from eval_tldr import evaluate_tldr
from eval_ifeval import evaluate_ifeval
from utils import setup_logging

logger = logging.getLogger(__name__)

BENCHMARKS = {
    "mmlu": {
        "eval_fn": evaluate_mmlu,
        "get_paper_score": lambda m: MMLUConfig().paper_scores.get(m),
    },
    "open_rewrite": {
        "eval_fn": evaluate_open_rewrite,
        "get_paper_score": lambda m: OpenRewriteConfig().paper_scores.get(m),
    },
    "tldr9": {
        "eval_fn": evaluate_tldr,
        "get_paper_score": lambda m: TLDR9Config().paper_scores.get(m),
    },
    "ifeval": {
        "eval_fn": evaluate_ifeval,
        "get_paper_score": lambda m: IFEvalConfig().paper_scores.get(m),
    },
}


def run_all_benchmarks(model_name: str, max_samples: int = None,
                       benchmarks: list = None):
    """Run all (or selected) benchmarks and produce a summary."""
    model_config = get_model_config(model_name)
    logger.info(f"Model: {model_name} ({model_config.hf_model_id})")

    selected = benchmarks or list(BENCHMARKS.keys())
    logger.info(f"Benchmarks: {selected}")

    results = {}
    paper_scores = {}
    total_start = time.time()

    for bench_name in selected:
        if bench_name not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {bench_name}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"  BENCHMARK: {bench_name}")
        logger.info(f"{'='*70}\n")

        bench_start = time.time()
        bench_info = BENCHMARKS[bench_name]
        paper_scores[bench_name] = bench_info["get_paper_score"](model_name)

        try:
            score = bench_info["eval_fn"](model_name, max_samples)
            results[bench_name] = score
        except Exception as e:
            logger.error(f"Benchmark '{bench_name}' failed: {e}", exc_info=True)
            results[bench_name] = None

        elapsed = time.time() - bench_start
        logger.info(f"  Time: {timedelta(seconds=int(elapsed))}")

    total_elapsed = time.time() - total_start

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"  SUMMARY: {model_name}")
    logger.info(f"{'='*70}")
    logger.info(f"{'Benchmark':<20} {'Ours':>8} {'Paper':>8} {'Diff':>8}")
    logger.info(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for bench_name in selected:
        ours = results.get(bench_name)
        paper = paper_scores.get(bench_name)
        if ours is not None and paper is not None:
            logger.info(f"{bench_name:<20} {ours:>7.1f}% {paper:>7.1f}% "
                        f"{ours - paper:>+7.1f}%")
        elif ours is not None:
            logger.info(f"{bench_name:<20} {ours:>7.1f}%     N/A")
        else:
            logger.info(f"{bench_name:<20} {'FAILED':>8}")

    logger.info(f"\nTotal time: {timedelta(seconds=int(total_elapsed))}")

    # Save summary
    summary = {
        "model": model_name,
        "hf_model_id": model_config.hf_model_id,
        "results": results,
        "paper_scores": paper_scores,
        "max_samples": max_samples,
        "total_time_seconds": int(total_elapsed),
    }

    summary_dir = os.path.join("results", model_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--model", type=str, default="llama-3.2-1b")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per benchmark (use 5-10 for smoke test)")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        choices=list(BENCHMARKS.keys()))
    args = parser.parse_args()

    setup_logging()
    results = run_all_benchmarks(args.model, args.max_samples, args.benchmarks)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for bench, score in results.items():
        print(f"  {bench}: {score:.1f}%" if score is not None
              else f"  {bench}: FAILED")
