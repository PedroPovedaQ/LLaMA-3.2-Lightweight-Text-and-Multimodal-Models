"""
compare_results.py — Model Comparison & Reporting
===================================================
Generates side-by-side tables, LaTeX output, and bar charts.

Usage:
    python compare_results.py \
        --results results/llama-3.2-1b results/llama-3.2-3b \
                  results/phi-3.5-mini results/gemma-2-2b
"""

import argparse
import os
import json
import logging

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MMLUConfig, OpenRewriteConfig, TLDR9Config, IFEvalConfig

logger = logging.getLogger(__name__)

BENCH_LABELS = {
    "mmlu": "MMLU (5-shot)",
    "open_rewrite": "Open-rewrite (rougeL)",
    "tldr9": "TLDR9+ (rougeL)",
    "ifeval": "IFEval",
}


def load_model_summary(results_dir):
    """
    Load a model's results from its results directory.

    Reads individual benchmark JSON files for accurate scores,
    falling back to summary.json if available.
    """
    model_name = os.path.basename(results_dir)

    # Start with summary.json if it exists
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"model": model_name, "results": {}, "paper_scores": {}}

    # Override with actual scores from individual benchmark files
    bench_files = {
        "mmlu": "mmlu.json",
        "open_rewrite": "open_rewrite.json",
        "tldr9": "tldr9.json",
        "ifeval": "ifeval.json",
    }

    for bench_name, filename in bench_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                bench_data = json.load(f)
            score = bench_data.get("metadata", {}).get("score", None)
            if score is not None:
                summary["results"][bench_name] = score
            paper = bench_data.get("metadata", {}).get("paper_score", None)
            if paper is not None:
                summary["paper_scores"][bench_name] = paper

    if not summary["results"]:
        raise FileNotFoundError(f"No results found in {results_dir}")

    return summary


def build_table(summaries):
    benchmarks = list(BENCH_LABELS.keys())
    data = {"Benchmark": [BENCH_LABELS[b] for b in benchmarks]}

    for summary in summaries:
        name = summary["model"]
        scores = summary.get("results", {})
        paper = summary.get("paper_scores", {})
        data[name] = [scores.get(b) for b in benchmarks]
        data[f"{name} (paper)"] = [paper.get(b) for b in benchmarks]

    return pd.DataFrame(data)


def plot_comparison(summaries, output_path):
    benchmarks = list(BENCH_LABELS.keys())
    labels = [BENCH_LABELS[b] for b in benchmarks]

    n_models = len(summaries)
    bar_width = 0.8 / (n_models + 1)
    x = np.arange(len(benchmarks))

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    for i, summary in enumerate(summaries):
        name = summary["model"]
        scores = summary.get("results", {})
        vals = [scores.get(b, 0) or 0 for b in benchmarks]

        offset = -bar_width * (n_models - 1) / 2 + bar_width * i
        bars = ax.bar(x + offset, vals, bar_width, label=name,
                      color=colors[i % len(colors)], edgecolor="black",
                      linewidth=0.5)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Lightweight Instruction-Tuned Model Benchmarks",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Chart saved: {output_path}")
    plt.close()


def main(results_dirs, output_dir="comparison_output"):
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    for rdir in results_dirs:
        try:
            summaries.append(load_model_summary(rdir))
            logger.info(f"Loaded: {summaries[-1]['model']}")
        except FileNotFoundError as e:
            logger.error(str(e))

    if not summaries:
        logger.error("No results found.")
        return

    df = build_table(summaries)

    # Print to console
    from tabulate import tabulate
    print("\n" + "="*70)
    print("  BENCHMARK COMPARISON")
    print("="*70)
    print(tabulate(df, headers="keys", tablefmt="grid",
                   floatfmt=".1f", showindex=False))

    # Save CSV
    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)

    # Save LaTeX
    latex_path = os.path.join(output_dir, "comparison_table.tex")
    df.to_latex(latex_path, index=False, float_format="%.1f", na_rep="—")

    # Save chart
    plot_path = os.path.join(output_dir, "comparison_chart.png")
    plot_comparison(summaries, plot_path)

    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="comparison_output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.results, args.output_dir)
