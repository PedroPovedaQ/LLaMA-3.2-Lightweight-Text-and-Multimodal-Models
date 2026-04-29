#!/usr/bin/env python3
"""Generate visual artifacts for a single benchmark run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from experiment_utils import REPORTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG/PDF artifacts for one benchmark run")
    parser.add_argument("--input", type=str, required=True, help="Path to raw benchmark JSON")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for generated files (defaults to results/reports/<run_id>)",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_display_name(model_id: str) -> str:
    normalized = model_id.rstrip("/")
    if normalized.startswith("./") or normalized.startswith("../"):
        return Path(normalized).name
    if "/" in normalized:
        return normalized.split("/")[-1]
    return normalized


def generate_benchmark_report(
    run_summary: Dict[str, object],
    output_prefix: Path,
) -> Dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    runtime = run_summary.get("runtime", {})
    benchmarks = run_summary.get("benchmarks", [])
    config = run_summary.get("config", {})

    model_id = str(runtime.get("model_id", "unknown-model"))
    model_name = _model_display_name(model_id)
    device = str(runtime.get("device", "unknown"))
    precision = str(runtime.get("precision", "unknown"))
    limit = config.get("limit", "unknown")

    benchmark_names = [str(item["name"]) for item in benchmarks]
    accuracies = [float(item["accuracy"]) for item in benchmarks]
    durations = [float(item["duration_sec"]) for item in benchmarks]

    accuracy_png = output_prefix.with_name(output_prefix.name + "_accuracy.png")
    runtime_png = output_prefix.with_name(output_prefix.name + "_runtime.png")
    report_pdf = output_prefix.with_name(output_prefix.name + "_report.pdf")

    colors = ["#4C78A8", "#2E5C95", "#E68613", "#72B7B2", "#54A24B", "#EECA3B"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(benchmark_names, accuracies, color=colors[: len(benchmark_names)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} Benchmark Accuracy")
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(accuracies):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(accuracy_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(benchmark_names, durations, color=colors[: len(benchmark_names)])
    ax.set_ylabel("Seconds")
    ax.set_title(f"{model_name} Benchmark Runtime")
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(durations):
        ax.text(i, value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(runtime_png, dpi=180)
    plt.close(fig)

    with PdfPages(report_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.clf()
        lines = [
            "Benchmark Run Summary",
            "",
            f"Model: {model_id}",
            f"Display name: {model_name}",
            f"Device: {device}",
            f"Precision: {precision}",
            f"Limit per benchmark: {limit}",
            f"Run ID: {run_summary.get('run_id', 'unknown')}",
            f"Generated at: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Results:",
        ]
        for item in benchmarks:
            lines.append(
                f"- {item['name']}: accuracy={float(item['accuracy']):.3f} "
                f"({item['correct']}/{item['num_examples']}), duration={float(item['duration_sec']):.2f}s"
            )
        fig.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        for image_path, title in [
            (accuracy_png, "Accuracy"),
            (runtime_png, "Runtime"),
        ]:
            fig = plt.figure(figsize=(11, 8.5))
            fig.clf()
            fig.suptitle(f"{model_name} {title}")
            image = plt.imread(image_path)
            ax = fig.add_axes([0.05, 0.08, 0.9, 0.84])
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    return {
        "accuracy_png": accuracy_png,
        "runtime_png": runtime_png,
        "report_pdf": report_pdf,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    run_summary = _load_json(input_path)
    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else REPORTS_DIR / str(run_summary.get("run_id", input_path.stem))
    )
    artifacts = generate_benchmark_report(run_summary, output_prefix)
    for key, path in artifacts.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
