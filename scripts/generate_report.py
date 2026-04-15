#!/usr/bin/env python3
"""Generate a PDF snapshot report from recent benchmark and efficiency runs."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "results" / "raw"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

TARGET_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]

DISPLAY_NAME = {
    "meta-llama/Llama-3.2-1B-Instruct": "LLaMA-3.2-1B",
    "meta-llama/Llama-3.2-3B-Instruct": "LLaMA-3.2-3B",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PDF project snapshot report")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "project_snapshot_report.pdf"),
        help="Output PDF path",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_benchmark(model_id: str) -> Optional[Tuple[str, Dict[str, object]]]:
    latest = None
    for path in sorted(glob.glob(str(RAW_DIR / "benchmark_*.json"))):
        data = load_json(path)
        if data.get("run_type") != "benchmark":
            continue
        runtime = data.get("runtime", {})
        config = data.get("config", {})
        if runtime.get("model_id") != model_id:
            continue
        if runtime.get("device") not in {"cuda", "cpu", "mps"}:
            continue
        if runtime.get("precision") != "fp16":
            continue
        bench_list = config.get("benchmarks")
        if (
            not isinstance(bench_list, list)
            or len(bench_list) != 3
            or frozenset(bench_list) != {"hellaswag", "arc", "gsm8k"}
        ):
            continue
        ts = datetime.fromisoformat(str(data.get("started_at_utc")))
        if latest is None or ts > latest[0]:
            latest = (ts, path, data)

    if latest is None:
        return None
    return latest[1], latest[2]


def get_latest_efficiency(model_id: str) -> Optional[Tuple[str, Dict[str, object]]]:
    latest = None
    for path in sorted(glob.glob(str(RAW_DIR / "efficiency_*.json"))):
        data = load_json(path)
        if data.get("run_type") != "efficiency":
            continue
        runtime = data.get("runtime", {})
        config = data.get("config", {})
        if runtime.get("model_id") != model_id:
            continue
        if runtime.get("device") not in {"cuda", "cpu", "mps"}:
            continue
        if runtime.get("precision") != "fp16":
            continue
        if config.get("num_prompts") != 5:
            continue
        if config.get("timed_runs") != 3:
            continue
        if config.get("max_new_tokens") != 64:
            continue
        ts = datetime.fromisoformat(str(data.get("started_at_utc")))
        if latest is None or ts > latest[0]:
            latest = (ts, path, data)

    if latest is None:
        return None
    return latest[1], latest[2]


def benchmark_summary(data: Dict[str, object]) -> Dict[str, float]:
    by_name = {b["name"]: b for b in data.get("benchmarks", [])}
    total_correct = sum(float(b["correct"]) for b in by_name.values())
    total_examples = sum(float(b["num_examples"]) for b in by_name.values())
    total_duration = sum(float(b["duration_sec"]) for b in by_name.values())
    return {
        "hellaswag_acc": float(by_name["hellaswag"]["accuracy"]),
        "arc_acc": float(by_name["arc"]["accuracy"]),
        "gsm8k_acc": float(by_name["gsm8k"]["accuracy"]),
        "overall_acc": total_correct / total_examples if total_examples else 0.0,
        "total_duration": total_duration,
    }


def efficiency_summary(data: Dict[str, object]) -> Dict[str, float]:
    metrics = data.get("metrics", {})
    memory = data.get("memory", {})
    return {
        "ttft_ms": float(metrics["ttft_ms"]["mean"]),
        "latency_ms_per_token": float(metrics["latency_ms_per_token"]["mean"]),
        "throughput_tok_s": float(metrics["throughput_tok_s"]["mean"]),
        "ram_after_run_gb": float(memory["ram_after_run_gb"]),
    }


def add_title_page(pdf: PdfPages, bench_files: Dict[str, str], eff_files: Dict[str, str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.clf()
    text = [
        "LLaMA 3.2 Project Snapshot Report",
        "",
        "Scope:",
        "- Benchmark comparison (HellaSwag, ARC, GSM8K)",
        "- Efficiency comparison (TTFT, latency/token, throughput, RAM)",
        "",
        "Data Sources (latest matching runs):",
    ]

    for model_id in TARGET_MODELS:
        text.append(
            f"- {DISPLAY_NAME[model_id]} benchmark: {Path(bench_files[model_id]).name}"
        )
        text.append(
            f"- {DISPLAY_NAME[model_id]} efficiency: {Path(eff_files[model_id]).name}"
        )

    text.append("")
    text.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")

    fig.text(0.05, 0.95, "\n".join(text), va="top", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def add_accuracy_page(pdf: PdfPages, bench: Dict[str, Dict[str, float]]) -> None:
    labels = [DISPLAY_NAME[m] for m in TARGET_MODELS]
    x = range(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Benchmark Accuracy and Runtime (fp16)")

    metrics = [
        ("hellaswag_acc", "HellaSwag Accuracy"),
        ("arc_acc", "ARC Accuracy"),
        ("gsm8k_acc", "GSM8K Accuracy"),
        ("total_duration", "Total Benchmark Duration (sec)"),
    ]

    for ax, (key, title) in zip(axes.flatten(), metrics):
        values = [bench[m][key] for m in TARGET_MODELS]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def add_efficiency_page(pdf: PdfPages, eff: Dict[str, Dict[str, float]]) -> None:
    labels = [DISPLAY_NAME[m] for m in TARGET_MODELS]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        "Efficiency Metrics (num_prompts=5, timed_runs=3, max_new_tokens=64, fp16)"
    )

    metrics = [
        ("ttft_ms", "TTFT Mean (ms)"),
        ("latency_ms_per_token", "Latency per Token Mean (ms)"),
        ("throughput_tok_s", "Throughput Mean (tok/s)"),
        ("ram_after_run_gb", "RAM After Run (GB)"),
    ]

    for ax, (key, title) in zip(axes.flatten(), metrics):
        values = [eff[m][key] for m in TARGET_MODELS]
        ax.bar(labels, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def add_interpretation_page(
    pdf: PdfPages,
    bench: Dict[str, Dict[str, float]],
    eff: Dict[str, Dict[str, float]],
) -> None:
    rank_acc = sorted(TARGET_MODELS, key=lambda m: bench[m]["overall_acc"], reverse=True)
    rank_speed = sorted(TARGET_MODELS, key=lambda m: bench[m]["total_duration"])  # lower is better

    fig = plt.figure(figsize=(11, 8.5))
    fig.clf()
    lines = [
        "Interpretation Snapshot",
        "",
        "Overall Accuracy Ranking:",
    ]
    for idx, model in enumerate(rank_acc, start=1):
        lines.append(f"{idx}. {DISPLAY_NAME[model]} ({bench[model]['overall_acc']:.3f})")

    lines.extend([
        "",
        "Benchmark Runtime Ranking (faster is better):",
    ])
    for idx, model in enumerate(rank_speed, start=1):
        lines.append(f"{idx}. {DISPLAY_NAME[model]} ({bench[model]['total_duration']:.1f}s)")

    lines.extend([
        "",
        "Efficiency Highlights:",
        f"- Fastest TTFT: {DISPLAY_NAME[min(TARGET_MODELS, key=lambda m: eff[m]['ttft_ms'])]}",
        f"- Best throughput: {DISPLAY_NAME[max(TARGET_MODELS, key=lambda m: eff[m]['throughput_tok_s'])]}",
        f"- Lowest latency/token: {DISPLAY_NAME[min(TARGET_MODELS, key=lambda m: eff[m]['latency_ms_per_token'])]}",
        "",
        "Caveat:",
        "- Benchmark sample size follows each run's configured limit; use larger limits for final claims.",
    ])

    fig.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_data: Dict[str, Dict[str, float]] = {}
    efficiency_data: Dict[str, Dict[str, float]] = {}
    benchmark_files: Dict[str, str] = {}
    efficiency_files: Dict[str, str] = {}

    missing = []
    for model_id in TARGET_MODELS:
        bench = get_latest_benchmark(model_id)
        eff = get_latest_efficiency(model_id)
        if bench is None:
            missing.append(f"Missing benchmark run for {model_id}")
        if eff is None:
            missing.append(f"Missing efficiency run for {model_id}")
        if bench is None or eff is None:
            continue

        bench_path, bench_json = bench
        eff_path, eff_json = eff
        benchmark_data[model_id] = benchmark_summary(bench_json)
        efficiency_data[model_id] = efficiency_summary(eff_json)
        benchmark_files[model_id] = bench_path
        efficiency_files[model_id] = eff_path

    if missing:
        raise RuntimeError("Cannot generate report:\n- " + "\n- ".join(missing))

    with PdfPages(output_path) as pdf:
        add_title_page(pdf, benchmark_files, efficiency_files)
        add_accuracy_page(pdf, benchmark_data)
        add_efficiency_page(pdf, efficiency_data)
        add_interpretation_page(pdf, benchmark_data, efficiency_data)

    print(f"[report] Wrote PDF report to: {output_path}")


if __name__ == "__main__":
    main()
