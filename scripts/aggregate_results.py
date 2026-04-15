#!/usr/bin/env python3
"""Aggregate raw experiment JSON files into normalized CSV metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from experiment_utils import PROCESSED_RESULTS_DIR, PROJECT_ROOT, RAW_RESULTS_DIR, csv_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw results into CSV")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_RESULTS_DIR),
        help="Directory containing benchmark_*.json and efficiency_*.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_RESULTS_DIR / "metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip automatic PDF report generation after CSV aggregation",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "reports" / "project_snapshot_report.pdf"),
        help="Output path passed to scripts/generate_report.py",
    )
    parser.add_argument(
        "--strict-report",
        action="store_true",
        help="Fail if report generation fails",
    )
    return parser.parse_args()


def load_json_files(input_dir: Path) -> List[Dict[str, object]]:
    payloads = []
    for path in sorted(input_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data["_source_file"] = str(path)
            payloads.append(data)
    return payloads


def row_base(run: Dict[str, object]) -> Dict[str, object]:
    runtime = run.get("runtime", {}) if isinstance(run.get("runtime"), dict) else {}
    return {
        "run_type": run.get("run_type"),
        "run_id": run.get("run_id"),
        "started_at_utc": run.get("started_at_utc"),
        "ended_at_utc": run.get("ended_at_utc"),
        "source_file": run.get("_source_file"),
        "model_id": runtime.get("model_id"),
        "precision": runtime.get("precision"),
        "device": runtime.get("device"),
    }


def rows_from_benchmark(run: Dict[str, object]) -> Iterable[Dict[str, object]]:
    base = row_base(run)
    entries = run.get("benchmarks", [])
    if not isinstance(entries, list):
        return

    for benchmark in entries:
        if not isinstance(benchmark, dict):
            continue

        benchmark_name = benchmark.get("name")
        num_examples = benchmark.get("num_examples")
        for metric_name, unit in [
            ("accuracy", "ratio"),
            ("correct", "count"),
            ("duration_sec", "sec"),
            ("examples_per_sec", "examples_per_sec"),
        ]:
            if metric_name not in benchmark:
                continue
            row = dict(base)
            row.update(
                {
                    "benchmark": benchmark_name,
                    "metric": metric_name,
                    "value": benchmark.get(metric_name),
                    "unit": unit,
                    "num_examples": num_examples,
                }
            )
            yield row


def rows_from_efficiency(run: Dict[str, object]) -> Iterable[Dict[str, object]]:
    base = row_base(run)

    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    memory = run.get("memory", {}) if isinstance(run.get("memory"), dict) else {}

    scalar_metric_specs = [
        ("ttft_ms", "ms"),
        ("latency_ms_per_token", "ms_per_token"),
        ("throughput_tok_s", "tokens_per_sec"),
    ]

    for metric_name, unit in scalar_metric_specs:
        metric_obj = metrics.get(metric_name)
        if not isinstance(metric_obj, dict):
            continue
        for field in ["mean", "std"]:
            if field not in metric_obj:
                continue
            row = dict(base)
            row.update(
                {
                    "benchmark": "efficiency",
                    "metric": f"{metric_name}_{field}",
                    "value": metric_obj.get(field),
                    "unit": unit,
                    "num_examples": run.get("config", {}).get("num_prompts"),
                }
            )
            yield row

    for metric_name, unit in [
        ("ram_before_load_gb", "gb"),
        ("ram_after_load_gb", "gb"),
        ("ram_after_run_gb", "gb"),
        ("gpu_peak_gb", "gb"),
    ]:
        if metric_name not in memory:
            continue
        row = dict(base)
        row.update(
            {
                "benchmark": "efficiency",
                "metric": metric_name,
                "value": memory.get(metric_name),
                "unit": unit,
                "num_examples": run.get("config", {}).get("num_prompts"),
            }
        )
        yield row


def to_rows(run: Dict[str, object]) -> Iterable[Dict[str, object]]:
    run_type = run.get("run_type")
    if run_type == "benchmark":
        yield from rows_from_benchmark(run)
    elif run_type == "efficiency":
        yield from rows_from_efficiency(run)


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "run_type",
        "run_id",
        "started_at_utc",
        "ended_at_utc",
        "source_file",
        "model_id",
        "precision",
        "device",
        "benchmark",
        "metric",
        "value",
        "unit",
        "num_examples",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_safe(row.get(k)) for k in fieldnames})


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    payloads = load_json_files(input_dir)
    rows: List[Dict[str, object]] = []
    for payload in payloads:
        rows.extend(list(to_rows(payload)))

    write_csv(rows, output_path)
    print(f"[aggregate] Loaded {len(payloads)} raw files")
    print(f"[aggregate] Wrote {len(rows)} rows to {output_path}")

    if args.skip_report:
        return

    report_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_report.py"),
        "--output",
        args.report_output,
    ]
    print(f"[aggregate] Generating PDF report: {args.report_output}")
    proc = subprocess.run(report_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        if proc.stdout.strip():
            print(proc.stdout.strip())
        print("[aggregate] Report generation complete")
        return

    print("[aggregate] Report generation failed")
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if args.strict_report:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
