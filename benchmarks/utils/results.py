"""Standardized result saving and loading."""

import json
import os
from datetime import datetime, timezone


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def save_result(result: dict, results_dir: str = RESULTS_DIR):
    """Save a benchmark result as JSON.

    File naming: {model_short}_{quant}_{benchmark}.json
    """
    os.makedirs(results_dir, exist_ok=True)

    filename = f"{result['model_short']}_{result['quant']}_{result['benchmark']}.json"
    path = os.path.join(results_dir, filename)

    result.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved: {path}")
    return path


def load_all_results(results_dir: str = RESULTS_DIR):
    """Load all result JSON files from the results directory."""
    results = []
    if not os.path.exists(results_dir):
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename)) as f:
                results.append(json.load(f))
    return results


def build_result_dict(model_name, model_short, quant, benchmark, score, metric, metrics, config):
    """Construct a standardized result dictionary.

    Args:
        model_name: Full HF model name.
        model_short: Short key (e.g. "llama-3.2-1b").
        quant: Quantization level used.
        benchmark: Benchmark name.
        score: The primary score (accuracy, exact match, etc.).
        metric: Name of the metric (e.g. "accuracy").
        metrics: An InferenceMetrics instance.
        config: Dict of eval config (n_shot, max_new_tokens, etc.).
    """
    return {
        "model": model_name,
        "model_short": model_short,
        "quant": quant,
        "benchmark": benchmark,
        "score": score,
        "metric": metric,
        "num_samples": metrics.samples_evaluated,
        "latency_ms_per_token": round(metrics.latency_ms_per_token, 2),
        "throughput_tokens_per_sec": round(metrics.throughput_tokens_per_sec, 2),
        "peak_memory_mb": round(metrics.peak_memory_mb, 2),
        "config": config,
    }
