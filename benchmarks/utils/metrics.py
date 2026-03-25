"""Latency, throughput, and memory tracking utilities."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class InferenceMetrics:
    """Collected metrics for a single benchmark run."""

    total_tokens_generated: int = 0
    total_time_s: float = 0.0
    peak_memory_mb: float = 0.0
    samples_evaluated: int = 0

    @property
    def latency_ms_per_token(self):
        if self.total_tokens_generated == 0:
            return 0.0
        return (self.total_time_s / self.total_tokens_generated) * 1000

    @property
    def throughput_tokens_per_sec(self):
        if self.total_time_s == 0.0:
            return 0.0
        return self.total_tokens_generated / self.total_time_s


@contextmanager
def track_latency():
    """Context manager that yields a dict and fills in elapsed_s on exit.

    Usage:
        with track_latency() as t:
            output = model.generate(...)
        print(t["elapsed_s"])
    """
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_s"] = time.perf_counter() - start


def get_peak_gpu_memory_mb():
    """Return peak GPU memory allocated in MB. Returns 0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_peak_gpu_memory():
    """Reset the peak memory tracker so the next run starts clean."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
