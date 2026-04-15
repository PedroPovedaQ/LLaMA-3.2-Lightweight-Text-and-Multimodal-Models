"""Base class for all benchmark tasks."""

from abc import ABC, abstractmethod

import torch
from datasets import load_dataset

from benchmarks.utils.metrics import InferenceMetrics, track_latency, reset_peak_gpu_memory, get_peak_gpu_memory_mb


class BenchmarkTask(ABC):
    """Base class that every benchmark must implement."""

    name: str = ""
    n_shot: int = 0
    metric_name: str = "accuracy"
    dataset_name: str = ""
    dataset_config: str = None
    dataset_split: str = "test"

    def __init__(self, max_samples=None):
        self.max_samples = max_samples
        self.few_shot_examples = []

    def load_data(self):
        """Load the dataset and prepare few-shot examples."""
        kwargs = {"name": self.dataset_config} if self.dataset_config else {}
        dataset = load_dataset(self.dataset_name, split=self.dataset_split, **kwargs)

        if self.n_shot > 0:
            train = load_dataset(self.dataset_name, split="train", **kwargs)
            self.few_shot_examples = list(train.select(range(self.n_shot)))

        if self.max_samples and self.max_samples < len(dataset):
            dataset = dataset.select(range(self.max_samples))

        return dataset

    @abstractmethod
    def format_prompt(self, example) -> str:
        """Build the full prompt including few-shot examples for a single eval example."""
        ...

    @abstractmethod
    def score(self, prediction, example) -> float:
        """Score a single prediction. Return 1.0 for correct, 0.0 for wrong."""
        ...

    @abstractmethod
    def evaluate_sample(self, model, tokenizer, example) -> dict:
        """Run inference on a single sample and return prediction + metadata.

        Should return a dict with at least:
            {"prediction": ..., "tokens_generated": int}
        """
        ...

    def run(self, model, tokenizer) -> tuple[float, InferenceMetrics]:
        """Run the full benchmark. Returns (score, metrics).

        This is the main entry point called by the runner. It loads data,
        iterates over samples, collects scores and timing, and returns
        the aggregate score plus inference metrics.
        """
        dataset = self.load_data()
        metrics = InferenceMetrics()
        reset_peak_gpu_memory()

        correct = 0
        total = 0

        for example in dataset:
            with track_latency() as t:
                result = self.evaluate_sample(model, tokenizer, example)

            sample_score = self.score(result["prediction"], example)
            correct += sample_score
            total += 1

            metrics.total_tokens_generated += result.get("tokens_generated", 0)
            metrics.total_time_s += t["elapsed_s"]

        metrics.samples_evaluated = total
        metrics.peak_memory_mb = get_peak_gpu_memory_mb()

        aggregate_score = correct / total if total > 0 else 0.0
        return aggregate_score, metrics


# ---------------------------------------------------------------------------
# Helpers shared across multiple-choice benchmarks (MMLU, HellaSwag, ARC)
# ---------------------------------------------------------------------------

ANSWER_CHOICES = ["A", "B", "C", "D"]


def score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices):
    """Score each candidate answer by log-likelihood and return the best index.

    Args:
        model: HF causal LM.
        tokenizer: Matching tokenizer.
        prompt: The prompt text (everything before the answer).
        choices: List of answer strings to score.

    Returns:
        (best_index, tokens_evaluated) — index into choices, and total tokens processed.
    """
    log_likelihoods = []
    total_tokens = 0

    for choice in choices:
        full_text = prompt + choice
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        answer_start = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Sum log-probs of answer tokens only
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        answer_ids = input_ids[:, answer_start:]
        token_log_probs = log_probs[:, answer_start - 1:, :]
        selected = torch.gather(token_log_probs, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        log_likelihoods.append(selected.sum().item())
        total_tokens += input_ids.shape[1]

    best_idx = log_likelihoods.index(max(log_likelihoods))
    return best_idx, total_tokens
