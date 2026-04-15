"""Base class for vision benchmark tasks.

Vision tasks use multimodal models (e.g. LLaMA 3.2-Vision) that accept
both image and text inputs. The model must be loaded with a processor
instead of a plain tokenizer.
"""

from abc import abstractmethod

import torch
from datasets import load_dataset

from benchmarks.utils.metrics import InferenceMetrics, track_latency, reset_peak_gpu_memory, get_peak_gpu_memory_mb


class VisionBenchmarkTask:
    """Base class for vision benchmarks."""

    name: str = ""
    n_shot: int = 0
    metric_name: str = "accuracy"
    dataset_name: str = ""
    dataset_config: str = None
    dataset_split: str = "test"

    def __init__(self, max_samples=None):
        self.max_samples = max_samples

    def load_data(self):
        kwargs = {"name": self.dataset_config} if self.dataset_config else {}
        dataset = load_dataset(self.dataset_name, split=self.dataset_split, **kwargs)

        if self.max_samples and self.max_samples < len(dataset):
            dataset = dataset.select(range(self.max_samples))

        return dataset

    @abstractmethod
    def format_prompt(self, example) -> str:
        """Build the text portion of the prompt."""
        ...

    @abstractmethod
    def get_images(self, example) -> list:
        """Extract PIL image(s) from the example."""
        ...

    @abstractmethod
    def score(self, prediction, example) -> float:
        """Score a single prediction."""
        ...

    def evaluate_sample(self, model, processor, example) -> dict:
        """Run inference on a single vision sample.

        Args:
            model: A multimodal model (e.g. MllamaForConditionalGeneration).
            processor: The matching processor (handles both image + text).
            example: A single dataset example.

        Returns:
            {"prediction": str, "tokens_generated": int}
        """
        prompt = self.format_prompt(example)
        images = self.get_images(example)

        inputs = processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        prediction_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        return {"prediction": prediction_text, "tokens_generated": generated_ids.shape[1]}

    def run(self, model, processor) -> tuple[float, InferenceMetrics]:
        """Run the full vision benchmark."""
        dataset = self.load_data()
        metrics = InferenceMetrics()
        reset_peak_gpu_memory()

        correct = 0
        total = 0

        for example in dataset:
            with track_latency() as t:
                result = self.evaluate_sample(model, processor, example)

            sample_score = self.score(result["prediction"], example)
            correct += sample_score
            total += 1

            metrics.total_tokens_generated += result.get("tokens_generated", 0)
            metrics.total_time_s += t["elapsed_s"]

        metrics.samples_evaluated = total
        metrics.peak_memory_mb = get_peak_gpu_memory_mb()

        aggregate_score = correct / total if total > 0 else 0.0
        return aggregate_score, metrics
