"""Nexus — Tool Use Evaluation (accuracy)."""

import torch

from benchmarks.tasks.base import BenchmarkTask


class NexusTask(BenchmarkTask):
    name = "nexus"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "Nexusflow/NexusRaven_API_evaluation"  # TODO: Confirm exact dataset
    dataset_split = "test"
    max_new_tokens = 512

    def format_prompt(self, example) -> str:
        # TODO: Implement with tool/API definitions and user query.
        raise NotImplementedError(
            "Nexus requires tool-use prompt formatting. "
            "See: https://nexusflow.ai/blogs/nexusraven"
        )

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {"prediction": prediction_text, "tokens_generated": generated_ids.shape[1]}

    def score(self, prediction, example) -> float:
        # TODO: Compare predicted API call against ground truth.
        raise NotImplementedError("Nexus scoring requires API call comparison.")
