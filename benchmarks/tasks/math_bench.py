"""MATH — Mathematical Problem Solving (0-shot CoT, exact match)."""

import re

import torch

from benchmarks.tasks.base import BenchmarkTask


class MATHTask(BenchmarkTask):
    name = "math"
    n_shot = 0
    metric_name = "exact_match"
    dataset_name = "hendrycks/competition_math"
    dataset_split = "test"
    max_new_tokens = 512

    def format_prompt(self, example) -> str:
        return (
            f"Solve the following math problem. Show your work step by step, "
            f"then give the final answer boxed.\n\n"
            f"Problem: {example['problem']}\n"
            f"Solution:"
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
        pred_answer = self._extract_boxed(prediction)
        gold_answer = self._extract_boxed(example["solution"])

        if pred_answer is None or gold_answer is None:
            return 0.0
        return 1.0 if pred_answer.strip() == gold_answer.strip() else 0.0

    @staticmethod
    def _extract_boxed(text):
        """Extract content from \\boxed{...}."""
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        return match.group(1) if match else None
