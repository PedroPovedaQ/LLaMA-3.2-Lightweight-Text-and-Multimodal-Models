"""MGSM — Multilingual Grade School Math (0-shot CoT, exact match)."""

import re

import torch

from benchmarks.tasks.base import BenchmarkTask


class MGSMTask(BenchmarkTask):
    name = "mgsm"
    n_shot = 0
    metric_name = "exact_match"
    dataset_name = "juletxara/mgsm"
    dataset_config = "en"
    dataset_split = "test"
    max_new_tokens = 256

    def format_prompt(self, example) -> str:
        return (
            f"Solve this math problem step by step, then give the final numeric answer.\n\n"
            f"Question: {example['question']}\n"
            f"Answer:"
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
        pred_num = self._extract_last_number(prediction)
        gold_num = example["answer_number"]

        if pred_num is None:
            return 0.0
        return 1.0 if abs(pred_num - gold_num) < 1e-3 else 0.0

    @staticmethod
    def _extract_last_number(text):
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            return float(numbers[-1].replace(",", ""))
        return None
