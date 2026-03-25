"""GSM8K — Grade School Math (8-shot chain-of-thought, exact match)."""

import re

import torch

from benchmarks.tasks.base import BenchmarkTask


class GSM8KTask(BenchmarkTask):
    name = "gsm8k"
    n_shot = 8
    metric_name = "exact_match"
    dataset_name = "openai/gsm8k"
    dataset_config = "main"
    dataset_split = "test"
    max_new_tokens = 256

    def format_prompt(self, example) -> str:
        """Build an 8-shot CoT prompt for a GSM8K problem.

        Each example has: question, answer (with chain-of-thought ending in #### <number>).
        """
        prompt = ""

        for fs in self.few_shot_examples:
            prompt += f"Question: {fs['question']}\nAnswer: {fs['answer']}\n\n"

        prompt += f"Question: {example['question']}\nAnswer:"
        return prompt

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

        # Decode only the newly generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        tokens_generated = generated_ids.shape[1]

        return {"prediction": prediction_text, "tokens_generated": tokens_generated}

    def score(self, prediction, example) -> float:
        predicted_num = self._extract_number(prediction)
        gold_num = self._extract_number(example["answer"])

        if predicted_num is None or gold_num is None:
            return 0.0
        return 1.0 if abs(predicted_num - gold_num) < 1e-3 else 0.0

    @staticmethod
    def _extract_number(text):
        """Extract the final number from GSM8K format (#### 42) or last number in text."""
        # Try the #### format first
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", ""))

        # Fallback: last number in the text
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            return float(numbers[-1].replace(",", ""))

        return None
