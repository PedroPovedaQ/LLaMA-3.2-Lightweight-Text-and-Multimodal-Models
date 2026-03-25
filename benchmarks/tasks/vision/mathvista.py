"""MathVista — Mathematical Reasoning with Visual Context (testmini, accuracy)."""

import re

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class MathVistaTask(VisionBenchmarkTask):
    name = "mathvista"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "AI4Math/MathVista"
    dataset_config = None
    dataset_split = "testmini"

    def format_prompt(self, example) -> str:
        question = example["question"]
        choices = example.get("choices", [])

        prompt = f"<|image|>\n\n{question}\n"
        if choices:
            for i, c in enumerate(choices):
                prompt += f"{chr(65 + i)}. {c}\n"
            prompt += "Answer with a single letter."
        else:
            prompt += "Give the final numeric answer."
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        gold = str(example["answer"]).strip().lower()
        pred = prediction.strip().lower()

        # Multiple choice
        if example.get("choices"):
            for char in pred.upper():
                if char in "ABCDEFGHIJ":
                    gold_letter = chr(65 + int(gold)) if gold.isdigit() else gold.upper()
                    return 1.0 if char == gold_letter else 0.0
            return 0.0

        # Numeric
        pred_nums = re.findall(r"-?[\d,]+\.?\d*", pred)
        gold_nums = re.findall(r"-?[\d,]+\.?\d*", gold)
        if pred_nums and gold_nums:
            try:
                return 1.0 if abs(float(pred_nums[-1].replace(",", "")) - float(gold_nums[-1].replace(",", ""))) < 1e-3 else 0.0
            except ValueError:
                return 0.0
        return 1.0 if gold in pred else 0.0
