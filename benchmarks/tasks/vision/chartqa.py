"""ChartQA — Chart and Diagram Understanding (0-shot CoT, relaxed accuracy)."""

import re

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class ChartQATask(VisionBenchmarkTask):
    name = "chartqa"
    n_shot = 0
    metric_name = "relaxed_accuracy"
    dataset_name = "HuggingFaceM4/ChartQA"
    dataset_config = None
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        question = example["query"]
        prompt = (
            f"<|image|>\n\n"
            f"Look at the chart and answer the following question.\n"
            f"Question: {question}\n"
            f"Think step by step, then give a short final answer."
        )
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        """Relaxed accuracy: correct if within 5% of the gold answer (for numbers)."""
        gold = str(example["label"]).strip().lower()
        pred = prediction.strip().lower()

        # Try numeric comparison with 5% tolerance
        try:
            gold_num = float(re.sub(r"[,%$]", "", gold))
            pred_nums = re.findall(r"-?[\d,]+\.?\d*", pred)
            if pred_nums:
                pred_num = float(pred_nums[-1].replace(",", ""))
                tolerance = abs(gold_num) * 0.05
                return 1.0 if abs(pred_num - gold_num) <= tolerance else 0.0
        except ValueError:
            pass

        # Fallback: exact string match
        return 1.0 if gold in pred else 0.0
