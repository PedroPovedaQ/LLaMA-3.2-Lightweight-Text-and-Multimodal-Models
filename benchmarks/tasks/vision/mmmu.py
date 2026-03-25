"""MMMU — Massive Multi-discipline Multimodal Understanding (0-shot CoT, accuracy)."""

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class MMMUTask(VisionBenchmarkTask):
    name = "mmmu"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "MMMU/MMMU"
    dataset_config = "validation"  # test labels are hidden
    dataset_split = "validation"

    def format_prompt(self, example) -> str:
        question = example["question"]
        options = example.get("options", [])

        prompt = f"<|image|>\n\n{question}\n"
        if options:
            for i, opt in enumerate(options):
                label = chr(65 + i)  # A, B, C, D...
                prompt += f"{label}. {opt}\n"
        prompt += "\nThink step by step, then give your final answer as a single letter."
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        gold = example["answer"].strip().upper()
        # Extract first capital letter from prediction
        for char in prediction.strip().upper():
            if char in "ABCDEFGHIJ":
                return 1.0 if char == gold else 0.0
        return 0.0
