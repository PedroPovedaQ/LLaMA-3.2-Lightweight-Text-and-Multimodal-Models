"""AI2 Diagram — Science Diagram Understanding (test, accuracy)."""

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class AI2DiagramTask(VisionBenchmarkTask):
    name = "ai2_diagram"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "derek-thomas/ScienceQA"  # TODO: Confirm exact AI2D dataset on HF
    dataset_config = None
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        question = example["question"]
        choices = example.get("choices", [])

        prompt = f"<|image|>\n\nLook at the diagram and answer the question.\n"
        prompt += f"Question: {question}\n"
        for i, c in enumerate(choices):
            prompt += f"{chr(65 + i)}. {c}\n"
        prompt += "Answer with a single letter."
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        gold_idx = example["answer"]
        gold_letter = chr(65 + gold_idx) if isinstance(gold_idx, int) else gold_idx.strip().upper()

        for char in prediction.strip().upper():
            if char in "ABCDEFGHIJ":
                return 1.0 if char == gold_letter else 0.0
        return 0.0
