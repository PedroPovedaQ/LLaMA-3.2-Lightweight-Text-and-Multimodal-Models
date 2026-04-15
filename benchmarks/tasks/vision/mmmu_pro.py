"""MMMU-Pro — Standard (10-option text) and Vision variants."""

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class MMMUProStandardTask(VisionBenchmarkTask):
    """MMMU-Pro Standard: 10-option multiple choice with text."""

    name = "mmmu_pro_standard"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "MMMU/MMMU_Pro"  # TODO: Confirm exact HF dataset path
    dataset_config = "standard"
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        question = example["question"]
        options = example.get("options", [])

        prompt = f"<|image|>\n\n{question}\n"
        for i, opt in enumerate(options):
            label = chr(65 + i)
            prompt += f"{label}. {opt}\n"
        prompt += "\nAnswer with a single letter."
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        gold = example["answer"].strip().upper()
        for char in prediction.strip().upper():
            if char in "ABCDEFGHIJ":
                return 1.0 if char == gold else 0.0
        return 0.0


class MMMUProVisionTask(VisionBenchmarkTask):
    """MMMU-Pro Vision: options embedded in the image itself."""

    name = "mmmu_pro_vision"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "MMMU/MMMU_Pro"  # TODO: Confirm exact HF dataset path
    dataset_config = "vision"
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        question = example["question"]
        prompt = (
            f"<|image|>\n\n{question}\n"
            f"The answer choices are shown in the image. "
            f"Answer with a single letter."
        )
        return prompt

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        gold = example["answer"].strip().upper()
        for char in prediction.strip().upper():
            if char in "ABCDEFGHIJ":
                return 1.0 if char == gold else 0.0
        return 0.0
