"""VQAv2 — Visual Question Answering v2 (test, accuracy)."""

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class VQAv2Task(VisionBenchmarkTask):
    name = "vqav2"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "HuggingFaceM4/VQAv2"
    dataset_config = None
    dataset_split = "validation"  # test requires submission to eval server

    def format_prompt(self, example) -> str:
        question = example["question"]
        return f"<|image|>\n\n{question}\nAnswer briefly."

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        """VQAv2 soft accuracy: min(#humans_that_said_answer / 3, 1)."""
        pred = prediction.strip().lower()
        answers = example.get("answers", [])

        if not answers:
            gold = str(example.get("multiple_choice_answer", "")).strip().lower()
            return 1.0 if pred == gold else 0.0

        # Count how many annotators gave this answer
        answer_texts = [a["answer"].strip().lower() for a in answers]
        count = sum(1 for a in answer_texts if a == pred)
        return min(count / 3.0, 1.0)
