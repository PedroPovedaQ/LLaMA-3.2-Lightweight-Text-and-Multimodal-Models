"""GPQA — Graduate-Level Reasoning (0-shot, accuracy)."""

from benchmarks.tasks.base import BenchmarkTask, ANSWER_CHOICES, score_multiple_choice_loglikelihood


class GPQATask(BenchmarkTask):
    name = "gpqa"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "Idavidrein/gpqa"
    dataset_config = "gpqa_diamond"
    dataset_split = "train"  # GPQA only has a train split publicly

    def format_prompt(self, example) -> str:
        question = example["Question"]
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        # Store shuffled order for scoring
        example["_choices"] = choices

        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"{ANSWER_CHOICES[i]}. {choice}\n"
        text += "Answer:"
        return text

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        choices = [f" {ANSWER_CHOICES[i]}" for i in range(4)]

        best_idx, tokens = score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices)
        return {"prediction": best_idx, "tokens_generated": tokens}

    def score(self, prediction, example) -> float:
        # Correct answer is always index 0 in our choices list
        return 1.0 if prediction == 0 else 0.0
