"""ARC-Challenge — Reading Comprehension (25-shot, log-likelihood accuracy)."""

from benchmarks.tasks.base import BenchmarkTask, ANSWER_CHOICES, score_multiple_choice_loglikelihood


class ARCChallengeTask(BenchmarkTask):
    name = "arc_challenge"
    n_shot = 25
    metric_name = "accuracy"
    dataset_name = "allenai/ai2_arc"
    dataset_config = "ARC-Challenge"
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        """Build a 25-shot prompt for an ARC-Challenge question.

        Each example has: question, choices (dict with text + label lists), answerKey.
        """
        prompt = ""

        for fs in self.few_shot_examples:
            prompt += self._format_single(fs, include_answer=True) + "\n\n"

        prompt += self._format_single(example, include_answer=False)
        return prompt

    def _format_single(self, example, include_answer=False):
        question = example["question"]
        choice_texts = example["choices"]["text"]
        choice_labels = example["choices"]["label"]

        text = f"Question: {question}\n"
        for label, choice in zip(choice_labels, choice_texts):
            text += f"{label}. {choice}\n"
        text += "Answer:"

        if include_answer:
            text += f" {example['answerKey']}"
        return text

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        choice_labels = example["choices"]["label"]
        choices = [f" {label}" for label in choice_labels]

        best_idx, tokens = score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices)
        return {"prediction": choice_labels[best_idx], "tokens_generated": tokens}

    def score(self, prediction, example) -> float:
        return 1.0 if prediction == example["answerKey"] else 0.0
