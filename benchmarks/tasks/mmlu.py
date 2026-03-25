"""MMLU — Multitask Language Understanding (5-shot, accuracy)."""

from benchmarks.tasks.base import BenchmarkTask, ANSWER_CHOICES, score_multiple_choice_loglikelihood


class MMLUTask(BenchmarkTask):
    name = "mmlu"
    n_shot = 5
    metric_name = "accuracy"
    dataset_name = "cais/mmlu"
    dataset_config = "all"
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        """Build a 5-shot prompt for an MMLU question.

        Each example has: question, choices (list of 4), answer (0-3), subject.
        """
        prompt = ""

        # Few-shot examples
        for fs in self.few_shot_examples:
            prompt += self._format_single(fs, include_answer=True) + "\n\n"

        # Target question (no answer)
        prompt += self._format_single(example, include_answer=False)
        return prompt

    def _format_single(self, example, include_answer=False):
        question = example["question"]
        choices = example["choices"]

        text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            text += f"{ANSWER_CHOICES[i]}. {choice}\n"
        text += "Answer:"

        if include_answer:
            text += f" {ANSWER_CHOICES[example['answer']]}"
        return text

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        choices = [f" {ANSWER_CHOICES[i]}" for i in range(len(example["choices"]))]

        best_idx, tokens = score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices)
        return {"prediction": best_idx, "tokens_generated": tokens}

    def score(self, prediction, example) -> float:
        return 1.0 if prediction == example["answer"] else 0.0
