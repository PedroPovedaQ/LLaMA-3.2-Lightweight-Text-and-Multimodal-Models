"""HellaSwag — Commonsense Reasoning (0-shot, log-likelihood accuracy)."""

from benchmarks.tasks.base import BenchmarkTask, score_multiple_choice_loglikelihood


class HellaSwagTask(BenchmarkTask):
    name = "hellaswag"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "Rowan/hellaswag"
    dataset_split = "validation"  # HellaSwag test labels are hidden

    def format_prompt(self, example) -> str:
        """Build the context prompt from activity label + context.

        Each example has: activity_label, ctx_a, ctx_b, endings (list of 4), label.
        """
        ctx = example["ctx_a"]
        if example.get("ctx_b"):
            ctx += " " + example["ctx_b"]
        return ctx

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        choices = [" " + ending for ending in example["endings"]]

        best_idx, tokens = score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices)
        return {"prediction": best_idx, "tokens_generated": tokens}

    def score(self, prediction, example) -> float:
        gold = int(example["label"])
        return 1.0 if prediction == gold else 0.0
