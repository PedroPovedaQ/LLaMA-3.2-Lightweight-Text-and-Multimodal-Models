"""Open-rewrite eval (0-shot, rougeL)."""

import torch
from rouge_score import rouge_scorer

from benchmarks.tasks.base import BenchmarkTask


class OpenRewriteTask(BenchmarkTask):
    name = "open_rewrite"
    n_shot = 0
    metric_name = "rougeL"
    dataset_name = "openai/openai_humaneval"  # TODO: Replace with correct dataset
    dataset_split = "test"
    max_new_tokens = 512

    def __init__(self, max_samples=None):
        super().__init__(max_samples)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def format_prompt(self, example) -> str:
        # TODO: Implement prompt formatting once dataset is confirmed.
        raise NotImplementedError("Needs correct dataset source for open-rewrite eval.")

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {"prediction": prediction_text, "tokens_generated": generated_ids.shape[1]}

    def score(self, prediction, example) -> float:
        reference = example["canonical_solution"]  # TODO: Adjust field name
        scores = self.scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
