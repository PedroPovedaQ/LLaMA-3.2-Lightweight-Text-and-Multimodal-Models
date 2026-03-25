"""TLDR9+ — Summarization (1-shot, rougeL)."""

import torch
from rouge_score import rouge_scorer

from benchmarks.tasks.base import BenchmarkTask


class TLDR9Task(BenchmarkTask):
    name = "tldr9"
    n_shot = 1
    metric_name = "rougeL"
    dataset_name = "webis/tldr-17"  # TODO: Confirm exact HF dataset for TLDR9+
    dataset_split = "test"
    max_new_tokens = 256

    def __init__(self, max_samples=None):
        super().__init__(max_samples)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def format_prompt(self, example) -> str:
        prompt = ""

        for fs in self.few_shot_examples:
            prompt += f"Article: {fs['content']}\nTL;DR: {fs['summary']}\n\n"

        prompt += f"Article: {example['content']}\nTL;DR:"
        return prompt

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
        reference = example["summary"]
        scores = self.scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
