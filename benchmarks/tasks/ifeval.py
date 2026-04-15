"""IFEval — Instruction Following Evaluation (accuracy)."""

import torch

from benchmarks.tasks.base import BenchmarkTask


class IFEvalTask(BenchmarkTask):
    name = "ifeval"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "google/IFEval"
    dataset_split = "train"
    max_new_tokens = 512

    def format_prompt(self, example) -> str:
        return example["prompt"]

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
        # TODO: Implement instruction-following constraint checking.
        # IFEval checks whether the response satisfies specific formatting
        # constraints (e.g., "write in all caps", "include exactly 3 bullet points").
        # See: https://arxiv.org/abs/2311.07911
        raise NotImplementedError(
            "IFEval scoring requires constraint verification logic. "
            "See the IFEval paper for the full list of verifiable instructions."
        )
