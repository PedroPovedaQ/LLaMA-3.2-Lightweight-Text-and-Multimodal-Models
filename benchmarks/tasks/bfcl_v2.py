"""BFCL V2 — Berkeley Function Calling Leaderboard (tool use, accuracy)."""

import torch

from benchmarks.tasks.base import BenchmarkTask


class BFCLV2Task(BenchmarkTask):
    name = "bfcl_v2"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    dataset_split = "test"
    max_new_tokens = 512

    def format_prompt(self, example) -> str:
        # TODO: Implement prompt formatting with function definitions and user query.
        # BFCL provides function schemas + user question, model must produce a valid call.
        raise NotImplementedError(
            "BFCL V2 requires function-calling prompt formatting. "
            "See: https://gorilla.cs.berkeley.edu/leaderboard.html"
        )

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
        # TODO: Parse predicted function call and compare against ground truth.
        # Needs AST-level comparison of function name + arguments.
        raise NotImplementedError("BFCL V2 scoring requires function call AST comparison.")
