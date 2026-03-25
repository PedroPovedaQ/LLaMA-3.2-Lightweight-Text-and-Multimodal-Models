"""NIH/Multi-needle — Needle-in-a-Haystack Long Context Retrieval (accuracy)."""

import torch

from benchmarks.tasks.base import BenchmarkTask


class NIHMultiNeedleTask(BenchmarkTask):
    name = "nih_multi_needle"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = ""  # TODO: This is typically a synthetic benchmark, not a HF dataset.
    dataset_split = "test"
    max_new_tokens = 128

    def load_data(self):
        # TODO: NIH/Multi-needle is usually generated synthetically.
        # Insert N "needle" facts into a long context of filler text,
        # then ask the model to retrieve all needles.
        raise NotImplementedError(
            "NIH/Multi-needle requires synthetic data generation. "
            "See: https://github.com/gkamradt/LLMTest_NeedleInAHaystack"
        )

    def format_prompt(self, example) -> str:
        context = example["context"]
        question = example["question"]
        return f"{context}\n\nQuestion: {question}\nAnswer:"

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=131072).to(model.device)

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
        # Check if all needle facts are retrieved
        needles = example["needles"]
        pred_lower = prediction.strip().lower()
        found = sum(1 for n in needles if n.lower() in pred_lower)
        return found / len(needles) if needles else 0.0
