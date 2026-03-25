"""InfiniteBench — Long Context Evaluation (128k, accuracy/QA)."""

import torch

from benchmarks.tasks.base import BenchmarkTask, score_multiple_choice_loglikelihood, ANSWER_CHOICES


class InfiniteBenchMCTask(BenchmarkTask):
    """InfiniteBench English Multiple Choice (128k context)."""

    name = "infinitebench_mc"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "xinrongzhang2022/InfiniteBench"
    dataset_config = "En.MC"
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        context = example["context"]
        question = example["question"]
        options = example["options"]

        text = f"{context}\n\nQuestion: {question}\n"
        for i, opt in enumerate(options):
            text += f"{ANSWER_CHOICES[i]}. {opt}\n"
        text += "Answer:"
        return text

    def evaluate_sample(self, model, tokenizer, example) -> dict:
        prompt = self.format_prompt(example)
        options = example["options"]
        choices = [f" {ANSWER_CHOICES[i]}" for i in range(len(options))]

        best_idx, tokens = score_multiple_choice_loglikelihood(model, tokenizer, prompt, choices)
        return {"prediction": best_idx, "tokens_generated": tokens}

    def score(self, prediction, example) -> float:
        return 1.0 if prediction == example["answer"] else 0.0


class InfiniteBenchQATask(BenchmarkTask):
    """InfiniteBench English QA (128k context)."""

    name = "infinitebench_qa"
    n_shot = 0
    metric_name = "accuracy"
    dataset_name = "xinrongzhang2022/InfiniteBench"
    dataset_config = "En.QA"
    dataset_split = "test"
    max_new_tokens = 128

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
        gold = example["answer"].strip().lower()
        pred = prediction.strip().lower()
        return 1.0 if gold in pred else 0.0
