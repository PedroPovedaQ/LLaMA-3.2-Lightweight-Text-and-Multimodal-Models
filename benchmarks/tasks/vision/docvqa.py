"""DocVQA — Document Visual Question Answering (ANLS metric)."""

from benchmarks.tasks.vision.base_vision import VisionBenchmarkTask


class DocVQATask(VisionBenchmarkTask):
    name = "docvqa"
    n_shot = 0
    metric_name = "anls"
    dataset_name = "lmms-lab/DocVQA"
    dataset_config = None
    dataset_split = "test"

    def format_prompt(self, example) -> str:
        question = example["question"]
        return f"<|image|>\n\n{question}\nAnswer briefly."

    def get_images(self, example) -> list:
        image = example.get("image")
        return [image] if image is not None else []

    def score(self, prediction, example) -> float:
        """Average Normalized Levenshtein Similarity (ANLS).

        Standard DocVQA metric: 1 - NLD(pred, gold) if NLD < threshold, else 0.
        Threshold is typically 0.5.
        """
        answers = example.get("answers", [example.get("answer", "")])
        if not answers:
            return 0.0

        pred = prediction.strip().lower()
        best_score = 0.0

        for ans in answers:
            gold = str(ans).strip().lower()
            nld = _normalized_levenshtein_distance(pred, gold)
            score = 1.0 - nld if nld < 0.5 else 0.0
            best_score = max(best_score, score)

        return best_score


def _normalized_levenshtein_distance(s1, s2):
    """Compute normalized Levenshtein distance between two strings."""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0

    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp

    return dp[n] / max(m, n)
