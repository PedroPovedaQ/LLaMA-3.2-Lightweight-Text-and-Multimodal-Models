from .mmlu import MMLUTask
from .gsm8k import GSM8KTask
from .hellaswag import HellaSwagTask
from .arc_challenge import ARCChallengeTask

TASKS = {
    "mmlu": MMLUTask,
    "gsm8k": GSM8KTask,
    "hellaswag": HellaSwagTask,
    "arc_challenge": ARCChallengeTask,
}
