# --- Text benchmarks ---
from .mmlu import MMLUTask
from .gsm8k import GSM8KTask
from .hellaswag import HellaSwagTask
from .arc_challenge import ARCChallengeTask
from .math_bench import MATHTask
from .gpqa import GPQATask
from .mgsm import MGSMTask
from .ifeval import IFEvalTask
from .open_rewrite import OpenRewriteTask
from .tldr9 import TLDR9Task
from .bfcl_v2 import BFCLV2Task
from .nexus import NexusTask
from .infinitebench import InfiniteBenchMCTask, InfiniteBenchQATask
from .nih_multi_needle import NIHMultiNeedleTask

# --- Vision benchmarks ---
from .vision import VISION_TASKS

TEXT_TASKS = {
    # General
    "mmlu": MMLUTask,
    "open_rewrite": OpenRewriteTask,
    "tldr9": TLDR9Task,
    "ifeval": IFEvalTask,
    # Tool use
    "bfcl_v2": BFCLV2Task,
    "nexus": NexusTask,
    # Math
    "gsm8k": GSM8KTask,
    "math": MATHTask,
    # Reasoning
    "arc_challenge": ARCChallengeTask,
    "gpqa": GPQATask,
    "hellaswag": HellaSwagTask,
    # Long context
    "infinitebench_mc": InfiniteBenchMCTask,
    "infinitebench_qa": InfiniteBenchQATask,
    "nih_multi_needle": NIHMultiNeedleTask,
    # Multilingual
    "mgsm": MGSMTask,
}

# Combined registry (text + vision)
TASKS = {**TEXT_TASKS, **VISION_TASKS}
