"""
config.py — Central Configuration
==================================
All hyperparameters, model definitions, benchmark settings, and prompt
templates for the LIGHTWEIGHT instruction-tuned model evaluation.

Models: Llama 3.2 1B, Llama 3.2 3B, Phi-3.5-mini, Gemma 2 2B
Benchmarks: MMLU (5-shot), Open-rewrite eval, TLDR9+, IFEval
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# 1. MODEL REGISTRY
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single text-only model."""
    hf_model_id: str              # HuggingFace model identifier
    model_family: str             # Used to select chat template / prompting
    torch_dtype: str = "bfloat16" # Precision (bfloat16 is native on H100/V100)
    device_map: str = "auto"      # Let accelerate handle device placement
    max_new_tokens: int = 512     # Max tokens to generate per response
    trust_remote_code: bool = False


MODEL_REGISTRY: Dict[str, ModelConfig] = {

    # ---- Primary models (from the paper) ----

    "llama-3.2-1b": ModelConfig(
        hf_model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_family="llama",
        max_new_tokens=512,
    ),

    "llama-3.2-3b": ModelConfig(
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_family="llama",
        max_new_tokens=512,
    ),

    # ---- Comparison models ----

    # Phi-3.5-mini (3.8B params)
    # Uses float16 and shorter max tokens to fit on 16GB V100.
    "phi-3.5-mini": ModelConfig(
        hf_model_id="microsoft/Phi-3.5-mini-instruct",
        model_family="phi",
        trust_remote_code=True,
        torch_dtype="float16",
        max_new_tokens=256,
    ),

    # Gemma 2 2B IT — the other comparison model in the paper's table
    "gemma-2-2b": ModelConfig(
        hf_model_id="google/gemma-2-2b-it",
        model_family="gemma",
        max_new_tokens=512,
    ),
}


# =============================================================================
# 2. BENCHMARK CONFIGURATIONS
# =============================================================================

@dataclass
class MMLUConfig:
    """
    MMLU (Massive Multitask Language Understanding)
    Paper setting: 5-shot, accuracy.

    57 subjects across STEM, humanities, social sciences, and more.
    Each question has 4 options (A/B/C/D).
    5-shot means we prepend 5 solved examples before the test question.
    """
    dataset_name: str = "cais/mmlu"
    dataset_config: str = "all"
    test_split: str = "test"
    few_shot_split: str = "validation"
    num_shots: int = 5
    metric: str = "accuracy"
    paper_scores: Dict[str, float] = field(default_factory=lambda: {
        "llama-3.2-1b": 49.3,
        "llama-3.2-3b": 63.4,
        "gemma-2-2b": 57.8,
        "phi-3.5-mini": 69.0,
    })


@dataclass
class OpenRewriteConfig:
    """
    Open-rewrite eval (0-shot, rougeL)
    Evaluates the model's ability to rewrite/paraphrase text.
    Scored with ROUGE-L against reference rewrites.
    """
    dataset_name: str = "euclaise/writingprompts"
    test_split: str = "test"
    num_shots: int = 0
    metric: str = "rougeL"
    max_samples: int = 500
    paper_scores: Dict[str, float] = field(default_factory=lambda: {
        "llama-3.2-1b": 41.6,
        "llama-3.2-3b": 40.1,
        "gemma-2-2b": 31.2,
        "phi-3.5-mini": 34.5,
    })


@dataclass
class TLDR9Config:
    """
    TLDR9+ (test, 1-shot, rougeL)
    Summarization benchmark using Reddit posts.
    1-shot = one example summary provided before the test input.
    Scored with ROUGE-L against reference summaries.

    Dataset columns: 'prompt' (the Reddit post) and 'label' (reference summary)
    """
    dataset_name: str = "CarperAI/openai_summarize_tldr"
    test_split: str = "test"
    num_shots: int = 1
    metric: str = "rougeL"
    max_samples: int = 500
    paper_scores: Dict[str, float] = field(default_factory=lambda: {
        "llama-3.2-1b": 16.8,
        "llama-3.2-3b": 19.0,
        "gemma-2-2b": 13.9,
        "phi-3.5-mini": 12.8,
    })


@dataclass
class IFEvalConfig:
    """
    IFEval (Instruction Following Evaluation)
    Tests whether models follow specific, verifiable instructions.
    Each prompt has one or more verifiable constraints.
    Metric: prompt-level strict accuracy (ALL constraints must be met).
    """
    dataset_name: str = "google/IFEval"
    test_split: str = "train"  # IFEval only has a train split on HF
    metric: str = "prompt_level_strict_accuracy"
    paper_scores: Dict[str, float] = field(default_factory=lambda: {
        "llama-3.2-1b": 59.5,
        "llama-3.2-3b": 77.4,
        "gemma-2-2b": 61.9,
        "phi-3.5-mini": 59.2,
    })


# =============================================================================
# 3. PROMPT TEMPLATES
# =============================================================================

MMLU_EXAMPLE_TEMPLATE = (
    "Question: {question}\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "Answer: {answer}"
)

MMLU_QUESTION_TEMPLATE = (
    "Question: {question}\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}\n"
    "Answer:"
)

TLDR_PROMPT = (
    "Summarize the following post in one or two sentences.\n\n"
    "Post: {post}\n\n"
    "TL;DR:"
)

REWRITE_PROMPT = (
    "Rewrite the following text in a different way while preserving "
    "the original meaning.\n\n"
    "Original: {text}\n\n"
    "Rewritten:"
)


# =============================================================================
# 4. PATHS
# =============================================================================

@dataclass
class PathConfig:
    results_dir: str = "results"
    hf_cache_dir: Optional[str] = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )


# =============================================================================
# 5. HELPERS
# =============================================================================

def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{model_name}' not found. Available: {available}")
    return MODEL_REGISTRY[model_name]


def get_all_model_names() -> List[str]:
    return list(MODEL_REGISTRY.keys())
