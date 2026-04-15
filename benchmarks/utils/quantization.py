"""Quantization wrapper — loads any model at FP16, INT8, or INT4.

See: https://github.com/PedroPovedaQ/LLaMA-3.2-Lightweight-Text-and-Multimodal-Models/issues/2
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

QUANT_LEVELS = ["fp16", "int8", "int4"]


def load_model(model_name, quant="fp16", cache_dir="./models"):
    """Load a model and tokenizer at the specified quantization level.

    Args:
        model_name: Key from MODEL_REGISTRY (e.g. "llama-3.2-1b") or a full HF model path.
        quant: One of "fp16", "int8", "int4".
        cache_dir: Directory to cache downloaded model weights.

    Returns:
        Tuple of (model, tokenizer).
    """
    hf_name = MODEL_REGISTRY.get(model_name, model_name)

    quantization_config = None
    kwargs = {}

    if quant == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quant == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unknown quant level: {quant!r}. Must be one of {QUANT_LEVELS}")

    if quantization_config:
        kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, device_map="auto", cache_dir=cache_dir, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_memory_footprint_mb(model):
    """Return the model's GPU memory footprint in MB."""
    return model.get_memory_footprint() / (1024 * 1024)
