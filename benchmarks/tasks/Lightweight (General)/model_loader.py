"""
model_loader.py — Text Model Loader
=====================================
Loads text-only instruction-tuned models from the registry.
All models here are small (1B-3.8B) and fit on a single GPU easily.

Supports: Llama 3.2 (1B/3B), Phi-3.5-mini, Gemma 2 2B
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import get_model_config, ModelConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(model_name: str):
    """
    Load a text model and its tokenizer.

    Args:
        model_name: Key from MODEL_REGISTRY (e.g., "llama-3.2-1b")

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    config = get_model_config(model_name)
    logger.info(f"Loading model: {model_name} ({config.hf_model_id})")

    torch_dtype = DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_model_id,
        trust_remote_code=config.trust_remote_code,
    )

    # Ensure we have a pad token (some models don't set one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load model ----
    # Try 'torch_dtype' first (works on transformers <=4.46),
    # fall back to 'dtype' (newer transformers versions).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            torch_dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )

    model.eval()
    logger.info(f"Model loaded. dtype={torch_dtype}, params={_count_params(model)}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str,
                      model_family: str = "llama",
                      max_new_tokens: int = 512,
                      use_chat_template: bool = True) -> str:
    """
    Generate a text response given a prompt.

    For instruction-tuned models, we use their chat template to format
    the prompt correctly. Each model family expects a different format.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The user message text
        model_family: "llama", "phi", or "gemma"
        max_new_tokens: Max tokens to generate
        use_chat_template: If True, wrap prompt in the model's chat format

    Returns:
        The model's generated text (decoded, prompt stripped).
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt
    else:
        formatted = prompt

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                       max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate (greedy decoding for reproducibility)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the input prompt)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip()


def generate_completion(model, tokenizer, prompt: str,
                        max_new_tokens: int = 10) -> str:
    """
    Generate a short completion (no chat template).

    Used for MMLU where we want the model to simply complete
    "Answer:" with a single letter.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip()


def _count_params(model) -> str:
    """Return human-readable parameter count."""
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(n)
