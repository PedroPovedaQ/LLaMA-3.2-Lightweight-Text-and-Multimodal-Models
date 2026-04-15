#!/usr/bin/env python3
"""Shared utilities for benchmark and efficiency experiments."""

from __future__ import annotations

import argparse
import json
import platform
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency at runtime
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoModelForCausalLM = Any  # type: ignore[assignment]
    AutoTokenizer = Any  # type: ignore[assignment]

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency at runtime
    BitsAndBytesConfig = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
PROCESSED_RESULTS_DIR = PROJECT_ROOT / "results" / "processed"


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime settings used in a run."""

    model_id: str
    precision: str
    device: str
    cache_dir: Optional[str]


@dataclass(frozen=True)
class ModelSpec:
    """Canonical per-model identifiers for HF downloads, Ollama pulls, and CLI keys."""

    hf_model_id: str
    ollama_tag: Optional[str]
    download_path: str
    description: str


# Model registry entry point:
# Add new models here (HF id, optional Ollama tag, local download path, description).
MODEL_SPECS: Dict[str, ModelSpec] = {
    "llama-3.2-1b": ModelSpec(
        hf_model_id="meta-llama/Llama-3.2-1B-Instruct",
        ollama_tag="llama3.2:1b",
        download_path="models/llama-3.2-1b",
        description="LLaMA 3.2 1B parameter model",
    ),
    "llama-3.2-3b": ModelSpec(
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        ollama_tag="llama3.2",
        download_path="models/llama-3.2-3b",
        description="LLaMA 3.2 3B parameter model",
    ),
    "phi-3-mini": ModelSpec(
        hf_model_id="microsoft/Phi-3-mini-4k-instruct",
        ollama_tag="phi3:mini",
        download_path="models/phi-3-mini",
        description="Microsoft Phi-3 Mini model",
    ),
    "gemma-2b": ModelSpec(
        hf_model_id="google/gemma-2b-it",
        ollama_tag=None,
        download_path="models/others/gemma-2b",
        description="Google Gemma 2B model",
    ),
    "tinyllama": ModelSpec(
        hf_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ollama_tag="tinyllama",
        download_path="models/others/tinyllama-1.1b",
        description="TinyLlama 1.1B model",
    ),
    "qwen2-1.5b": ModelSpec(
        hf_model_id="Qwen/Qwen2-1.5B-Instruct",
        ollama_tag=None,
        download_path="models/others/qwen2-1.5b",
        description="Qwen2 1.5B model",
    ),
}

MODEL_KEYS: Dict[str, str] = {k: v.hf_model_id for k, v in MODEL_SPECS.items()}


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return MODEL_KEYS[args.model_key]


def make_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def ensure_results_dirs() -> None:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def safe_take(dataset: Iterable[dict], limit: int) -> List[dict]:
    if limit <= 0:
        return []
    return [dataset[i] for i in range(min(limit, len(dataset)))]


def parse_mc_answer(text: str, choices: List[str]) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None

    # Priority 1: direct single-letter prediction.
    letter_matches = re.findall(r"\b([A-Z])\b", cleaned.upper())
    for candidate in letter_matches:
        if candidate in LETTERS[: len(choices)]:
            return candidate

    # Priority 2: text overlap with one option.
    lowered = cleaned.lower()
    matched = []
    for idx, choice in enumerate(choices):
        choice_text = choice.strip().lower()
        if choice_text and choice_text in lowered:
            matched.append(LETTERS[idx])

    if len(matched) == 1:
        return matched[0]
    return None


def normalize_numeric(value: str) -> Optional[str]:
    raw = value.strip().replace(",", "")
    if not raw:
        return None

    try:
        dec = Decimal(raw)
    except InvalidOperation:
        return None

    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal("1")))
    return format(dec.normalize(), "f").rstrip("0").rstrip(".")


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    return normalize_numeric(matches[-1])


def resolve_device(device: str) -> str:
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    normalized = device.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    if normalized not in {"cuda", "cpu", "mps"}:
        raise ValueError("--device must be one of: auto, cuda, cpu, mps")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA selected but no CUDA device is available")
    if normalized == "mps" and not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        raise ValueError("MPS selected but no MPS device is available")
    return normalized


def _build_model_load_kwargs(precision: str, device: str) -> Dict[str, object]:
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    precision = precision.lower()
    kwargs: Dict[str, object] = {"low_cpu_mem_usage": True}

    if precision == "fp16":
        # Always set fp16 explicitly so runtime metadata matches the actual load dtype
        # on every device, including CPU.
        kwargs["torch_dtype"] = torch.float16
    elif precision == "bf16":
        if device == "cuda":
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            raise ValueError("bf16 is only supported with CUDA in this baseline pipeline")
    elif precision == "int4":
        if device != "cuda":
            raise ValueError("int4 quantization via bitsandbytes requires CUDA")
        if BitsAndBytesConfig is None:
            raise ValueError(
                "int4 requested but BitsAndBytesConfig is unavailable. Install transformers+bitsandbytes"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = "auto"
    elif precision == "int2":
        raise ValueError(
            "int2 is not supported in this transformers baseline. Use GGUF/llama.cpp for 2-bit runs."
        )
    else:
        raise ValueError("--precision must be one of: fp16, bf16, int4, int2")

    if "device_map" not in kwargs and device == "cuda":
        kwargs["device_map"] = "auto"

    return kwargs


def _should_retry_with_remote_code(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = [
        "trust_remote_code",
        "requires you to execute the configuration file",
        "requires you to execute the modeling file",
        "custom code",
        "remote code",
    ]
    return any(marker in message for marker in retry_markers)


def _from_pretrained_with_fallback(factory, model_id: str, cache_dir: Optional[str], **kwargs):
    try:
        return factory.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False,
            **kwargs,
        )
    except Exception as exc:
        if not _should_retry_with_remote_code(exc):
            raise
        return factory.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **kwargs,
        )


def load_model_and_tokenizer(
    model_id: str,
    precision: str,
    device: str,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any, RuntimeConfig]:
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")
    if AutoModelForCausalLM is Any or AutoTokenizer is Any:
        raise ImportError(
            "transformers is required. Install dependencies: pip install -r requirements.txt"
        )

    runtime = RuntimeConfig(
        model_id=model_id,
        precision=precision.lower(),
        device=resolve_device(device),
        cache_dir=cache_dir,
    )

    candidate_dir = Path(runtime.model_id).expanduser()
    if candidate_dir.exists() and candidate_dir.is_dir():
        has_hf_weights = any(
            [
                (candidate_dir / "model.safetensors").exists(),
                (candidate_dir / "pytorch_model.bin").exists(),
                (candidate_dir / "model.safetensors.index.json").exists(),
                any(candidate_dir.glob("*.safetensors")),
            ]
        )
        has_llama_original = (candidate_dir / "original" / "consolidated.00.pth").exists()
        if has_llama_original and not has_hf_weights:
            raise ValueError(
                "Detected llama-model original checkpoint format at "
                f"'{candidate_dir}'. This benchmark runner uses Transformers "
                "and requires HF-formatted weights (model.safetensors or pytorch_model.bin). "
                "Use --provider hf for benchmark-ready checkpoints, or use the Ollama benchmark path."
            )

    tokenizer = _from_pretrained_with_fallback(
        AutoTokenizer,
        runtime.model_id,
        runtime.cache_dir,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = _build_model_load_kwargs(runtime.precision, runtime.device)
    model = _from_pretrained_with_fallback(
        AutoModelForCausalLM,
        runtime.model_id,
        runtime.cache_dir,
        **model_kwargs,
    )

    if runtime.device in {"cpu", "mps"}:
        model = model.to(runtime.device)

    model.eval()
    return model, tokenizer, runtime


def model_input_device(model: Any, runtime_device: str):
    if torch is None:
        raise ImportError("PyTorch is required. Install dependencies: pip install -r requirements.txt")

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(runtime_device)


def format_user_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def capture_hardware_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }

    if psutil is not None:
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)

    if torch is not None and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_count"] = torch.cuda.device_count()
    return info


def save_json(payload: Dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return output_path


def csv_safe(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)
