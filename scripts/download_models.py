#!/usr/bin/env python3
"""
Model download/pull script for small model comparison project.
Supports both Hugging Face and Ollama providers.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from experiment_utils import MODEL_SPECS


def setup_huggingface() -> bool:
    """Setup Hugging Face authentication if needed."""
    try:
        from huggingface_hub import whoami

        try:
            user = whoami()
            print(f"✅ Already logged in to Hugging Face as: {user['name']}")
            return True
        except Exception:
            print("🔐 Hugging Face login required for some models")
            print("Run: hf auth login")
            return False
    except ImportError:
        print("⚠️  huggingface_hub not installed")
        return False


def ensure_ollama_available() -> bool:
    """Verify ollama CLI exists."""
    try:
        proc = subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        version = (proc.stdout or proc.stderr).strip()
        print(f"✅ Ollama available: {version}")
        return True
    except FileNotFoundError:
        print("❌ Ollama CLI not found. Install from: https://ollama.com/download")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to run ollama CLI: {exc}")
        return False


def get_model_config() -> Dict[str, Dict[str, Optional[str]]]:
    """Get configuration for all target models across providers."""
    return {
        key: {
            "hf_name": spec.hf_model_id,
            "ollama_name": spec.ollama_tag,
            "path": spec.download_path,
            "description": spec.description,
        }
        for key, spec in MODEL_SPECS.items()
    }


def download_model_hf(model_name: str, model_path: str, cache_dir: Optional[str] = None) -> bool:
    """Download a model from Hugging Face and save locally."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"📥 Downloading from Hugging Face: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
        )

        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)

        print(f"✅ {model_name} downloaded to {model_path}")
        return True

    except Exception as exc:
        print(f"❌ Failed to download {model_name}: {exc}")
        return False


def pull_model_ollama(ollama_name: str, model_path: str, model_key: str) -> bool:
    """Pull a model from Ollama and write a local manifest pointer."""
    try:
        print(f"📥 Pulling from Ollama: {ollama_name}...")
        subprocess.run(["ollama", "pull", ollama_name], check=True)

        os.makedirs(model_path, exist_ok=True)
        manifest_path = Path(model_path) / "OLLAMA_MODEL.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_key": model_key,
                    "provider": "ollama",
                    "ollama_name": ollama_name,
                    "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
                    "note": "Model blob is stored in Ollama's local model store.",
                },
                f,
                indent=2,
                ensure_ascii=True,
            )

        print(f"✅ {ollama_name} pulled via Ollama")
        print(f"📄 Wrote pointer manifest: {manifest_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to pull {ollama_name} via Ollama: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/pull models for comparison")
    parser.add_argument(
        "--provider",
        choices=["hf", "ollama"],
        default="hf",
        help="Model provider: hf (Hugging Face) or ollama",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "all",
            "llama-3.2-1b",
            "llama-3.2-3b",
            "phi-3-mini",
            "gemma-2b",
            "tinyllama",
            "qwen2-1.5b",
        ],
        default=["llama-3.2-1b", "llama-3.2-3b", "phi-3-mini"],
        help="Models to download/pull",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face downloads",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    model_config = get_model_config()

    if "all" in args.models:
        models_to_process = list(model_config.keys())
    else:
        models_to_process = args.models

    if args.provider == "hf":
        print("🤗 Using Hugging Face provider...")
        setup_huggingface()
    else:
        print("🦙 Using Ollama provider...")
        if not ensure_ollama_available():
            raise SystemExit(1)

    print(f"\n📥 Processing {len(models_to_process)} models...")
    print("=" * 50)

    success_count = 0
    for model_key in models_to_process:
        config = model_config[model_key]
        print(f"\n📦 {config['description']}")

        if args.provider == "hf":
            if download_model_hf(config["hf_name"], config["path"], args.cache_dir):
                success_count += 1
        else:
            if not config.get("ollama_name"):
                print(
                    f"⚠️  No Ollama mapping for {model_key}. "
                    "Use --provider hf for this model."
                )
                continue
            if pull_model_ollama(config["ollama_name"], config["path"], model_key):
                success_count += 1

    print("\n" + "=" * 50)
    print(f"✅ Successfully processed {success_count}/{len(models_to_process)} models")

    if success_count < len(models_to_process):
        print("\n⚠️  Some models failed. Check:")
        print("1. Internet connection")
        if args.provider == "hf":
            print("2. Hugging Face authentication / gated model approval")
        else:
            print("2. Ollama installation and available model tags")
        print("3. Sufficient disk space")


if __name__ == "__main__":
    main()
