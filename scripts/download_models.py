#!/usr/bin/env python3
"""
Model download/pull script for small model comparison project.
Supports Hugging Face, Ollama, and Llama CLI providers.
"""

import argparse
import json
import os
import subprocess
import sys
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


def ensure_llama_cli_available() -> bool:
    """Verify llama-model CLI exists."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import llama_models; print(llama_models.__file__)"],
            check=True,
            capture_output=True,
            text=True,
        )
        location = (proc.stdout or "").strip()
        print(f"✅ llama-models Python package available: {location}")
        return True
    except Exception:
        print("❌ llama-models package not available. Install with: python -m pip install -U llama-models")
        return False


def _resolve_llama_checkpoint_dir(model_id: str) -> str:
    """Best-effort local checkpoint path used by llama-model download."""
    try:
        from llama_models.utils.model_utils import model_local_dir

        return str(Path(model_local_dir(model_id)))
    except Exception:
        return str(Path.home() / ".llama" / "checkpoints" / model_id)


def get_model_config() -> Dict[str, Dict[str, Optional[str]]]:
    """Get configuration for all target models across providers."""
    llama_cli_ids = {
        "llama-3.2-1b": "Llama3.2-1B-Instruct",
        "llama-3.2-3b": "Llama3.2-3B-Instruct",
    }
    return {
        key: {
            "hf_name": spec.hf_model_id,
            "ollama_name": spec.ollama_tag,
            "llama_model_id": llama_cli_ids.get(key),
            "path": spec.download_path,
            "description": spec.description,
        }
        for key, spec in MODEL_SPECS.items()
    }


def default_models_for_provider(provider: str) -> list[str]:
    """Return provider-specific default model set."""
    if provider == "llama-cli":
        # llama-cli path currently mapped to Llama models in this project.
        return ["llama-3.2-1b", "llama-3.2-3b"]
    return ["llama-3.2-1b", "llama-3.2-3b", "phi-3-mini"]


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


def _from_pretrained_with_fallback(factory, model_name: str, cache_dir: Optional[str], **kwargs):
    try:
        return factory.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=False,
            **kwargs,
        )
    except Exception as exc:
        if not _should_retry_with_remote_code(exc):
            raise
        return factory.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **kwargs,
        )


def download_model_hf(model_name: str, model_path: str, cache_dir: Optional[str] = None) -> bool:
    """Download a model from Hugging Face and save locally."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"📥 Downloading from Hugging Face: {model_name}...")

        tokenizer = _from_pretrained_with_fallback(
            AutoTokenizer,
            model_name,
            cache_dir,
        )

        model = _from_pretrained_with_fallback(
            AutoModelForCausalLM,
            model_name,
            cache_dir,
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


def pull_model_llama_cli(
    llama_model_id: str,
    model_path: str,
    model_key: str,
    source: str,
    hf_token: Optional[str],
    meta_url: Optional[str],
    ignore_patterns: str,
) -> bool:
    """Download a model using Meta's llama-model CLI and write pointer manifest."""
    project_root = Path(__file__).parent.parent
    wrapper = project_root / "scripts" / "llama_cli_download_wrapper.py"
    cmd = [
        sys.executable,
        str(wrapper),
        "--source",
        source,
        "--model-id",
        llama_model_id,
    ]
    if source == "huggingface":
        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if ignore_patterns:
            cmd.extend(["--ignore-patterns", ignore_patterns])
    elif source == "meta" and meta_url:
        cmd.extend(["--meta-url", meta_url])

    print(f"📥 Downloading via llama-model ({source}): {llama_model_id}...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        combined = f"{stdout}\n{stderr}".strip()
        print(f"❌ Failed llama-model download for {llama_model_id}")
        if combined:
            print(combined)
        return False

    checkpoint_dir = _resolve_llama_checkpoint_dir(llama_model_id)
    os.makedirs(model_path, exist_ok=True)
    manifest_path = Path(model_path) / "LLAMA_CLI_MODEL.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_key": model_key,
                "provider": "llama-cli",
                "source": source,
                "llama_model_id": llama_model_id,
                "local_checkpoint_dir": checkpoint_dir,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "llama-model often downloads original checkpoint format "
                    "(original/consolidated.00.pth). Current benchmark scripts "
                    "use Transformers and require HF-formatted weights."
                ),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(f"✅ {llama_model_id} downloaded via llama-model")
    print(f"📄 Wrote pointer manifest: {manifest_path}")
    print(f"📂 Local checkpoint dir: {checkpoint_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/pull models for comparison")
    parser.add_argument(
        "--provider",
        choices=["hf", "ollama", "llama-cli"],
        default="llama-cli",
        help="Model provider: hf (Hugging Face), ollama, or llama-cli",
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
        default=None,
        help="Models to download/pull (default depends on --provider)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face downloads",
    )
    parser.add_argument(
        "--llama-source",
        choices=["meta", "huggingface"],
        default="huggingface",
        help="llama-cli source backend (only used with --provider llama-cli)",
    )
    parser.add_argument(
        "--meta-url",
        type=str,
        default=None,
        help="Signed llama.meta.com URL for --llama-source meta",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token forwarded to llama-model download when using --llama-source huggingface",
    )
    parser.add_argument(
        "--llama-ignore-patterns",
        type=str,
        default="",
        help="Patterns ignored by llama-model download when --llama-source huggingface",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    model_config = get_model_config()

    if args.models is None:
        models_to_process = default_models_for_provider(args.provider)
    elif "all" in args.models:
        models_to_process = list(model_config.keys())
    else:
        models_to_process = args.models

    if args.provider == "hf":
        print("🤗 Using Hugging Face provider...")
        setup_huggingface()
    elif args.provider == "ollama":
        print("🦙 Using Ollama provider...")
        if not ensure_ollama_available():
            raise SystemExit(1)
    else:
        print("🦙 Using Llama CLI provider...")
        if not ensure_llama_cli_available():
            raise SystemExit(1)
        if args.llama_source == "meta" and not args.meta_url:
            print("⚠️  --meta-url is not set. llama-model may prompt interactively for each model.")

    print(f"\n📥 Processing {len(models_to_process)} models...")
    print("=" * 50)

    success_count = 0
    for model_key in models_to_process:
        config = model_config[model_key]
        print(f"\n📦 {config['description']}")

        if args.provider == "hf":
            if download_model_hf(config["hf_name"], config["path"], args.cache_dir):
                success_count += 1
        elif args.provider == "ollama":
            if not config.get("ollama_name"):
                print(
                    f"⚠️  No Ollama mapping for {model_key}. "
                    "Use --provider hf for this model."
                )
                continue
            if pull_model_ollama(config["ollama_name"], config["path"], model_key):
                success_count += 1
        else:
            if not config.get("llama_model_id"):
                print(
                    f"⚠️  No llama-cli mapping for {model_key}. "
                    "Use --provider hf or --provider ollama for this model."
                )
                continue
            if pull_model_llama_cli(
                config["llama_model_id"],
                config["path"],
                model_key,
                source=args.llama_source,
                hf_token=args.hf_token,
                meta_url=args.meta_url,
                ignore_patterns=args.llama_ignore_patterns,
            ):
                success_count += 1

    print("\n" + "=" * 50)
    print(f"✅ Successfully processed {success_count}/{len(models_to_process)} models")

    if success_count < len(models_to_process):
        print("\n⚠️  Some models failed. Check:")
        print("1. Internet connection")
        if args.provider == "hf":
            print("2. Hugging Face authentication / gated model approval")
        elif args.provider == "ollama":
            print("2. Ollama installation and available model tags")
        else:
            print("2. llama-model CLI install/version and source credentials")
            print("3. If using --llama-source meta: signed URL validity")
            print("4. Sufficient disk space")
            return
        print("3. Sufficient disk space")


if __name__ == "__main__":
    main()
