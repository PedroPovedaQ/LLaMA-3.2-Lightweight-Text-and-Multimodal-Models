#!/usr/bin/env python3
"""
Model download script for small model comparison project.
Downloads and sets up target models for evaluation.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

def setup_huggingface():
    """Setup Hugging Face authentication if needed."""
    try:
        from huggingface_hub import login, whoami
        try:
            user = whoami()
            print(f"✅ Already logged in to Hugging Face as: {user['name']}")
            return True
        except:
            print("🔐 Hugging Face login required for some models")
            print("Run: huggingface-cli login")
            return False
    except ImportError:
        print("⚠️  huggingface_hub not installed")
        return False

def download_model(model_name: str, model_path: str, cache_dir: str = None) -> bool:
    """Download a model from Hugging Face."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"📥 Downloading {model_name}...")
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Download model  
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto"  # Use appropriate dtype
        )
        
        # Save locally
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"✅ {model_name} downloaded to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {str(e)}")
        return False

def get_model_config() -> Dict[str, Dict]:
    """Get configuration for all target models."""
    return {
        "llama-3.2-1b": {
            "hf_name": "meta-llama/Llama-3.2-1B-Instruct",
            "path": "models/llama-3.2-1b",
            "description": "LLaMA 3.2 1B parameter model"
        },
        "llama-3.2-3b": {
            "hf_name": "meta-llama/Llama-3.2-3B-Instruct", 
            "path": "models/llama-3.2-3b",
            "description": "LLaMA 3.2 3B parameter model"
        },
        "phi-3-mini": {
            "hf_name": "microsoft/Phi-3-mini-4k-instruct",
            "path": "models/phi-3-mini", 
            "description": "Microsoft Phi-3 Mini model"
        },
        "gemma-2b": {
            "hf_name": "google/gemma-2b-it",
            "path": "models/others/gemma-2b",
            "description": "Google Gemma 2B model"
        },
        "tinyllama": {
            "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "path": "models/others/tinyllama-1.1b",
            "description": "TinyLlama 1.1B model"
        },
        "qwen2-1.5b": {
            "hf_name": "Qwen/Qwen2-1.5B-Instruct",
            "path": "models/others/qwen2-1.5b", 
            "description": "Qwen2 1.5B model"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Download models for comparison")
    parser.add_argument(
        "--models", 
        nargs="+",
        choices=["all", "llama-3.2-1b", "llama-3.2-3b", "phi-3-mini", "gemma-2b", "tinyllama", "qwen2-1.5b"],
        default=["llama-3.2-1b", "llama-3.2-3b", "phi-3-mini"],
        help="Models to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads"
    )
    
    args = parser.parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🤗 Setting up Hugging Face...")
    setup_huggingface()
    
    # Get model configuration
    model_config = get_model_config()
    
    # Determine which models to download
    if "all" in args.models:
        models_to_download = list(model_config.keys())
    else:
        models_to_download = args.models
    
    print(f"\n📥 Downloading {len(models_to_download)} models...")
    print("=" * 50)
    
    # Download models
    success_count = 0
    for model_key in models_to_download:
        if model_key not in model_config:
            print(f"❌ Unknown model: {model_key}")
            continue
            
        config = model_config[model_key]
        print(f"\n📦 {config['description']}")
        
        if download_model(
            config["hf_name"], 
            config["path"], 
            args.cache_dir
        ):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"✅ Successfully downloaded {success_count}/{len(models_to_download)} models")
    
    if success_count < len(models_to_download):
        print("\n⚠️  Some downloads failed. Check:")
        print("1. Internet connection")
        print("2. Hugging Face authentication (for gated models)")
        print("3. Sufficient disk space")

if __name__ == "__main__":
    main()