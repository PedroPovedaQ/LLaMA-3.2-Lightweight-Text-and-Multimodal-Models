#!/usr/bin/env python3
"""
Setup script for small model comparison project.
Creates necessary directories and validates environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected - CPU-only mode")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False

def create_directories():
    """Create necessary project directories."""
    project_root = Path(__file__).parent.parent
    directories = [
        "models/llama-3.2-1b",
        "models/llama-3.2-3b", 
        "models/phi-3-mini",
        "models/others",
        "benchmarks/datasets",
        "benchmarks/evaluation",
        "benchmarks/metrics",
        "results/raw",
        "results/processed",
        "results/reports",
        "notebooks/outputs",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")

def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Models and data
models/*/pytorch_model.bin
models/*/model.safetensors
models/*/*.bin
benchmarks/datasets/*
!benchmarks/datasets/.gitkeep

# Results
results/raw/*
results/processed/*
logs/*
!*/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter
.ipynb_checkpoints/
notebooks/outputs/*

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"""
    
    project_root = Path(__file__).parent.parent
    with open(project_root / ".gitignore", "w") as f:
        f.write(gitignore_content)
    print("📄 Created .gitignore")

def create_gitkeep_files():
    """Create .gitkeep files in empty directories."""
    project_root = Path(__file__).parent.parent
    directories = [
        "benchmarks/datasets",
        "results/raw", 
        "results/processed",
        "results/reports",
        "logs"
    ]
    
    for directory in directories:
        gitkeep_path = project_root / directory / ".gitkeep"
        gitkeep_path.touch()

def main():
    print("🚀 Setting up Small Model Comparison Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create project structure
    print("\n📁 Creating project directories...")
    create_directories()
    create_gitkeep_files()
    
    # Create .gitignore
    print("\n📄 Creating configuration files...")
    create_gitignore()
    
    # Check GPU
    print("\n🔍 Checking hardware...")
    check_gpu()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Download models: python scripts/download_models.py")
    print("3. Run benchmarks: python scripts/run_benchmarks.py")

if __name__ == "__main__":
    main()