# Quick Setup Guide

## 1. Initialize Project

```bash
# Run setup script
python scripts/setup.py

# Install dependencies  
pip install -r requirements.txt
```

## 2. Set up GitHub Repository

```bash
# Add files to git
git add .
git commit -m "Initial project setup for small model comparison"

# Create GitHub repo (replace with your repo URL)
git remote add origin https://github.com/yourusername/small-model-comparison.git
git branch -M main
git push -u origin main
```

## 3. Download Models

```bash
# Default path: llama-cli provider (downloads LLaMA 3.2 1B + 3B)
python scripts/download_models.py

# Add Phi-3-mini for comparison
python scripts/download_models.py --provider hf --models phi-3-mini

# Or download all configured models
python scripts/download_models.py --models all

# Pull core models from Ollama (reliable mirror path, no HF gate needed)
python scripts/download_models.py --provider ollama --models llama-3.2-1b llama-3.2-3b phi-3-mini

# Download Llama checkpoints via Meta's llama-model CLI (HF-backed source)
python scripts/download_models.py --provider llama-cli --models llama-3.2-1b llama-3.2-3b --llama-source huggingface

# Download from Meta signed URL (no HF dependency for model download)
python scripts/download_models.py --provider llama-cli --models llama-3.2-1b --llama-source meta --meta-url 'https://...llamameta.net/*?...'
```

## 4. Run Initial Tests

```bash
# Test model loading
python -c "from transformers import AutoTokenizer; print('✅ Transformers working')"

# Check GPU availability  
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## 5. Start Development

```bash
# Open notebooks
jupyter notebook notebooks/

# Or start with model comparison notebook
jupyter notebook notebooks/model_comparison.ipynb
```

## Important Notes

- **Hugging Face Login**: Some models require authentication
  ```bash
  hf auth login
  ```

- **Ollama Mirror Option**: install Ollama first if using `--provider ollama`
  ```bash
  ollama --version
  ```

- **Llama CLI Option**: install `llama-model` first if using `--provider llama-cli`
  ```bash
  python -m pip install -U llama-models
  llama-model --help
  ```

- **Current Benchmark Loader Note**: Transformers-based benchmark scripts expect HF-formatted weights (`model.safetensors` / `pytorch_model.bin`).

- **Disk Space**: Models require ~10-50GB total storage
- **Memory**: 8GB+ RAM recommended for 3B models
- **GPU**: Optional but recommended for faster evaluation
