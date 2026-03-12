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
# Download core models (LLaMA 3.2 + Phi-3)
python scripts/download_models.py

# Or download all models
python scripts/download_models.py --models all
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
  huggingface-cli login
  ```

- **Disk Space**: Models require ~10-50GB total storage
- **Memory**: 8GB+ RAM recommended for 3B models
- **GPU**: Optional but recommended for faster evaluation