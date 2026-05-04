#!/bin/bash
#SBATCH --job-name=llm_bench
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_NID@ucf.edu  # Uncomment and set your email

# =============================================================================
# Lightweight Model Benchmark Suite — UCF Newton
#
# Evaluates: Llama 3.2 1B, Llama 3.2 3B, Phi-3.5-mini, Gemma 2 2B
# Benchmarks: MMLU (5-shot), Open-rewrite, TLDR9+, IFEval
#
# All 4 models are small (<8GB VRAM), run sequentially on a single GPU.
# Estimated total time: ~8-10 hours.
#
# Prerequisites:
#   - conda env 'llama_eval' created with: conda create -n llama_eval python=3.10
#   - pip install -r requirements.txt
#   - pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   - hf auth login (with a valid Hugging Face token)
# =============================================================================

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

# ---- Activate environment ----
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate llama_eval
export OPENBLAS_NUM_THREADS=1

# ---- Cache directory ----
if [ -d "/scratch/$USER" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" logs

cd ~/llama32_eval

# =============================================================================
# Run each model on all benchmarks
# =============================================================================

echo ""
echo "===== MODEL 1/4: Llama 3.2 1B ====="
python run_all.py --model llama-3.2-1b

echo ""
echo "===== MODEL 2/4: Llama 3.2 3B ====="
python run_all.py --model llama-3.2-3b

echo ""
echo "===== MODEL 3/4: Phi-3.5-mini ====="
python run_all.py --model phi-3.5-mini

echo ""
echo "===== MODEL 4/4: Gemma 2 2B ====="
python run_all.py --model gemma-2-2b

# =============================================================================
# Generate comparison
# =============================================================================
echo ""
echo "===== GENERATING COMPARISON ====="
python compare_results.py \
    --results results/llama-3.2-1b results/llama-3.2-3b \
              results/phi-3.5-mini results/gemma-2-2b \
    --output_dir comparison_output

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Results in: results/ and comparison_output/"
echo "=========================================="
