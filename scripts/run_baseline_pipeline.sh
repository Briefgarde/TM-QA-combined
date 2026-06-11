#!/bin/bash
# Submit from the project root: sbatch scripts/run_baseline_pipeline.sh
# For a debug run of N samples:  DEBUG_N=10 sbatch scripts/run_baseline_pipeline.sh
# The tmqa conda env must already exist.

#SBATCH --job-name=baseline_pipeline
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

REPO_DIR="$SLURM_SUBMIT_DIR"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tmqa

mkdir -p "${REPO_DIR}/logs"

echo "Host:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "DEBUG_N:   ${DEBUG_N:-None (full run)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${REPO_DIR}/evaluation"
/home/users/v/vollert/.conda/envs/tmqa/bin/python testBaselinePipeline.py
