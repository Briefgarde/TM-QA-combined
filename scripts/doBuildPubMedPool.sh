#!/bin/bash
# Submit from the project root: sbatch scripts/doBuildPubMedPool.sh
# The tmqa conda env must already exist.

#SBATCH --job-name=fetch_n_build_pool_pubMedQA
#SBATCH --partition=public-longrun-cpu
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

REPO_DIR="$SLURM_SUBMIT_DIR"

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tmqa

mkdir -p "${REPO_DIR}/logs"

echo "Host:      $(hostname)"
echo "Start:     $(date)"

cd "${REPO_DIR}/data/PubMedQA"
/home/users/v/vollert/.conda/envs/tmqa/bin/python buildingPubMedQAAbstract.py
