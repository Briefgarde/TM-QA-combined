#!/bin/bash
# Generic pipeline runner for evaluation/testingScript.py (reranking + generation).
#
# Submit from the project root:   sbatch scripts/run_testing_pipeline.sh
# Every variable below can be overridden at submit time (sbatch forwards your env), e.g.:
#   DEBUG_N=10 sbatch scripts/run_testing_pipeline.sh
#   MODE_CONTEXT_ASSEMBLING=full TEST_RERANKER=false sbatch scripts/run_testing_pipeline.sh
#   DATASET=PubMedQA-oracle POOL_PATH=pubmedqa_oracle_pool_fixed.json TEST_RERANKER=false \
#       sbatch scripts/run_testing_pipeline.sh
#   DATASET=PubMedQA-rag POOL_PATH=../data/PubMedQA/pubmedqa_rag_pool_fixed.json TEST_RERANKER=false \
#       sbatch scripts/run_testing_pipeline.sh
# The tmqa conda env must already exist.

#SBATCH --job-name=testing_pipeline
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# Absolute paths so logs always land in the project-root logs/ folder,
# whatever directory sbatch is called from (SBATCH lines can't use variables).
#SBATCH --output=/home/users/v/vollert/TM-QA-combined/logs/%x_%j.out
#SBATCH --error=/home/users/v/vollert/TM-QA-combined/logs/%x_%j.err

REPO_DIR="/home/users/v/vollert/TM-QA-combined"
PYTHON_BIN="/home/users/v/vollert/.conda/envs/tmqa/bin/python"

# ============================================================
# Configuration. Pattern: export VAR="${VAR:-default}"
# -> uses the value passed at submit time if any, otherwise the default.
# Booleans accept true/false, 1/0, yes/no (anything else makes the Python script stop).
# Relative paths are relative to evaluation/ (the script is run from there).
# ============================================================

# ---------- General / data ----------
export DEBUG_N="${DEBUG_N:-}"                                  # empty = full run, N = first N queries only
export USE_PRECOMPUTE_FILE="${USE_PRECOMPUTE_FILE:-true}"      # must stay true, testingScript only loads pools
export POOL_PATH="${POOL_PATH:-test_pool.json}"                # test_pool.json | pubmedqa_oracle_pool_fixed.json | ../data/PubMedQA/pubmedqa_rag_pool_fixed.json
export DATASET="${DATASET:-BioASQ-training14b}"                # label saved in metadata: BioASQ-training14b | PubMedQA-oracle | PubMedQA-rag
export SPLIT="${SPLIT:-test}"                                  # label saved in metadata and filename

# ---------- Reranker ----------
# Reranker model: local path or HF name (read by testingScript.py as MODEL_NAME_RERANKER).
export MODEL_NAME_RERANKER="${MODEL_NAME_RERANKER:-../models/finetuning/finetuningResult/run_lr1e-05_epoch5warm_0-1_margin1_maxPos10_maxRneg3_maxHneg2/best_model}"
#   base model alternative: ncbi/MedCPT-Cross-Encoder
export TEST_RERANKER="${TEST_RERANKER:-true}"                  # true only for BioASQ; PubMedQA pools have no labels (all 0.0)
export OUTPUT_DIR_RERANKER_RESULT="${OUTPUT_DIR_RERANKER_RESULT:-result/${DATASET}/reranker}"  # required if TEST_RERANKER=true

# ---------- Generation ----------
export MODEL_NAME_GEN="${MODEL_NAME_GEN:-stanford-crfm/BioMedLM}"
export MODE_CONTEXT_ASSEMBLING="${MODE_CONTEXT_ASSEMBLING:-reranked}"  # reranked (top-k 1,2,3,5) | full
export MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-100}"                   # tokens reserved for generation
export MAX_CONTEXT_TOKEN="${MAX_CONTEXT_TOKEN:-1024}"          # model's TOTAL window; context budget = this - MAX_NEW_TOKEN
# $'...' is required so that \n becomes a real newline (plain "..." would pass a literal backslash-n).
# To override at submit time, also use $'...': PROMPT_TEMPLATE=$'Context: {context}\nQ: {query}\nA:' sbatch ...
DEFAULT_PROMPT_TEMPLATE=$'Context: {context}\nQuestion: {query}\nAnswer:'
export PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$DEFAULT_PROMPT_TEMPLATE}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
export USE_NEWLINE_STOP="${USE_NEWLINE_STOP:-true}"
export BATCH_SIZE_GEN="${BATCH_SIZE_GEN:-4}"
export BERTSCORE_MODEL_ROBERTA="${BERTSCORE_MODEL_ROBERTA:-roberta-large}"
export BERTSCORE_MODEL_BIOMEDICAL="${BERTSCORE_MODEL_BIOMEDICAL:-microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract}"
export OUTPUT_DIR_GEN="${OUTPUT_DIR_GEN:-result/${DATASET}/generation}"

# ============================================================

module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tmqa

mkdir -p "${REPO_DIR}/logs"

echo "Host:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "----- Configuration -----"
echo "DEBUG_N:                    ${DEBUG_N:-None (full run)}"
echo "USE_PRECOMPUTE_FILE:        $USE_PRECOMPUTE_FILE"
echo "POOL_PATH:                  $POOL_PATH"
echo "DATASET:                    $DATASET"
echo "SPLIT:                      $SPLIT"
echo "MODEL_NAME_RERANKER:        $MODEL_NAME_RERANKER"
echo "TEST_RERANKER:              $TEST_RERANKER"
echo "OUTPUT_DIR_RERANKER_RESULT: $OUTPUT_DIR_RERANKER_RESULT"
echo "MODEL_NAME_GEN:             $MODEL_NAME_GEN"
echo "MODE_CONTEXT_ASSEMBLING:    $MODE_CONTEXT_ASSEMBLING"
echo "MAX_NEW_TOKEN:              $MAX_NEW_TOKEN"
echo "MAX_CONTEXT_TOKEN:          $MAX_CONTEXT_TOKEN"
echo "PROMPT_TEMPLATE:            $PROMPT_TEMPLATE"
echo "REPETITION_PENALTY:         $REPETITION_PENALTY"
echo "USE_NEWLINE_STOP:           $USE_NEWLINE_STOP"
echo "BATCH_SIZE_GEN:             $BATCH_SIZE_GEN"
echo "BERTSCORE_MODEL_ROBERTA:    $BERTSCORE_MODEL_ROBERTA"
echo "BERTSCORE_MODEL_BIOMEDICAL: $BERTSCORE_MODEL_BIOMEDICAL"
echo "OUTPUT_DIR_GEN:             $OUTPUT_DIR_GEN"
echo "-------------------------"

cd "${REPO_DIR}/evaluation"

# Fail fast instead of after loading the reranker.
if [ ! -f "$POOL_PATH" ]; then
    echo "ERROR: POOL_PATH '$POOL_PATH' not found (relative to ${REPO_DIR}/evaluation)" >&2
    exit 1
fi

"$PYTHON_BIN" testingScript.py
