#!/bin/bash
# CurveRL evaluation launcher. Reuses run_qwen3_training.sh in val-only mode.
#
# Usage:
#   EVAL_DATASET=aime25 CKPT_PATH=checkpoints/.../global_step_1000 bash run_qwen3_eva.sh
# To evaluate a local HF model directory (any dir with config.json):
#   EVAL_DATASET=aime25 CKPT_PATH=/path/to/local/hf_model bash run_qwen3_eva.sh
# To evaluate a Hugging Face Hub model directly, set MODEL_PATH instead and use
# run_qwen3_training.sh with VAL_ONLY=True.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CKPT_PATH:-}" && $# -gt 0 ]]; then
  CKPT_PATH="$1"; shift
fi
if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "CKPT_PATH is required (a global_step_* directory or an HF model dir)." >&2
  exit 1
fi
if [[ -z "${EVAL_DATASET:-}" ]]; then
  echo "EVAL_DATASET is required (e.g. aime25, math_500)." >&2
  exit 1
fi

resolve_project_path() {
  if [[ "$1" == /* || -e "$1" ]]; then
    printf '%s\n' "$1"
  elif [[ -e "${PROJECT_ROOT}/$1" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/$1"
  else
    printf '%s\n' "$1"
  fi
}

CKPT_DIR="$(resolve_project_path "${CKPT_PATH}")"
CKPT_DIR="${CKPT_DIR%/}"
[[ "$(basename -- "${CKPT_DIR}")" == "actor" ]] && CKPT_DIR="$(dirname -- "${CKPT_DIR}")"

# Two accepted layouts:
#   1. Trained checkpoint:  .../global_step_N  (must contain actor/huggingface/)
#   2. Local HF model dir:  any directory with config.json
# The training script saves a merged HF model at global_step_*/actor/huggingface/
# by default (CHECKPOINT_SAVE_CONTENTS includes hf_model), which is what we load
# from here — it sidesteps FSDP world_size matching during eval.
if [[ "$(basename -- "${CKPT_DIR}")" == global_step_* ]]; then
  HF_DIR="${CKPT_DIR}/actor/huggingface"
  if [[ -d "${HF_DIR}" ]]; then
    export MODEL_PATH="${HF_DIR}"
    export RESUME_MODE="${RESUME_MODE:-disable}"
  else
    echo "Expected merged HF model at ${HF_DIR} (training writes it when CHECKPOINT_SAVE_CONTENTS includes hf_model)." >&2
    exit 1
  fi
elif [[ -f "${CKPT_DIR}/config.json" ]]; then
  export MODEL_PATH="${CKPT_DIR}"
  export RESUME_MODE="${RESUME_MODE:-disable}"
else
  echo "CKPT_PATH must be a global_step_* directory or a local HF model dir with config.json." >&2
  echo "To evaluate a Hugging Face Hub model directly, set MODEL_PATH and call run_qwen3_training.sh with VAL_ONLY=True." >&2
  exit 1
fi

EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${PROJECT_ROOT}/data/${EVAL_DATASET}/test.parquet}"
[[ -f "${EVAL_DATASET_PATH}" ]] || { echo "EVAL_DATASET_PATH not found: ${EVAL_DATASET_PATH}" >&2; exit 1; }

CKPT_RUN_NAME="$(basename -- "$(dirname -- "${CKPT_DIR}")")"
CKPT_STEP_NAME="$(basename -- "${CKPT_DIR}")"

# Infer the estimator from the checkpoint run name unless caller forces it.
if [[ -z "${ADVANTAGE_ESTIMATOR:-}" ]]; then
  case "${CKPT_RUN_NAME}" in
    grpo_*)    ADVANTAGE_ESTIMATOR="grpo" ;;
    maxrl_*)   ADVANTAGE_ESTIMATOR="maxrl" ;;
    curverl_*) ADVANTAGE_ESTIMATOR="curverl" ;;
    *)         ADVANTAGE_ESTIMATOR="curverl" ;;
  esac
fi
export ADVANTAGE_ESTIMATOR

export NUM_PER_PROMPT_ROLLOUTS_VALIDATION="${NUM_PER_PROMPT_ROLLOUTS_VALIDATION:-2048}"
export VAL_GEN_BATCH_SIZE="${VAL_GEN_BATCH_SIZE:-256}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export VAL_ONLY="True"
export VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-True}"
export TEST_DATASET_PATH="['${EVAL_DATASET_PATH}']"
export PROJECT_NAME="${PROJECT_NAME:-CurveRL_Eval}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eva_${EVAL_DATASET}_${CKPT_RUN_NAME}_${CKPT_STEP_NAME}}"

echo "EVAL_DATASET=${EVAL_DATASET}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "MODEL_PATH=${MODEL_PATH:-<from-checkpoint>}"
echo "RESUME_MODE=${RESUME_MODE}"
echo "ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"

bash "${PROJECT_ROOT}/qwen3_experiments/run_qwen3_training.sh" "$@"
