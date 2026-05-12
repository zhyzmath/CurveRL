#!/bin/bash
# CurveRL training launcher for Qwen3-Base on math reasoning (POLARIS).
# Override defaults via environment variables, e.g.
#   ADVANTAGE_ESTIMATOR=curverl CURVERL_POOL_NUM=10 bash run_qwen3_training.sh
# Supports ADVANTAGE_ESTIMATOR in {curverl, grpo, maxrl}.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ============ Data ============
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${PROJECT_ROOT}/data/polaris/train.parquet}"
MATH500_VAL_DATA="${MATH500_VAL_DATA:-${PROJECT_ROOT}/data/math_500/test.parquet}"
TEST_DATASET_PATH="${TEST_DATASET_PATH:-['${MATH500_VAL_DATA}']}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
VAL_ONLY="${VAL_ONLY:-False}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-}"

# ============ Training hyperparameters (paper defaults) ============
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-curverl}"

FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
PER_GPU_MINI_BATCH_SIZE="${PER_GPU_MINI_BATCH_SIZE:-16}"
NUM_PER_PROMPT_ROLLOUTS="${NUM_PER_PROMPT_ROLLOUTS:-8}"
NUM_PER_PROMPT_ROLLOUTS_VALIDATION="${NUM_PER_PROMPT_ROLLOUTS_VALIDATION:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
GRAD_CLIP="${GRAD_CLIP:-0.3}"
KL_COEFF="${KL_COEFF:-0.0}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

# ============ Rollout / vLLM ============
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
# When set, validation generation runs in `ceil(val_n / VAL_GEN_BATCH_SIZE)` rounds.
VAL_GEN_BATCH_SIZE="${VAL_GEN_BATCH_SIZE:-}"

CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}"

# ============ Reward manager ============
REWARD_MANAGER="${REWARD_MANAGER:-multi_thread}"
REWARD_MANAGER_ARGS=()
if [[ "${REWARD_MANAGER}" == "multi_thread" ]]; then
  REWARD_NUM_ACTORS="${REWARD_NUM_ACTORS:-16}"
  REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-8}"
  REWARD_IN_FLIGHT_BATCHES_PER_ACTOR="${REWARD_IN_FLIGHT_BATCHES_PER_ACTOR:-4}"
  REWARD_PER_ITEM_TIMEOUT_S="${REWARD_PER_ITEM_TIMEOUT_S:-1}"
  REWARD_PER_BATCH_TIMEOUT_S="${REWARD_PER_BATCH_TIMEOUT_S:-10}"
  REWARD_POLL_INTERVAL_S="${REWARD_POLL_INTERVAL_S:-0.5}"
  REWARD_TIMEOUT_SCORE="${REWARD_TIMEOUT_SCORE:-0.0}"
  REWARD_MANAGER_ARGS=(
    "+reward_model.reward_kwargs.num_reward_actors=${REWARD_NUM_ACTORS}"
    "+reward_model.reward_kwargs.batch_size=${REWARD_BATCH_SIZE}"
    "+reward_model.reward_kwargs.in_flight_batches_per_actor=${REWARD_IN_FLIGHT_BATCHES_PER_ACTOR}"
    "+reward_model.reward_kwargs.per_item_timeout_s=${REWARD_PER_ITEM_TIMEOUT_S}"
    "+reward_model.reward_kwargs.per_batch_timeout_s=${REWARD_PER_BATCH_TIMEOUT_S}"
    "+reward_model.reward_kwargs.poll_interval_s=${REWARD_POLL_INTERVAL_S}"
    "+reward_model.reward_kwargs.timeout_score=${REWARD_TIMEOUT_SCORE}"
  )
fi

# ============ CurveRL-specific (single knob: t_0 in Algorithm 1) ============
CURVERL_ARGS=()
if [[ "${ADVANTAGE_ESTIMATOR}" == "curverl" ]]; then
  CURVERL_POOL_NUM="${CURVERL_POOL_NUM:-10}"
  CURVERL_ARGS=("algorithm.curverl_pool_num=${CURVERL_POOL_NUM}")
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}_poolnum${CURVERL_POOL_NUM}_rollout${NUM_PER_PROMPT_ROLLOUTS}}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}_rollout${NUM_PER_PROMPT_ROLLOUTS}}"
fi

PROJECT_NAME="${PROJECT_NAME:-CurveRL_Qwen3}"
CHECKPOINT_SAVE_PATH="${CHECKPOINT_SAVE_PATH:-${CHECKPOINT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

# ============ Device / Ray detection ============
detect_nnodes() {
  if [[ -n "${NNODES:-}" ]]; then :;
  elif [[ -n "${SLURM_NNODES:-}" ]]; then NNODES="${SLURM_NNODES}";
  else NNODES=1; fi
  export NNODES
}

detect_n_gpus_per_node() {
  if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then :;
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _devs <<< "${CUDA_VISIBLE_DEVICES}"
    N_GPUS_PER_NODE="${#_devs[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS_PER_NODE="$(nvidia-smi --list-gpus | wc -l)"
  else
    echo "Unable to detect GPU count: set N_GPUS_PER_NODE." >&2
    exit 1
  fi
  export N_GPUS_PER_NODE
}

detect_ray_cpus_per_node() {
  if [[ -n "${RAY_NUM_CPUS_PER_NODE:-}" ]]; then :;
  else
    local cap avail
    avail="$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
    cap="$((N_GPUS_PER_NODE * 16))"
    if (( avail > cap )); then RAY_NUM_CPUS_PER_NODE="${cap}"; else RAY_NUM_CPUS_PER_NODE="${avail}"; fi
  fi
  export RAY_NUM_CPUS_PER_NODE
}

detect_nnodes
detect_n_gpus_per_node
detect_ray_cpus_per_node

JOB_ID="${SLURM_JOB_ID:-$$}"
RAY_DIR="${RAY_DIR:-/tmp/${USER:-curverl}/${JOB_ID}_qwen3_${ADVANTAGE_ESTIMATOR}}"
mkdir -p "${RAY_DIR}" "${CHECKPOINT_SAVE_PATH}"

# Multi-node Ray: expect `ip_head` or `RAY_ADDRESS` (host:port) set by the launcher.
if (( NNODES > 1 )); then
  if [[ -n "${ip_head:-}" ]]; then
    export RAY_ADDRESS="${RAY_ADDRESS:-${ip_head}}"
  elif [[ -n "${RAY_ADDRESS:-}" ]]; then
    export ip_head="${RAY_ADDRESS}"
  else
    echo "NNODES=${NNODES} requires an existing Ray cluster. Set ip_head or RAY_ADDRESS." >&2
    exit 1
  fi
else
  unset RAY_ADDRESS ip_head 2>/dev/null || true
fi

# ============ Environment ============
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-AUTO}"
export SEED="${SEED:-79}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"
export RAY_DISABLE_IMPORT_WARNING="${RAY_DISABLE_IMPORT_WARNING:-1}"
# Set WANDB_ENTITY in your shell or override below to your wandb entity.
export WANDB_ENTITY="${WANDB_ENTITY:-WANDB_ENTITY}"

if (( NNODES == 1 )); then
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

# ============ Validation ============
if [[ ! -f "${TRAIN_DATASET_PATH}" ]]; then
  echo "TRAIN_DATASET_PATH not found: ${TRAIN_DATASET_PATH}" >&2
  exit 1
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "NNODES=${NNODES}  N_GPUS_PER_NODE=${N_GPUS_PER_NODE}  RAY_NUM_CPUS_PER_NODE=${RAY_NUM_CPUS_PER_NODE}"
if [[ "${ADVANTAGE_ESTIMATOR}" == "curverl" ]]; then
  echo "CURVERL_POOL_NUM=${CURVERL_POOL_NUM}"
fi

# ============ Training ============
CMD=(
  python3
  -W ignore
  -m verl.trainer.main_ppo
  "algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR}"
  "data.train_files=${TRAIN_DATASET_PATH}"
  "data.val_files=${TEST_DATASET_PATH}"
  "data.train_batch_size=${FULL_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PER_GPU_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEFF}"
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
  "actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-False}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}"
  "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PER_GPU_MINI_BATCH_SIZE}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${NUM_PER_PROMPT_ROLLOUTS}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PER_GPU_MINI_BATCH_SIZE}"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.rollout.val_kwargs.n=${NUM_PER_PROMPT_ROLLOUTS_VALIDATION}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.6"
  "actor_rollout_ref.rollout.val_kwargs.top_p=0.95"
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
  "actor_rollout_ref.rollout.multi_turn.enable=False"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_penalty=low_var_kl"
  "algorithm.kl_ctrl.kl_coef=${KL_COEFF}"
  "reward_model.reward_manager=${REWARD_MANAGER}"
  "trainer.balance_batch=True"
  "trainer.critic_warmup=0"
  "trainer.val_before_train=True"
  "trainer.val_only=${VAL_ONLY}"
  "trainer.val_on_last_step=True"
  "trainer.logger=['console','wandb']"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${CHECKPOINT_SAVE_PATH}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=50"
  "trainer.max_actor_ckpt_to_keep=400"
  "trainer.max_critic_ckpt_to_keep=400"
  "trainer.test_freq=50"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_SAVE_CONTENTS}"
  "ray_init.ray_dir=${RAY_DIR}"
  "ray_init.num_cpus=${RAY_NUM_CPUS_PER_NODE}"
)

[[ -n "${RESUME_FROM_PATH}" ]] && CMD+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
[[ -n "${VAL_BATCH_SIZE}" ]] && CMD+=("data.val_batch_size=${VAL_BATCH_SIZE}")
[[ -n "${VALIDATION_SHUFFLE}" ]] && CMD+=("data.validation_shuffle=${VALIDATION_SHUFFLE}")
[[ -n "${ROLLOUT_MAX_NUM_SEQS}" ]] && CMD+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}")
[[ -n "${VAL_GEN_BATCH_SIZE}" ]] && CMD+=("actor_rollout_ref.rollout.val_kwargs.gen_batch_size=${VAL_GEN_BATCH_SIZE}")
[[ ${#CURVERL_ARGS[@]} -gt 0 ]] && CMD+=("${CURVERL_ARGS[@]}")
[[ ${#REWARD_MANAGER_ARGS[@]} -gt 0 ]] && CMD+=("${REWARD_MANAGER_ARGS[@]}")

"${CMD[@]}" "$@"
