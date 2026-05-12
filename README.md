# CurveRL

Official implementation of **CurveRL: Principled Distribution-Aware Context Reweighting for LLM Reasoning**.

CurveRL performs prompt reweighting in RLVR as **context-distribution control**. Instead of weighting each prompt by a pointwise transformation of its empirical pass rate $\hat{p}$, CurveRL uses the prompt's position in the *evolving distribution* of pass rates over a sliding window of past training batches. The per-prompt weight is

$$w_t(\hat{p}) = \frac{\hat{f}_{\mathrm{ref}}(\hat{p})}{\hat{F}_{\mathrm{ref}}(\hat{p})}$$

where $\hat{f}_\mathrm{ref}$ and $\hat{F}_\mathrm{ref}$ are the density and CDF of pass rates estimated from the last $t_0$ training batches. See Algorithm 1 in the paper for the full update.

The implementation is built on top of [`verl`](https://github.com/volcengine/verl) (Ray + Hydra + FSDP + vLLM) and [`maxrl`](https://github.com/tajwarfahim/maxrl).

## Repository layout

- `verl/` — the `verl` training stack (Ray driver, FSDP workers, rollout engines, dataset/reward managers).
- `verl/trainer/ppo/core_algos.py` — advantage estimators and losses. `AdvantageEstimator.CURVERL` and `compute_curverl_outcome_advantage` live here.
- `verl/trainer/ppo/ray_trainer.py` — main `RayPPOTrainer` loop, threads the CurveRL sliding-window state across iterations and checkpoints (`curverl_state.pt`).
- `verl/trainer/config/ppo_trainer.yaml` — canonical Hydra config.
- `qwen3_experiments/run_qwen3_training.sh` — single training launcher.
- `qwen3_experiments/run_qwen3_eva.sh` — single evaluation launcher.
- `examples/curverl_data_preprocess/` — preprocessors for POLARIS-53K (training) and the eight math reasoning benchmarks reported in the paper (AIME 2025, BeyondAIME, HMMT 02/25, HMMT 02/26, MATH-500, BRUMO 2025, HMMT 11/25, Minerva).

## Installation

Paper experiments use Python 3.10, CUDA 12.8, and 8x NVIDIA B200. Other recent CUDA / Ampere or Hopper GPUs should also work, you may need to relax some version pins in `requirements.txt`.

```bash
conda create -n curverl python=3.10 -y
conda activate curverl

# Install runtime dependencies first (vLLM, FlashAttention 2, Ray, etc.).
pip install -r requirements.txt

# Install this repository in editable mode.
pip install --no-deps -e .
```

For SGLang as the rollout backend instead of vLLM, add `pip install -r requirements_sglang.txt`. For Ascend NPU, see `requirements-npu.txt`.

## Data preparation

The paper trains on POLARIS-53K and evaluates on eight math reasoning benchmarks. Generate the parquet files once:

```bash
# Training data
python examples/curverl_data_preprocess/polaris.py --local_dir data/polaris

# Five benchmarks reported in the main paper
python examples/curverl_data_preprocess/aime25.py     --local_dir data/aime25
python examples/curverl_data_preprocess/beyondaime.py --local_dir data/beyondaime
python examples/curverl_data_preprocess/hmmt2502.py   --local_dir data/hmmt2502
python examples/curverl_data_preprocess/hmmt2602.py   --local_dir data/hmmt2602
python examples/curverl_data_preprocess/math_500.py   --local_dir data/math_500

# Three additional benchmarks reported in the appendix
python examples/curverl_data_preprocess/brumo25.py    --local_dir data/brumo25
python examples/curverl_data_preprocess/hmmt2511.py   --local_dir data/hmmt2511
python examples/curverl_data_preprocess/minerva.py    --local_dir data/minerva
```

## Training

The launcher reads all knobs from environment variables and dispatches a single `verl.trainer.main_ppo` job.

### CurveRL ($t_0 = 10$)

```bash
ADVANTAGE_ESTIMATOR=curverl \
CURVERL_POOL_NUM=10 \
MODEL_PATH=Qwen/Qwen3-1.7B-Base \
bash qwen3_experiments/run_qwen3_training.sh
```

For Qwen3-4B-Base, set `MODEL_PATH=Qwen/Qwen3-4B-Base`.

### Baselines

```bash
# GRPO
ADVANTAGE_ESTIMATOR=grpo \
MODEL_PATH=Qwen/Qwen3-1.7B-Base \
bash qwen3_experiments/run_qwen3_training.sh

# MaxRL
ADVANTAGE_ESTIMATOR=maxrl \
MODEL_PATH=Qwen/Qwen3-1.7B-Base \
bash qwen3_experiments/run_qwen3_training.sh
```

## Evaluation

The same launcher runs in evaluation mode:

```bash
EVAL_DATASET=aime25 \
CKPT_PATH=checkpoints/CurveRL_Qwen3/curverl_Qwen3-1.7B-Base_poolnum10_rollout8/global_step_1000 \
bash qwen3_experiments/run_qwen3_eva.sh
```

`CKPT_PATH` may be either a `global_step_*` checkpoint directory (which must
contain `actor/huggingface/`, written by default during training) or any local
HuggingFace model directory. To evaluate a Hub model directly, set `MODEL_PATH`
and call `qwen3_experiments/run_qwen3_training.sh` with `VAL_ONLY=True`.

Defaults match the paper:

- `NUM_PER_PROMPT_ROLLOUTS_VALIDATION=2048` for 1.7B, set to `1024` for 4B.
- `temperature=0.6`, `top_p=0.95`.

Useful overrides for managing rollout pressure during large pass@N evaluations:

- `NUM_PER_PROMPT_ROLLOUTS_VALIDATION` — total rollouts per prompt.
- `VAL_GEN_BATCH_SIZE` — split each prompt's rollouts into `ceil(N / VAL_GEN_BATCH_SIZE)` rounds.
- `ROLLOUT_MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `MAX_MODEL_LEN` — vLLM scheduling limits.

## CurveRL hyperparameter

The CurveRL update is governed by exactly **one** hyperparameter, the sliding-window size $t_0$:

| Setting | shell env | Hydra |
|---|---|---|
| $t_0$ (number of past training batches kept in the window) | `CURVERL_POOL_NUM` | `algorithm.curverl_pool_num` |

`CURVERL_POOL_NUM=10` is the paper default. `CURVERL_POOL_NUM=0` falls back to using only the current batch as the reference distribution.

Other implementation choices are fixed in code to match the paper:

- `num_bins = N + 1` (histogram resolution matches the discrete support of $\hat{p}$).
- All-fail ($\hat{p} = 0$) and all-success ($\hat{p} = 1$) groups are excluded from both the active reweighting set and the sliding pool.
- The weight is the raw ratio $\hat{f}_\mathrm{ref}(\hat{p}) / \hat{F}_\mathrm{ref}(\hat{p})$.

## Citation

```bibtex
@inproceedings{curverl2026,
  title     = {CurveRL: Principled Distribution-Aware Context Reweighting for LLM Reasoning},
  author    = {Anonymous},
  booktitle = {NeurIPS},
  year      = {2026},
}
```

## License

Apache License 2.0, see `LICENSE`. This repository extends [`verl`](https://github.com/volcengine/verl) (also Apache 2.0).
