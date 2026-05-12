# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

POLICY_LOSS_REGISTRY = {}


def register_policy_loss(name):
    def decorator(func):
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    MAXRL = "maxrl"
    CURVERL = "curverl"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert (
            kl_ctrl.horizon > 0
        ), f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.MAXRL)
def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        N = len(id2score[index[0]])  # number of responses per prompt

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)

            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2mean[index[i]] + epsilon)

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def _scores_to_successes(scores: torch.Tensor) -> torch.Tensor:
    """Map scalar sequence scores to success indicators in [0, 1]."""
    if torch.all((scores >= 0.0) & (scores <= 1.0)):
        return scores.to(dtype=torch.float32)
    return (scores > 0.0).to(dtype=torch.float32)


def _flatten_metric_vector(prefix: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}/{idx}": float(val)
        for idx, val in enumerate(values.detach().cpu().tolist())
    }


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _normalize_prompt_key(raw_key):
    if raw_key is None:
        return None
    if isinstance(raw_key, np.generic):
        return raw_key.item()
    if isinstance(raw_key, np.ndarray):
        if raw_key.ndim == 0:
            return raw_key.item()
        return tuple(raw_key.tolist())
    if isinstance(raw_key, list):
        return tuple(raw_key)
    return raw_key


def _prompt_keys_are_valid(group_prompt_keys: list[object]) -> bool:
    if not group_prompt_keys:
        return False
    normalized_keys = [_normalize_prompt_key(key) for key in group_prompt_keys]
    if any(key is None for key in normalized_keys):
        return False
    return len(set(normalized_keys)) > 1


def _collect_prompt_group_stats(
    successes: torch.Tensor,
    index: np.ndarray,
    prompt_keys: Optional[np.ndarray] = None,
) -> dict[str, object]:
    id2indices = defaultdict(list)
    ordered_ids = []
    for sample_idx in range(successes.shape[0]):
        uid = index[sample_idx]
        if uid not in id2indices:
            ordered_ids.append(uid)
        id2indices[uid].append(sample_idx)

    prompt_sample_indices = []
    prompt_success_counts = []
    prompt_trial_counts = []
    prompt_success_rates = []
    group_prompt_keys = []
    normalized_prompt_keys = None
    if prompt_keys is not None:
        normalized_prompt_keys = [_normalize_prompt_key(key) for key in prompt_keys]

    for uid in ordered_ids:
        sample_indices = id2indices[uid]
        prompt_sample_indices.append(sample_indices)
        prompt_success = successes[sample_indices]
        prompt_trial_count = float(len(sample_indices))
        prompt_success_count = prompt_success.sum()
        prompt_success_counts.append(prompt_success_count)
        prompt_trial_counts.append(prompt_success.new_tensor(prompt_trial_count))
        prompt_success_rates.append(prompt_success_count / max(prompt_trial_count, 1.0))

        if normalized_prompt_keys is not None:
            first_key = normalized_prompt_keys[sample_indices[0]]
            if all(normalized_prompt_keys[idx] == first_key for idx in sample_indices):
                group_prompt_keys.append(first_key)
            else:
                group_prompt_keys.append(None)

    if prompt_success_counts:
        prompt_success_counts = torch.stack(prompt_success_counts).to(
            dtype=torch.float32
        )
        prompt_trial_counts = torch.stack(prompt_trial_counts).to(dtype=torch.float32)
        prompt_success_rates = torch.stack(prompt_success_rates).to(dtype=torch.float32)
    else:
        prompt_success_counts = torch.empty(
            0, dtype=torch.float32, device=successes.device
        )
        prompt_trial_counts = torch.empty(
            0, dtype=torch.float32, device=successes.device
        )
        prompt_success_rates = torch.empty(
            0, dtype=torch.float32, device=successes.device
        )

    if normalized_prompt_keys is None:
        group_prompt_keys = [None] * len(ordered_ids)

    return {
        "group_ids": ordered_ids,
        "sample_indices": prompt_sample_indices,
        "success_counts": prompt_success_counts,
        "trial_counts": prompt_trial_counts,
        "success_rates": prompt_success_rates,
        "prompt_keys": group_prompt_keys,
        "stable_prompt_keys_valid": _prompt_keys_are_valid(group_prompt_keys),
    }


def _copy_curverl_state(state: Optional[dict]) -> dict[str, object]:
    """Deep-copy a CurveRL sliding-window state into a fresh dict."""
    if state is None:
        return {
            "pool_values": [],
            "pool_entry_batch_ids": [],
            "pool_batch_ids": [],
            "next_pool_batch_id": 0,
            "num_updates": 0,
        }
    pool_values = [float(v) for v in state.get("pool_values", [])]
    pool_entry_batch_ids = [int(b) for b in state.get("pool_entry_batch_ids", [])]
    if len(pool_entry_batch_ids) != len(pool_values):
        raise ValueError(
            "CurveRL state is malformed: pool_entry_batch_ids and pool_values must be aligned."
        )
    pool_batch_ids = [int(b) for b in state.get("pool_batch_ids", [])]
    return {
        "pool_values": pool_values,
        "pool_entry_batch_ids": pool_entry_batch_ids,
        "pool_batch_ids": pool_batch_ids,
        "next_pool_batch_id": int(state.get("next_pool_batch_id", 0)),
        "num_updates": int(state.get("num_updates", 0)),
    }


def _append_curverl_batch(
    state: dict[str, object],
    prompt_success_rates: torch.Tensor,
    pool_num: int,
) -> dict[str, object]:
    """Append a batch of in-(0,1) success rates to the sliding window; evict the
    oldest batch when more than `pool_num` batches are stored."""
    next_state = _copy_curverl_state(state)
    values_cpu = prompt_success_rates.detach().to(dtype=torch.float32, device="cpu")
    keep_mask = (values_cpu > 0.0) & (values_cpu < 1.0)

    current_batch_id = int(next_state["next_pool_batch_id"])
    next_state["next_pool_batch_id"] = current_batch_id + 1
    next_state["pool_batch_ids"].append(current_batch_id)

    for idx, value in enumerate(values_cpu.tolist()):
        if not bool(keep_mask[idx].item()):
            continue
        next_state["pool_values"].append(float(value))
        next_state["pool_entry_batch_ids"].append(current_batch_id)

    # `pool_num == 0` collapses the window to the current batch by evicting
    # everything; the next call falls back to "reference = current batch".
    while len(next_state["pool_batch_ids"]) > pool_num:
        evicted = next_state["pool_batch_ids"].pop(0)
        keep_idx = [
            i for i, b in enumerate(next_state["pool_entry_batch_ids"]) if b != evicted
        ]
        next_state["pool_values"] = [next_state["pool_values"][i] for i in keep_idx]
        next_state["pool_entry_batch_ids"] = [
            next_state["pool_entry_batch_ids"][i] for i in keep_idx
        ]

    next_state["num_updates"] += 1
    return next_state


def _compute_curverl_histogram(
    reference_values: torch.Tensor,
    num_bins: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Estimate the reference distribution F_ref, f_ref from a window of past
    pass rates. Returns mass (f_ref), cdf (F_ref), and raw weights f_ref / F_ref."""
    values = torch.clamp(
        reference_values.detach().to(dtype=torch.float32, device="cpu"),
        min=0.0,
        max=1.0,
    )
    if values.numel() == 0:
        counts = torch.zeros(num_bins, dtype=torch.float32)
    else:
        bin_indices = torch.floor(values * num_bins).to(dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)
        counts = torch.bincount(bin_indices, minlength=num_bins).to(dtype=torch.float32)
    mass = counts / torch.clamp(counts.sum(), min=epsilon)
    cdf = torch.cumsum(mass, dim=0)
    raw_weights = mass / torch.clamp(cdf, min=epsilon)
    return {"counts": counts, "mass": mass, "cdf": cdf, "raw_weights": raw_weights}


def _map_curverl_prompt_weights(
    prompt_values: torch.Tensor,
    bin_weights: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """Map per-prompt p̂ values to per-prompt weights via the precomputed bin weights."""
    values = torch.clamp(
        prompt_values.detach().to(dtype=torch.float32, device="cpu"), min=0.0, max=1.0
    )
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    bin_indices = torch.floor(values * num_bins).to(dtype=torch.long)
    bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)
    return bin_weights[bin_indices]


@register_adv_est(AdvantageEstimator.CURVERL)
def compute_curverl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    token_level_scores: Optional[torch.Tensor] = None,
    config=None,
    curverl_state: Optional[dict] = None,
    return_stats: bool = False,
    **kwargs,
):
    """Compute CurveRL prompt weights (Algorithm 1 in the paper).

    For each prompt in the current batch:
      1. Compute empirical pass rate p̂ from its rollouts.
      2. If p̂ ∈ (0, 1), look up its weight from the histogram-based ratio
         w(p̂) = f_ref(p̂) / F_ref(p̂), estimated over a sliding window of the
         last `curverl_pool_num` training batches' pass rates.
      3. Centered advantage per rollout: w(p̂) * (success_i - p̂).
      4. Append the batch's filtered p̂ to the window and evict the oldest batch
         if more than `curverl_pool_num` batches are stored.

    All-fail and all-success groups get advantage 0 and are excluded from the pool.
    """
    assert config is not None, "config is required for CurveRL"

    pool_num = int(config.get("curverl_pool_num", 10))
    if pool_num < 0:
        raise ValueError(f"curverl_pool_num must be >= 0, got {pool_num}")

    score_source = (
        token_level_scores if token_level_scores is not None else token_level_rewards
    )
    scores = score_source.sum(dim=-1)

    with torch.no_grad():
        successes = _scores_to_successes(scores)
        train_stats = _collect_prompt_group_stats(successes=successes, index=index)
        next_state = _copy_curverl_state(curverl_state)

        # Histogram resolution matches the discrete support of empirical pass
        # rates: with N rollouts per prompt, p̂ ∈ {0/N, 1/N, ..., N/N}.
        max_trials = (
            int(train_stats["trial_counts"].max().item())
            if train_stats["trial_counts"].numel() > 0
            else 1
        )
        num_bins = max(max_trials + 1, 2)

        if not train_stats["group_ids"]:
            advantages = torch.zeros_like(token_level_rewards)
            if return_stats:
                metric_dict = {
                    "algorithm/curverl/num_prompts_in_state": int(
                        len(next_state["pool_values"])
                    ),
                    "algorithm/curverl/num_batches_in_state": int(
                        len(next_state["pool_batch_ids"])
                    ),
                    "algorithm/curverl/pool_num": int(pool_num),
                    "algorithm/curverl/weight_mean": 1.0,
                }
                return advantages, advantages, next_state, metric_dict
            return advantages, advantages

        current_p_hat = train_stats["success_rates"].detach().cpu()
        # Reference distribution comes from the *previous* batches (lagged
        # window). On the very first step the window is empty, so fall back to
        # the current batch's filtered p̂.
        if next_state["pool_values"]:
            reference_values = torch.tensor(
                next_state["pool_values"], dtype=torch.float32
            )
        else:
            reference_values = current_p_hat[
                (current_p_hat > 0.0) & (current_p_hat < 1.0)
            ]

        histogram = _compute_curverl_histogram(
            reference_values=reference_values,
            num_bins=num_bins,
            epsilon=epsilon,
        )

        included_mask = (current_p_hat > 0.0) & (current_p_hat < 1.0)
        prompt_weights_cpu = torch.zeros_like(current_p_hat)
        if reference_values.numel() > 0 and included_mask.any():
            prompt_weights_cpu[included_mask] = _map_curverl_prompt_weights(
                prompt_values=current_p_hat[included_mask],
                bin_weights=histogram["raw_weights"],
                num_bins=num_bins,
            )

        prompt_weights = prompt_weights_cpu.to(
            device=successes.device, dtype=torch.float32
        )
        advantages = torch.zeros_like(successes)
        for prompt_idx, sample_indices in enumerate(train_stats["sample_indices"]):
            prompt_baseline = train_stats["success_rates"][prompt_idx]
            advantages[sample_indices] = prompt_weights[prompt_idx] * (
                successes[sample_indices] - prompt_baseline
            )

        advantages = (
            advantages.to(dtype=token_level_rewards.dtype).unsqueeze(-1) * response_mask
        )

        # Append current batch's filtered p̂ to the pool *after* the histogram
        # has been computed so the reference distribution remains a lagged one.
        next_state = _append_curverl_batch(
            state=next_state,
            prompt_success_rates=train_stats["success_rates"],
            pool_num=pool_num,
        )

        if return_stats:
            included_weights = (
                prompt_weights_cpu[included_mask]
                if included_mask.any()
                else torch.zeros(1, dtype=torch.float32)
            )
            metric_dict = {
                "algorithm/curverl/num_prompts_in_state": int(
                    len(next_state["pool_values"])
                ),
                "algorithm/curverl/num_batches_in_state": int(
                    len(next_state["pool_batch_ids"])
                ),
                "algorithm/curverl/pool_num": int(pool_num),
                "algorithm/curverl/weight_mean": float(included_weights.mean().item()),
                "algorithm/curverl/weight_max": float(included_weights.max().item()),
                "algorithm/curverl/nonempty_bins": int(
                    (histogram["counts"] > 0).sum().item()
                ),
            }
            metric_dict.update(
                _flatten_metric_vector("algorithm/curverl/pool_mass", histogram["mass"])
            )
            metric_dict.update(
                _flatten_metric_vector("algorithm/curverl/pool_cdf", histogram["cdf"])
            )
            return advantages, advantages, next_state, metric_dict

    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS
)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (
            (token_level_rewards * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_grpo_loss_objective(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_grpo_with_off_policy_distillation_loss_objective(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    off_policy_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        f" but got: {clip_ratio_c}."
    )

    # --- PPO part (same as before) ---
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # detach & (ideally) normalize advantages upstream
    advantages = advantages.detach()

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
        response_mask,
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # --- Off-policy SFT loss ---
    # off_policy_mask is 1 for tokens where you want SFT behaviour
    off_policy_mask = off_policy_mask.unsqueeze(1).bool()

    # Standard SFT / NLL: just maximize log_prob on target tokens
    sft_losses = -advantages * log_prob  # shape matches pg_losses
    sft_losses = sft_losses

    # --- Combine ---
    combined_losses = torch.where(
        off_policy_mask,  # if True -> SFT, else -> PPO
        sft_losses,
        pg_losses,
    )

    pg_loss = agg_loss(
        loss_mat=combined_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    off_policy_mask=None,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    loss_type: str = "grpo",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        loss_type (str, optional):
            Type of loss objective (SFT, GRPO etc.). Defaults to grpo
    """
    loss_calculation_kwargs = {
        "old_log_prob": old_log_prob,
        "log_prob": log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        "cliprange": cliprange,
        "cliprange_low": cliprange_low,
        "cliprange_high": cliprange_high,
        "clip_ratio_c": clip_ratio_c,
        "loss_agg_mode": loss_agg_mode,
    }

    if loss_type == "grpo":
        return compute_grpo_loss_objective(**loss_calculation_kwargs)

    elif loss_type == "empo":
        return compute_grpo_loss_objective(**loss_calculation_kwargs)

    elif loss_type == "grpo_with_filtered_sft":
        assert off_policy_mask is not None
        assert (
            off_policy_mask.ndim == 1
            and off_policy_mask.shape[0] == response_mask.shape[0]
        )

        loss_calculation_kwargs["off_policy_mask"] = off_policy_mask

        return compute_grpo_with_off_policy_distillation_loss_objective(
            **loss_calculation_kwargs,
        )

    else:
        raise ValueError(f"Given loss type {loss_type} is not supported yet.")


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    clip_cov_ratio = (
        config.policy_loss.clip_cov_ratio
        if config.policy_loss.clip_cov_ratio is not None
        else 0.0002
    )
    cliprange = config.clip_ratio
    cliprange_low = (
        config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    )
    cliprange_high = (
        config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    )
    clip_cov_ub = (
        config.policy_loss.clip_cov_ub
        if config.policy_loss.clip_cov_ub is not None
        else 5.0
    )
    clip_cov_lb = (
        config.policy_loss.clip_cov_lb
        if config.policy_loss.clip_cov_lb is not None
        else 1.0
    )

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    kl_cov_ratio = (
        config.policy_loss.kl_cov_ratio
        if config.policy_loss.kl_cov_ratio is not None
        else 0.0002
    )
    ppo_kl_coef = (
        config.policy_loss.ppo_kl_coef
        if config.policy_loss.ppo_kl_coef is not None
        else 1.0
    )

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (
            all_valid_logp - all_valid_logp.mean()
        )
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1],
                large_cov_idxs % advantages.shape[1],
            ]

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(
        loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(
        scores: torch.Tensor, reweight_method: str, weight_pow: float
    ) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where(
                (scores == max_score) | (scores == min_score), 1.0, 0.0
            )
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {
        key: tensor[sample_indices] for key, tensor in data.batch.items()
    }

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
