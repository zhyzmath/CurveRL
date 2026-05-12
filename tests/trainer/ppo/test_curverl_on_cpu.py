# Copyright 2026 CurveRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU tests for the CurveRL advantage estimator (Algorithm 1 in the paper)."""

import sys
import types

import numpy as np
import torch


def _install_minimal_verl_import_stubs() -> None:
    if "ray" not in sys.modules:
        ray = types.ModuleType("ray")

        class ObjectRef:  # noqa: D401
            pass

        ray.ObjectRef = ObjectRef
        ray.get = lambda x: x
        sys.modules["ray"] = ray

    if "tensordict" not in sys.modules:
        tensordict = types.ModuleType("tensordict")

        class _LazyLegacy:
            def set(self):
                return None

        class TensorDict(dict):
            pass

        tensordict.set_lazy_legacy = lambda enabled: _LazyLegacy()
        tensordict.TensorDict = TensorDict
        tensordict.__version__ = "0.5.0"
        sys.modules["tensordict"] = tensordict


_install_minimal_verl_import_stubs()

from verl.trainer.ppo.core_algos import (  # noqa: E402
    _append_curverl_batch,
    _compute_curverl_histogram,
    _copy_curverl_state,
    compute_curverl_outcome_advantage,
)


def _config(pool_num: int = 10) -> dict:
    return {"curverl_pool_num": pool_num}


def _make_batch(p_hats: list[float], n_rollouts: int):
    """Build a synthetic (rewards, mask, index) batch where each prompt's
    rollouts sum to exactly p̂ * N successes."""
    rewards = []
    index = []
    for prompt_idx, p_hat in enumerate(p_hats):
        successes = round(p_hat * n_rollouts)
        for i in range(n_rollouts):
            rewards.append(1.0 if i < successes else 0.0)
            index.append(f"uid-{prompt_idx}")
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    response_mask = torch.ones_like(rewards_tensor)
    return rewards_tensor, response_mask, np.array(index, dtype=object)


def test_all_fail_or_all_success_groups_get_zero_advantage():
    rewards, mask, idx = _make_batch([0.0, 1.0, 0.5], n_rollouts=4)
    advantages, _, _, metrics = compute_curverl_outcome_advantage(
        token_level_rewards=rewards,
        token_level_scores=rewards,
        response_mask=mask,
        index=idx,
        config=_config(pool_num=0),
        return_stats=True,
    )
    # Groups 0 (all-fail) and 1 (all-success) -> zero advantage.
    assert torch.allclose(advantages[0:4], torch.zeros(4, 1))
    assert torch.allclose(advantages[4:8], torch.zeros(4, 1))
    # Group 2 (p̂ = 0.5, all alone in window) -> single-bin reference, weight 1.
    expected_group2 = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]], dtype=torch.float32)
    assert torch.allclose(advantages[8:12], expected_group2, atol=1e-6)
    # With pool_num=0 the pool is fully evicted after the update.
    assert metrics["algorithm/curverl/num_prompts_in_state"] == 0


def test_single_bin_weight_is_one():
    """When all in-(0,1) prompts share a single histogram bin, the f_ref/F_ref
    weight at that bin is 1 and the advantage is just (success - p̂)."""
    # Two prompts with the same p̂ -> both land in the same histogram bin.
    rewards, mask, idx = _make_batch([0.5, 0.5], n_rollouts=4)
    advantages, _, _, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards,
        token_level_scores=rewards,
        response_mask=mask,
        index=idx,
        config=_config(pool_num=0),
        return_stats=True,
    )
    expected = torch.tensor(
        [[0.5], [0.5], [-0.5], [-0.5]] * 2,
        dtype=torch.float32,
    )
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_two_bin_weight_ratio():
    """Two equally-populated bins produce ratios mass/cdf = [1.0, 0.5]."""
    rewards, mask, idx = _make_batch([0.25, 0.75], n_rollouts=4)
    advantages, _, _, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards,
        token_level_scores=rewards,
        response_mask=mask,
        index=idx,
        config=_config(pool_num=0),
        return_stats=True,
    )
    # Bin 1 weight: mass=0.5/cdf=0.5 = 1.0. Bin 3 weight: 0.5/1.0 = 0.5.
    expected = torch.tensor(
        [
            [1 - 0.25],
            [-0.25],
            [-0.25],
            [-0.25],
            [0.5 * (1 - 0.75)],
            [0.5 * (1 - 0.75)],
            [0.5 * (1 - 0.75)],
            [0.5 * -0.75],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_pool_window_evicts_old_batches():
    """The sliding window must keep only the last `pool_num` batches' p̂s."""
    pool_num = 3
    state = None
    for step in range(5):
        # Each batch has two in-(0,1) prompts so the pool grows by 2 per step.
        success_rates = torch.tensor([0.25, 0.5], dtype=torch.float32)
        state = _append_curverl_batch(
            state=state if state is not None else _copy_curverl_state(None),
            prompt_success_rates=success_rates,
            pool_num=pool_num,
        )
    # After 5 steps with pool_num=3 -> 3 batches retained, 6 values total.
    assert len(state["pool_batch_ids"]) == pool_num
    assert len(state["pool_values"]) == pool_num * 2
    assert state["next_pool_batch_id"] == 5


def test_zero_and_one_excluded_from_pool():
    """All-fail and all-success groups must be excluded from the sliding pool."""
    state = _append_curverl_batch(
        state=_copy_curverl_state(None),
        prompt_success_rates=torch.tensor([0.0, 0.5, 1.0, 0.75], dtype=torch.float32),
        pool_num=10,
    )
    assert sorted(state["pool_values"]) == [0.5, 0.75]


def test_lagged_reference_window():
    """At step t the reference distribution must be built from the *previous*
    batches' p̂ only — not include the current batch.

    Numeric witness: with 4 rollouts num_bins = 5, so p̂ = 0.5 lands in bin 2
    and p̂ = 0.25 lands in bin 1. After step 1 the pool holds {0.5} (bin 2).
    At step 2 with p̂ = 0.25 the lagged reference puts all mass in bin 2, so
    F_ref(bin 1) = 0 → weight ≈ 0 → step-2 advantages are ≈ 0. If the
    implementation accidentally included the current batch the reference
    would also populate bin 1 and step-2 advantages would be nonzero.
    """
    config = _config(pool_num=10)
    rewards, mask, idx = _make_batch([0.5], n_rollouts=4)
    _, _, state_after_step1, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards,
        token_level_scores=rewards,
        response_mask=mask,
        index=idx,
        config=config,
        curverl_state=None,
        return_stats=True,
    )
    assert state_after_step1["pool_values"] == [0.5]

    rewards2, mask2, idx2 = _make_batch([0.25], n_rollouts=4)
    advantages2, _, state_after_step2, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards2,
        token_level_scores=rewards2,
        response_mask=mask2,
        index=idx2,
        config=config,
        curverl_state=state_after_step1,
        return_stats=True,
    )
    # Lagged-window correctness witness: ~0 advantages at step 2.
    assert torch.allclose(advantages2, torch.zeros_like(advantages2), atol=1e-5)
    # The new batch is appended only *after* the histogram is built.
    assert sorted(state_after_step2["pool_values"]) == [0.25, 0.5]


def test_pool_num_zero_uses_only_current_batch():
    """`curverl_pool_num=0` must evict every batch after the update so the
    next step's reference is the current batch alone."""
    config = _config(pool_num=0)
    rewards, mask, idx = _make_batch([0.5], n_rollouts=4)
    _, _, state1, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards,
        token_level_scores=rewards,
        response_mask=mask,
        index=idx,
        config=config,
        curverl_state=None,
        return_stats=True,
    )
    assert state1["pool_values"] == []
    assert state1["pool_batch_ids"] == []

    # Step 2 with a different p̂ uses only the current batch as reference, so
    # the single bin gets weight 1 and the advantage is exactly (success - p̂).
    rewards2, mask2, idx2 = _make_batch([0.25], n_rollouts=4)
    advantages2, _, state2, _ = compute_curverl_outcome_advantage(
        token_level_rewards=rewards2,
        token_level_scores=rewards2,
        response_mask=mask2,
        index=idx2,
        config=config,
        curverl_state=state1,
        return_stats=True,
    )
    expected = torch.tensor(
        [[1 - 0.25], [-0.25], [-0.25], [-0.25]], dtype=torch.float32
    )
    assert torch.allclose(advantages2, expected, atol=1e-6)
    assert state2["pool_values"] == []


def test_histogram_mass_and_cdf_are_monotone():
    values = torch.tensor([0.1, 0.3, 0.3, 0.7], dtype=torch.float32)
    hist = _compute_curverl_histogram(
        reference_values=values, num_bins=5, epsilon=1e-12
    )
    assert torch.isclose(hist["mass"].sum(), torch.tensor(1.0), atol=1e-6)
    cdf = hist["cdf"]
    assert torch.all(cdf[1:] >= cdf[:-1] - 1e-6)
    assert torch.isclose(cdf[-1], torch.tensor(1.0), atol=1e-6)


def test_state_round_trip_keeps_window():
    """Copying the state must yield an independent equivalent dict."""
    state_a = _append_curverl_batch(
        state=_copy_curverl_state(None),
        prompt_success_rates=torch.tensor([0.25, 0.75], dtype=torch.float32),
        pool_num=10,
    )
    state_b = _copy_curverl_state(state_a)
    state_b["pool_values"].append(99.0)
    assert state_a["pool_values"] != state_b["pool_values"]
    assert state_a["pool_batch_ids"] == [0]
    assert state_b["pool_batch_ids"] == [0]
