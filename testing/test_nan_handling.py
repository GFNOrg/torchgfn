"""Tests for NaN/inf detection guards across the library.

These tests verify that the library catches numerical issues at the source
and raises informative errors rather than silently propagating NaN/inf.

All guards are behind `debug=True` to avoid torch.compile graph breaks.
"""

import pytest
import torch

from gfn.estimators import DiscretePolicyEstimator, LogitBasedEstimator
from gfn.gflownet import FMGFlowNet, TBGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP


def _make_hypergrid_tb(debug=True):
    """Creates a HyperGrid env + TBGFlowNet with debug=True for testing."""
    env = HyperGrid(ndim=2, height=4, debug=debug, validate_modes=False)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    pf_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    pb_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1)
    pf = DiscretePolicyEstimator(
        pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )
    return env, pf, pb


# ---------------------------------------------------------------------------
# 1. Non-finite log_rewards in get_scores()
# ---------------------------------------------------------------------------


def test_get_scores_raises_on_non_finite_log_rewards():
    """get_scores() raises ValueError when log_rewards contain -inf.

    This catches the common case where env.reward() returns zero for some
    states, producing log(0) = -inf, which would cause inf/NaN loss.
    """
    env, pf, pb = _make_hypergrid_tb()
    gflownet = TBGFlowNet(pf=pf, pb=pb, debug=True)
    sampler = Sampler(pf)

    torch.manual_seed(0)
    trajectories = sampler.sample_trajectories(env, n=5)

    # Manually inject -inf into log_rewards (simulating zero reward).
    trajectories._log_rewards = torch.full((trajectories.batch_size,), -float("inf"))

    with pytest.raises(ValueError, match="Non-finite log_rewards"):
        gflownet.get_scores(trajectories)


def test_get_scores_raises_on_nan_log_rewards():
    """get_scores() raises ValueError when log_rewards contain NaN.

    This catches the case where env.reward() returns negative values,
    producing log(negative) = NaN.
    """
    env, pf, pb = _make_hypergrid_tb()
    gflownet = TBGFlowNet(pf=pf, pb=pb, debug=True)
    sampler = Sampler(pf)

    torch.manual_seed(0)
    trajectories = sampler.sample_trajectories(env, n=5)

    # Manually inject NaN into log_rewards (simulating negative reward).
    trajectories._log_rewards = torch.full((trajectories.batch_size,), float("nan"))

    with pytest.raises(ValueError, match="Non-finite log_rewards"):
        gflownet.get_scores(trajectories)


def test_get_scores_ok_with_clipped_log_rewards():
    """log_reward_clip_min prevents -inf from reaching the non-finite check."""
    env, pf, pb = _make_hypergrid_tb()
    gflownet = TBGFlowNet(pf=pf, pb=pb, log_reward_clip_min=-100.0, debug=True)
    sampler = Sampler(pf)

    torch.manual_seed(0)
    trajectories = sampler.sample_trajectories(env, n=5)

    # Inject -inf — clipping should save us.
    trajectories._log_rewards = torch.full((trajectories.batch_size,), -float("inf"))

    # Should not raise — clipping converts -inf to -100.
    scores = gflownet.get_scores(trajectories)
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# 2. NaN module output in estimator
# ---------------------------------------------------------------------------


def test_nan_module_output_raises():
    """Estimator raises ValueError when neural network outputs NaN.

    This catches exploding gradients or numerical instability at the source,
    before NaN can propagate through log_softmax -> Categorical -> log_prob.
    """
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    states = env.reset(batch_shape=(3,))
    masks = states.forward_masks

    # Create logits with NaN (simulating exploding gradients).
    logits = torch.randn(3, env.n_actions)
    logits[1, 2] = float("nan")

    with pytest.raises(ValueError, match="Module output contains NaN"):
        LogitBasedEstimator._compute_logits_for_distribution(
            logits=logits,
            masks=masks,
            sf_index=env.n_actions - 1,
            sf_bias=0.0,
            temperature=1.0,
            epsilon=0.0,
            debug=True,
        )


def test_finite_module_output_ok():
    """Normal (finite) module outputs pass through without error."""
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    states = env.reset(batch_shape=(3,))
    masks = states.forward_masks

    logits = torch.randn(3, env.n_actions)

    # Should not raise.
    result = LogitBasedEstimator._compute_logits_for_distribution(
        logits=logits,
        masks=masks,
        sf_index=env.n_actions - 1,
        sf_bias=0.0,
        temperature=1.0,
        epsilon=0.0,
        debug=True,
    )
    assert torch.isfinite(result[masks]).all()


# ---------------------------------------------------------------------------
# 3. NaN scores in flow matching loss
# ---------------------------------------------------------------------------


def test_fm_loss_raises_on_nan_scores():
    """Flow matching loss raises ValueError when scores contain NaN.

    This occurs when both incoming and outgoing flows for a state are all
    -inf (no valid flow paths), producing logsumexp(-inf) - logsumexp(-inf)
    = -inf - (-inf) = NaN.
    """
    env = HyperGrid(ndim=2, height=4, debug=True, validate_modes=False)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module,
        n_actions=env.n_actions,
        preprocessor=preprocessor,
    )
    gflownet = FMGFlowNet(estimator, debug=True)

    torch.manual_seed(0)
    trajectories = gflownet.sample_trajectories(env, n=6)
    states_container = gflownet.to_training_samples(trajectories)
    states = states_container.intermediary_states

    if len(states) == 0:
        pytest.skip("No intermediary states sampled")

    # Sabotage the logF module to produce all -inf outputs ->
    # logsumexp(-inf) - logsumexp(-inf) = NaN.
    with torch.no_grad():
        for p in gflownet.logF.module.parameters():
            p.fill_(float("-inf"))

    with pytest.raises(ValueError, match="NaN in flow matching scores"):
        gflownet.flow_matching_loss(env, states)
