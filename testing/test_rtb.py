from typing import cast

import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import (
    RelativeLogPartitionVarianceGFlowNet,
    RelativeTrajectoryBalanceGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP


def _make_hypergrid_estimators():
    """Build simple forward policies for HyperGrid prior/posterior."""
    env = HyperGrid(ndim=2, height=4)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)

    pf_module_post = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)
    pf_module_prior = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)

    pf_post = DiscretePolicyEstimator(
        module=pf_module_post,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pf_prior = DiscretePolicyEstimator(
        module=pf_module_prior,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    return env, pf_post, pf_prior


def test_rtb_loss_backward_and_grads():
    torch.manual_seed(0)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=1.0,
    )
    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=8, save_logprobs=True, save_estimator_outputs=False
    )

    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)

    loss.backward()

    # Posterior parameters and logZ should receive gradients.
    assert any(p.grad is not None for p in pf_post.parameters())
    assert any(p.grad is not None for p in gfn.logz_parameters())

    # Prior parameters are not part of the RTB graph and should have no grads.
    assert all(p.grad is None for p in pf_prior.parameters())


def test_rtb_loss_forward_only_path():
    """Ensure RTB loss works with recalculate_all_logprobs=False."""
    torch.manual_seed(1)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=0.5,
    )
    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=4, save_logprobs=True, save_estimator_outputs=False
    )

    # Use cached log_probs; should not rely on any backward policy.
    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# RelativeLogPartitionVarianceGFlowNet tests
# ---------------------------------------------------------------------------


def test_rel_lpv_loss_backward_and_grads():
    """RelLPV produces finite loss with correct gradient flow."""
    torch.manual_seed(0)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeLogPartitionVarianceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        beta=1.0,
    )
    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=8, save_logprobs=True, save_estimator_outputs=False
    )

    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)

    loss.backward()

    # Posterior parameters should receive gradients.
    assert any(p.grad is not None for p in pf_post.parameters())

    # Prior parameters should NOT receive gradients.
    assert all(p.grad is None for p in pf_prior.parameters())


def test_rel_lpv_no_logz_parameters():
    """RelLPV should have no logZ-related parameters."""
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeLogPartitionVarianceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
    )

    logz_params = [k for k, _ in gfn.named_parameters() if "logZ" in k or "logz" in k]
    assert len(logz_params) == 0


def test_rel_lpv_matches_rtb_at_optimal_logz():
    """RelLPV loss should equal RTB loss evaluated at the batch-optimal logZ."""
    torch.manual_seed(42)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    beta = 1.5
    gfn_lpv = RelativeLogPartitionVarianceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        beta=beta,
    )
    gfn_rtb = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        beta=beta,
        init_logZ=0.0,
    )

    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=16, save_logprobs=True, save_estimator_outputs=False
    )

    # Compute RTB scores manually at the batch-optimal logZ.
    from gfn.utils.prob_calculations import get_trajectory_pfs

    with torch.no_grad():
        log_pf_post = gfn_rtb.trajectory_log_probs_forward(
            trajectories, recalculate_all_logprobs=True
        ).sum(dim=0)
        log_pf_prior = get_trajectory_pfs(
            pf_prior, trajectories, fill_value=0.0, recalculate_all_logprobs=True
        ).sum(dim=0)
        log_rewards = cast(torch.Tensor, trajectories.log_rewards)

        scores = log_pf_post - log_pf_prior - beta * log_rewards
        optimal_logZ = -scores.mean()
        rtb_at_optimal = (0.5 * (scores + optimal_logZ).pow(2)).mean()

        lpv_loss = gfn_lpv.loss(env, trajectories, recalculate_all_logprobs=True)

    assert torch.allclose(rtb_at_optimal, lpv_loss, atol=1e-5)


def test_rel_lpv_prior_not_in_parameters():
    """Prior should not leak into the module's parameters."""
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeLogPartitionVarianceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
    )

    all_names = [k for k, _ in gfn.named_parameters()]
    assert not any("prior" in k for k in all_names)
    assert gfn.prior_pf is pf_prior
