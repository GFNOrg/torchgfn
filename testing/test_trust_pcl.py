"""Tests for the Trust-PCL equivalence (TrustPCLGFlowNet + utilities).

Verifies:
  - RL-native constructor maps to correct RTB parameters
  - alpha and v_soft_s0 properties match the equivalence
  - Trust-PCL loss == alpha^2 * RTB loss (numerically)
  - Gradient flow is identical to RTB
  - Parameter conversion roundtrips
"""

from typing import cast

import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet, TrustPCLGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP
from gfn.utils.trust_pcl import rtb_to_trust_pcl_params, trust_pcl_to_rtb_params


def _make_hypergrid_estimators():
    """Build simple forward policies for HyperGrid prior/posterior."""
    env = HyperGrid(ndim=2, height=4)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    pf_post = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pf_prior = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    for p in pf_prior.parameters():
        p.requires_grad_(False)
    return env, pf_post, pf_prior


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_alpha_equals_inverse_beta():
    """alpha property must equal 1/beta."""
    env, pf, prior = _make_hypergrid_estimators()
    for alpha_val in [0.5, 1.0, 2.0]:
        gfn = TrustPCLGFlowNet(policy=pf, reference_policy=prior, alpha=alpha_val)
        assert torch.allclose(
            torch.tensor(alpha_val),
            cast(torch.Tensor, gfn.alpha),
            atol=1e-6,
        )
        assert torch.allclose(
            torch.tensor(1.0 / alpha_val),
            cast(torch.Tensor, gfn.beta),
            atol=1e-6,
        )


def test_v_soft_s0_equals_alpha_times_logz():
    """v_soft_s0 must equal alpha * logZ."""
    env, pf, prior = _make_hypergrid_estimators()
    gfn = TrustPCLGFlowNet(
        policy=pf,
        reference_policy=prior,
        alpha=2.0,
        init_v_soft_s0=4.0,
    )
    # init_v_soft_s0 = 4.0, alpha = 2.0 → logZ = 4.0 * (1/2.0) = 2.0
    logZ_val = cast(torch.Tensor, gfn.logZ).item()
    assert abs(logZ_val - 2.0) < 1e-6

    # v_soft_s0 = alpha * logZ = 2.0 * 2.0 = 4.0
    assert torch.allclose(gfn.v_soft_s0, torch.tensor(4.0), atol=1e-6)


def test_rl_constructor_maps_to_rtb():
    """RL-style constructor args must map correctly to RTB internals."""
    env, pf, prior = _make_hypergrid_estimators()
    gfn = TrustPCLGFlowNet(
        policy=pf,
        reference_policy=prior,
        alpha=0.5,
        init_v_soft_s0=1.0,
    )
    # alpha=0.5 → beta=2.0
    assert torch.allclose(cast(torch.Tensor, gfn.beta), torch.tensor(2.0), atol=1e-6)
    # init_v_soft_s0=1.0, alpha=0.5 → logZ = 1.0 * 2.0 = 2.0
    assert abs(cast(torch.Tensor, gfn.logZ).item() - 2.0) < 1e-6
    # pf and prior_pf should be the same objects
    assert gfn.pf is pf
    assert gfn.prior_pf is prior


# ---------------------------------------------------------------------------
# Loss equivalence
# ---------------------------------------------------------------------------


def test_trust_pcl_loss_equals_alpha_sq_rtb():
    """Trust-PCL loss must equal alpha^2 * RTB loss."""
    torch.manual_seed(42)
    env, pf, prior = _make_hypergrid_estimators()

    alpha = 2.0
    beta = 1.0 / alpha

    gfn_rtb = RelativeTrajectoryBalanceGFlowNet(
        pf=pf, prior_pf=prior, beta=beta, init_logZ=0.5
    )
    gfn_tpcl = TrustPCLGFlowNet(
        policy=pf, reference_policy=prior, alpha=alpha, init_v_soft_s0=alpha * 0.5
    )

    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=16, save_logprobs=True)

    with torch.no_grad():
        loss_rtb = gfn_rtb.loss(env, trajs, recalculate_all_logprobs=True)
        loss_tpcl = gfn_tpcl.loss(env, trajs, recalculate_all_logprobs=True)

    expected = alpha**2 * loss_rtb
    # atol=1e-5 accommodates float32 accumulation error (~1e-7 per op)
    # across the ~100 multiply-adds in the loss computation.
    assert torch.allclose(
        expected, loss_tpcl, atol=1e-5
    ), f"Expected {expected.item()}, got {loss_tpcl.item()}"


def test_gradients_flow_same_as_rtb():
    """Trust-PCL should produce gradients for the same parameters as RTB."""
    torch.manual_seed(0)
    env, pf, prior = _make_hypergrid_estimators()

    gfn = TrustPCLGFlowNet(policy=pf, reference_policy=prior, alpha=1.0)
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    loss = gfn.loss(env, trajs, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)
    loss.backward()

    # Policy (posterior) should receive gradients.
    assert any(p.grad is not None for p in pf.parameters())
    # logZ should receive gradients.
    assert any(p.grad is not None for p in gfn.logz_parameters())
    # Reference (prior) should NOT receive gradients.
    assert all(p.grad is None for p in prior.parameters())


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_param_roundtrip_float():
    """RTB → Trust-PCL → RTB conversion must be the identity."""
    logZ, beta = 2.0, 0.5
    tpcl = rtb_to_trust_pcl_params(logZ, beta)
    rtb = trust_pcl_to_rtb_params(tpcl["alpha"], tpcl["v_soft_s0"])
    assert abs(rtb["logZ"] - logZ) < 1e-10
    assert abs(rtb["beta"] - beta) < 1e-10


def test_param_roundtrip_tensor():
    """Roundtrip with tensors."""
    logZ = torch.tensor(3.0)
    beta = torch.tensor(0.25)
    tpcl = rtb_to_trust_pcl_params(logZ, beta)
    rtb = trust_pcl_to_rtb_params(tpcl["alpha"], tpcl["v_soft_s0"])
    assert torch.allclose(torch.tensor(rtb["logZ"]), logZ, atol=1e-6)
    assert torch.allclose(torch.tensor(rtb["beta"]), beta, atol=1e-6)


def test_rtb_to_trust_pcl_values():
    """Spot-check known conversion values."""
    result = rtb_to_trust_pcl_params(logZ=2.0, beta=0.5)
    assert result["alpha"] == 2.0
    assert result["v_soft_s0"] == 4.0


def test_get_scores_public_api():
    """RelativeTBBase.get_scores() should return the same as _compute_rtb_scores."""
    torch.manual_seed(0)
    env, pf, prior = _make_hypergrid_estimators()
    gfn = TrustPCLGFlowNet(policy=pf, reference_policy=prior, alpha=1.0)
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    with torch.no_grad():
        scores_public = gfn.get_scores(trajs, recalculate_all_logprobs=True, env=env)
        scores_private = gfn._compute_rtb_scores(
            env, trajs, recalculate_all_logprobs=True
        )

    assert torch.allclose(scores_public, scores_private)
