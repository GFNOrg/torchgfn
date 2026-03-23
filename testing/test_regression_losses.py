"""Tests for pluggable regression losses (gfn.gflownet.losses).

Covers:
  - Unit properties of each loss function
  - Numerical stability at extreme residuals
  - Backward compatibility (default loss_fn reproduces old behavior)
  - Integration with TB, RTB, and DB using alternative losses
"""

import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import (
    DBGFlowNet,
    RelativeTrajectoryBalanceGFlowNet,
    TBGFlowNet,
)
from gfn.gflownet.losses import LinexLoss, ShiftedCoshLoss, SquaredLoss
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_LOSSES = [SquaredLoss(), ShiftedCoshLoss(), LinexLoss(1.0), LinexLoss(0.5)]


def _residuals():
    """A range of test residuals including 0, positive, negative, and large values."""
    return torch.tensor([-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0])


# ---------------------------------------------------------------------------
# Unit tests: properties of each loss
# ---------------------------------------------------------------------------


def test_all_losses_zero_at_zero():
    """Every loss must return 0 when the residual is 0."""
    zero = torch.tensor(0.0)
    for loss_fn in ALL_LOSSES:
        result = loss_fn(zero)
        assert torch.allclose(
            result, torch.tensor(0.0), atol=1e-7
        ), f"{loss_fn} at 0 gave {result.item()}"


def test_all_losses_nonnegative():
    """Every loss must be non-negative for all residuals."""
    r = _residuals()
    for loss_fn in ALL_LOSSES:
        result = loss_fn(r)
        assert (result >= -1e-7).all(), f"{loss_fn} gave negative: {result}"


def test_all_losses_gradient_zero_at_zero():
    """The gradient of every loss at t=0 should be 0 (minimum)."""
    for loss_fn in ALL_LOSSES:
        r = torch.tensor(0.0, requires_grad=True)
        g = loss_fn(r)
        g.backward()
        assert r.grad is not None
        assert torch.allclose(
            r.grad, torch.tensor(0.0), atol=1e-6
        ), f"{loss_fn} gradient at 0: {r.grad.item()}"


def test_squared_loss_matches_t_squared():
    """SquaredLoss must exactly equal t^2."""
    r = _residuals()
    assert torch.allclose(SquaredLoss()(r), r.pow(2))


def test_shifted_cosh_symmetric():
    """ShiftedCosh must satisfy g(t) == g(-t)."""
    r = _residuals()
    loss = ShiftedCoshLoss()
    assert torch.allclose(loss(r), loss(-r), atol=1e-6)


def test_linex_asymmetric():
    """LinexLoss must be asymmetric: g(t) != g(-t) for t != 0."""
    r = torch.tensor([1.0, 2.0])
    loss = LinexLoss(1.0)
    assert not torch.allclose(loss(r), loss(-r))


def test_linex_alpha_validation():
    """LinexLoss(alpha=0) should raise ValueError."""
    try:
        LinexLoss(alpha=0.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_numerical_stability_large_residuals():
    """No NaN or Inf for residuals up to |t| = 100."""
    r = torch.linspace(-100, 100, 201)
    for loss_fn in ALL_LOSSES:
        result = loss_fn(r)
        assert torch.isfinite(result).all(), f"{loss_fn} not finite at extreme values"


# ---------------------------------------------------------------------------
# Backward compatibility: default loss_fn == old behavior
# ---------------------------------------------------------------------------


def _make_tb_setup():
    """Build a TBGFlowNet with HyperGrid and sample trajectories."""
    torch.manual_seed(0)
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    preproc = KHotPreprocessor(env.height, env.ndim)
    pf_module = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)
    pb_module = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions - 1)
    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=True,
    )
    return env, pf, pb


def test_tb_default_loss_fn_backward_compat():
    """TBGFlowNet with default loss_fn must match old t^2 behavior."""
    env, pf, pb = _make_tb_setup()
    gfn = TBGFlowNet(pf=pf, pb=pb)
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    with torch.no_grad():
        loss = gfn.loss(env, trajs, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)

    # Verify it's actually using SquaredLoss
    assert isinstance(gfn.loss_fn, SquaredLoss)


# ---------------------------------------------------------------------------
# Integration: alternative losses produce finite loss + gradients
# ---------------------------------------------------------------------------


def test_tb_with_shifted_cosh():
    """TBGFlowNet with ShiftedCoshLoss produces finite loss and gradients."""
    env, pf, pb = _make_tb_setup()
    gfn = TBGFlowNet(pf=pf, pb=pb, loss_fn=ShiftedCoshLoss())
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    loss = gfn.loss(env, trajs, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in pf.parameters())


def test_tb_with_linex():
    """TBGFlowNet with LinexLoss(1) produces finite loss and gradients."""
    env, pf, pb = _make_tb_setup()
    gfn = TBGFlowNet(pf=pf, pb=pb, loss_fn=LinexLoss(1.0))
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    loss = gfn.loss(env, trajs, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in pf.parameters())


def test_rtb_with_shifted_cosh():
    """RTB with ShiftedCoshLoss produces finite loss and gradients."""
    torch.manual_seed(0)
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    preproc = KHotPreprocessor(env.height, env.ndim)
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

    gfn = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        loss_fn=ShiftedCoshLoss(),
    )
    sampler = Sampler(estimator=pf_post)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)

    loss = gfn.loss(env, trajs, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in pf_post.parameters())


def test_db_with_shifted_cosh():
    """DBGFlowNet with ShiftedCoshLoss produces finite loss and gradients."""
    from gfn.estimators import ScalarEstimator

    torch.manual_seed(0)
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    preproc = KHotPreprocessor(env.height, env.ndim)
    pf = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pb = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions - 1),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=True,
    )
    logF = ScalarEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=1),
        preprocessor=preproc,
    )

    gfn = DBGFlowNet(pf=pf, pb=pb, logF=logF, loss_fn=ShiftedCoshLoss())
    sampler = Sampler(estimator=pf)
    trajs = sampler.sample_trajectories(env, n=8, save_logprobs=True)
    transitions = gfn.to_training_samples(trajs)

    loss = gfn.loss(env, transitions, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in pf.parameters())
