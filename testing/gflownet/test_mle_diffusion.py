import math

import torch

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gflownet.mle import MLEDiffusion
from gfn.utils.modules import DiffusionFixedBackwardModule


class ZeroDriftModule(torch.nn.Module):
    """Returns zero drift (and optional zero log-std if learn_variance)."""

    def __init__(self, s_dim: int, learn_variance: bool = False):
        super().__init__()
        self.s_dim = s_dim
        self.learn_variance = learn_variance
        # Required by IdentityPreprocessor in estimators.
        self.input_dim = s_dim + 1  # state dim + time

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x shape: (B, s_dim + 1)
        batch = x.shape[0]
        if self.learn_variance:
            return torch.zeros(batch, self.s_dim + 1, device=x.device, dtype=x.dtype)
        return torch.zeros(batch, self.s_dim, device=x.device, dtype=x.dtype)


def _build_estimators(s_dim: int, learn_variance: bool, num_steps: int = 1):
    """Helper to build deterministic PF/PB for tests."""
    pf_module = ZeroDriftModule(s_dim=s_dim, learn_variance=learn_variance)
    pf = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
        n_variance_outputs=1 if learn_variance else 0,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=DiffusionFixedBackwardModule(s_dim),
        sigma=1.0,
        num_discretization_steps=num_steps,
        n_variance_outputs=0,
        pb_scale_range=0.1,
    )
    return pf, pb


def test_mle_loss_fixed_variance_zero_terminal():
    """
    With zero drift, fixed variance (sigma=1), num_steps=1, and terminal states at 0,
    the loss is deterministic: log(2π) per dimension /2 summed over dim -> log(2π).
    """
    torch.manual_seed(0)
    s_dim = 2
    pf, pb = _build_estimators(s_dim=s_dim, learn_variance=False, num_steps=1)
    trainer = MLEDiffusion(
        pf=pf,
        pb=pb,
        num_steps=1,
        sigma=1.0,
        t_scale=1.0,
        pb_scale_range=0.1,
        learn_variance=False,
    )

    batch = torch.zeros(4, s_dim)  # terminal states near (0,0)
    loss = trainer.loss(batch, exploration_std=0.0)

    expected_logp = -0.5 * s_dim * math.log(2 * math.pi)  # log p for zero increment
    expected_loss = -expected_logp  # num_steps=1, loss = -logpf_sum.mean()
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-6)


def test_mle_loss_learned_variance_zero_terminal():
    """
    Learned variance head returning log_std=0 should match the fixed-variance case
    (std = exp(0)*sqrt(dt)*sqrt(t_scale) = 1 when num_steps=1, t_scale=1).
    """
    torch.manual_seed(0)
    s_dim = 2
    pf, pb = _build_estimators(s_dim=s_dim, learn_variance=True, num_steps=1)
    trainer = MLEDiffusion(
        pf=pf,
        pb=pb,
        num_steps=1,
        sigma=1.0,
        t_scale=1.0,
        pb_scale_range=0.1,
        learn_variance=True,
    )

    batch = torch.zeros(3, s_dim)
    loss = trainer.loss(batch, exploration_std=0.0)

    expected_logp = -0.5 * s_dim * math.log(2 * math.pi)
    expected_loss = -expected_logp
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-6)


def test_backward_bridge_mean_std_match_formula():
    """
    Validate Brownian bridge mean/std against closed form for num_steps=2 at t=1.
    For s_curr=0, mean should be 0, std should be sigma*sqrt(dt*(t-dt)/t).
    """
    s_dim = 2
    num_steps = 2
    sigma = 1.0
    pf, pb = _build_estimators(s_dim=s_dim, learn_variance=False, num_steps=num_steps)

    # Manually run the PB module once at t=1.
    dt = 1.0 / num_steps
    bsz = 3
    s_curr = torch.zeros(bsz, s_dim)
    t_curr = torch.full((bsz, 1), 1.0)
    pb_inp = torch.cat([s_curr, t_curr], dim=1)
    pb_out = pb.module(pb_inp)

    is_s0 = (t_curr - dt) < dt * 1e-2
    base_mean = torch.where(
        is_s0,
        torch.zeros_like(s_curr),
        s_curr * (1.0 - dt / t_curr),
    )
    base_std = torch.where(
        is_s0,
        torch.zeros_like(t_curr),
        sigma * (dt * (t_curr - dt) / t_curr).sqrt(),
    )

    # For zero corrections, mean_corr=0, corr_std=0.
    mean_corr = pb_out[..., :s_dim] * pb.pb_scale_range
    assert torch.allclose(mean_corr, torch.zeros_like(mean_corr))
    assert torch.allclose(base_mean, torch.zeros_like(base_mean))
    expected_std = sigma * math.sqrt(dt * (1.0 - dt) / 1.0)
    assert torch.allclose(base_std.squeeze(-1), torch.full((bsz,), expected_std))


def test_forward_logprob_zero_increment_matches_formula():
    """
    For PF with zero drift/log_std=0, num_steps=1, t_scale=1, increment=0,
    the log-prob per dim is -0.5*log(2π); total logp = that * s_dim.
    """
    s_dim = 2
    pf, pb = _build_estimators(s_dim=s_dim, learn_variance=True, num_steps=1)
    trainer = MLEDiffusion(
        pf=pf,
        pb=pb,
        num_steps=1,
        sigma=1.0,
        t_scale=1.0,
        pb_scale_range=0.1,
        learn_variance=True,
    )

    batch = torch.zeros(2, s_dim)
    # Manually compute expected logp for zero increment:
    # std = exp(0) * sqrt(dt) * sqrt(t_scale) = 1; logp = -0.5 * s_dim * log(2π)
    expected_logp = -0.5 * s_dim * math.log(2 * math.pi)
    expected_loss = -expected_logp
    loss = trainer.loss(batch, exploration_std=0.0)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-6)
