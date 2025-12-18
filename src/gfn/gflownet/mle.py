"""
MLE loss for diffusion GFlowNets (forward PF with optional PB).

Key equations (per time step, shapes in comments):
  - Backward bridge (s_t -> s_{t-dt}):
      mean_bb = s_t * (1 - dt / t)      # (B, s_dim)
      std_bb  = sigma * sqrt(dt*(t-dt)/t)  # (B, 1) broadcast
    With learned PB corrections:
      mean = mean_bb + mean_corr
      std  = sqrt(std_bb^2 + corr_std^2)
  - Forward PF log-prob for increment Δ = s_t - s_{t-dt}:
      If PF predicts log_std:
        σ = exp(log_std) * sqrt(dt) * sqrt(t_scale); optionally combine exploration
        log p = -0.5 * Σ_i [ ((Δ - dt μ)_i / σ_i)^2 + 2 log σ_i + log 2π ]
      Else (fixed variance):
        σ = sigma * sqrt(dt) * sqrt(t_scale); optionally combine exploration
        log p = -0.5 * Σ_i [ ((Δ - dt μ)_i / σ)^2 + log(2π σ^2) ]
  - Loss = -mean over batch of Σ_t log p_t

Tensor conventions:
  - terminal_states: (B, s_dim) or (B, s_dim + 1) with last dim an extra
    terminal indicator column; we drop the last dim if present.
  - Times: scalar dt = 1/num_steps; t_curr = 1 - i*dt; t_fwd = 1 - (i+1)*dt.

Usage (user owns optimizer/loop):
```python
gfn = MLEDiffusion(pf=pf, pb=None, num_steps=100, sigma=2.0, t_scale=1.0)
opt = torch.optim.Adam(gfn.parameters(), lr=1e-3)
for it in n_iterations:
    # Sample a batch of terminal states.
    batch = env.sample(batch_size)  # batch shape (B, s_dim)
    opt.zero_grad()
    # Calculate the MLE loss under the backward / forward diffusion process.
    loss = gfn.loss(batch, exploration_std=0.0)
    loss.backward()
    opt.step()
```
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

try:  # torch._dynamo may be absent or flagged private by linters
    from torch._dynamo import disable as dynamo_disable
except Exception:  # pragma: no cover

    def dynamo_disable(fn):  # type: ignore[return-type]
        return fn


from gfn.env import Env
from gfn.estimators import (
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
)
from gfn.gflownet.base import GFlowNet
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.modules import DiffusionFixedBackwardModule

# Relative tolerance for detecting initial/terminal states in diffusion trajectories.
# Must be synchronized with TERMINAL_TIME_EPS in gfn.gym.diffusion_sampling and
# _DIFFUSION_TERMINAL_TIME_EPS in gfn.estimators.
_DIFFUSION_TERMINAL_TIME_EPS = 1e-2


class MLEDiffusion(GFlowNet):
    """
    Maximum-likelihood diffusion GFlowNet (PF with optional PB).

    The caller owns the training loop; this class provides:
      - sampling via the forward PF (for API compatibility)
      - `.loss(env, terminal_states, ...)` computing the MLE objective
    """

    def __init__(
        self,
        pf: PinnedBrownianMotionForward,
        pb: Optional[PinnedBrownianMotionBackward] = None,
        *,
        num_steps: int,
        sigma: float,
        t_scale: float = 1.0,
        pb_scale_range: float = 0.1,
        learn_variance: bool = False,
        reduction: str = "mean",
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.pf = pf
        if pb is None:
            # Constant PB estimator (no learned parameters)
            pb = PinnedBrownianMotionBackward(
                s_dim=pf.s_dim,
                pb_module=DiffusionFixedBackwardModule(pf.s_dim),
                sigma=sigma,
                num_discretization_steps=num_steps,
                n_variance_outputs=0,
                pb_scale_range=pb_scale_range,
            ).to(next(pf.parameters()).device)
        self.pb = pb
        self.s_dim = pf.s_dim
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
        self.sigma = sigma
        self.t_scale = t_scale
        self.pb_scale_range = pb_scale_range
        self.learn_variance = learn_variance
        self.reduction = reduction
        self.debug = debug

        # Sampler for base-class API (sample_trajectories).
        self.sampler = Sampler(estimator=self.pf)

    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditions: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ):
        return self.sampler.sample_trajectories(
            env,
            n,
            conditions=conditions,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
            **policy_kwargs,
        )

    def to_training_samples(self, trajectories):
        return trajectories

    def loss(
        self,
        env: Env,
        terminal_states: Any,
        recalculate_all_logprobs: bool = True,
        *,
        exploration_std: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        """
        Compute the MLE objective given terminal states sampled from the target.

        Args:
            terminal_states: torch.Tensor or States; shape (B, s_dim) or (B, s_dim+1).
            exploration_std: extra state-space noise (combined in quadrature with PF std).
        Returns:
            Scalar loss (mean reduction).
        """
        del env  # unused
        del recalculate_all_logprobs  # unused
        device, dtype, s_curr = self._extract_samples(terminal_states)

        bsz, dim = s_curr.shape
        assert dim == self.s_dim, f"Expected s_dim={self.s_dim}, got {dim}"
        dt = self.dt

        # Tolerance for detecting initial state (t ≈ 0). Uses the module-level constant
        # which must stay synchronized with TERMINAL_TIME_EPS in diffusion_sampling.py
        # and _DIFFUSION_TERMINAL_TIME_EPS in estimators.py.
        eps_s0 = dt * _DIFFUSION_TERMINAL_TIME_EPS

        sqrt_dt_t_scale = math.sqrt(dt * self.t_scale)
        base_std_fixed = self.sigma * sqrt_dt_t_scale
        log_2pi = math.log(2 * math.pi)

        logpf_sum = torch.zeros(bsz, device=device, dtype=dtype)
        exploration_std_t = torch.as_tensor(
            exploration_std, device=device, dtype=dtype
        ).clamp(min=0.0)
        exploration_var = exploration_std_t**2

        # Precompute time grids to avoid per-step allocations.
        all_t_fwd = torch.linspace(
            1.0 - dt, 0.0, self.num_steps, device=device, dtype=dtype
        )
        all_t_curr = torch.linspace(1.0, dt, self.num_steps, device=device, dtype=dtype)

        for i in range(self.num_steps):
            # Times: forward transition index t_fwd corresponds to s_prev -> s_curr.
            t_fwd = all_t_fwd[i].expand(bsz, 1)
            t_curr = all_t_curr[i].expand(bsz, 1)

            # Backward sampler: Brownian bridge base + optional PB corrections.
            pb_inp = torch.cat([s_curr, t_curr], dim=1)
            pb_out = self.pb.module(pb_inp)

            # Base Brownian bridge mean/std toward 0 at t=0.
            is_s0 = (t_curr - dt) < eps_s0
            not_s0 = (~is_s0).float()

            base_mean = s_curr * (1.0 - dt / t_curr) * not_s0
            base_std = self.sigma * (dt * (t_curr - dt) / t_curr).sqrt() * not_s0

            # Learned corrections (PB): mean_corr, optional log-std corr.
            mean_corr = pb_out[..., :dim] * self.pb.pb_scale_range
            if self.pb.n_variance_outputs > 0:
                log_std_corr = pb_out[..., [-1]] * self.pb.pb_scale_range
                corr_std = torch.exp(log_std_corr)
            else:
                corr_std = torch.zeros_like(base_std)

            bwd_std = (base_std**2 + corr_std**2).sqrt()
            noise = torch.randn_like(s_curr, device=device, dtype=dtype)
            s_prev = base_mean + mean_corr + bwd_std * noise

            # Forward log-prob under PF for the observed increment (s_prev -> s_curr).
            model_inp = torch.cat([s_prev, t_fwd], dim=1)
            module_out = self.pf.module(model_inp)
            increment = s_curr - s_prev

            # Case where module outputs learned variance.
            if self.pf.n_variance_outputs > 0:
                drift = module_out[..., :dim]
                log_std = module_out[..., [-1]]
                std = torch.exp(log_std) * sqrt_dt_t_scale
                std = torch.sqrt(std**2 + exploration_var)
                diff = increment - dt * drift
                logpf_step = -0.5 * ((diff / std) ** 2 + 2 * std.log() + log_2pi).sum(
                    dim=1
                )
            # Fixed variance case.
            else:
                drift = module_out
                std = torch.sqrt(base_std_fixed**2 + exploration_var)
                diff = increment - dt * drift
                logpf_step = -0.5 * ((diff / std) ** 2).sum(dim=1) - 0.5 * dim * (
                    log_2pi + 2 * torch.log(std)
                )

            logpf_sum += logpf_step
            s_curr = s_prev

        if self.debug and torch.isnan(logpf_sum).any():
            raise ValueError("NaNs in logpf_sum during MLE loss.")

        # TODO: Use included loss reduction helpers.
        loss = -(logpf_sum.mean() if self.reduction == "mean" else logpf_sum.sum())
        if self.debug:
            self._assert_no_nan(logpf_sum)
        return loss

    @dynamo_disable
    def _assert_no_nan(self, logpf_sum: torch.Tensor) -> None:
        if torch.isnan(logpf_sum).any():
            raise ValueError("NaNs in logpf_sum during MLE loss.")

    @dynamo_disable
    def _extract_samples(
        self, terminal_states: Any
    ) -> tuple[torch.device, torch.dtype, torch.Tensor]:
        """
        Normalize input to a (B, s_dim) tensor.
        Accepts torch.Tensor or States; drops a final column if size matches s_dim+1.
        """
        if isinstance(terminal_states, States):
            tensor = terminal_states.tensor
        elif torch.is_tensor(terminal_states):
            tensor = terminal_states
        else:
            raise TypeError(f"Unsupported terminal_states type: {type(terminal_states)}")

        if tensor.shape[-1] == self.s_dim + 1:
            tensor = tensor[..., :-1]
        device = tensor.device
        dtype = tensor.dtype
        return device, dtype, tensor
