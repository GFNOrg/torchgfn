r"""Soft (entropy-regularized) RL quantities for GFlowNet policy-gradient training.

GFlowNet training with a fixed backward policy is equivalent to entropy-regularized
RL at :math:`\alpha = \gamma = 1` (Tiapkin et al., 2024). On the same DAG, define an
episodic MDP whose action at :math:`s_t` is the choice of a child :math:`s_{t+1}`, and
give it the per-step reward

.. math::

    r(s_t, s_{t+1}) = \begin{cases}
        \log P_B(s_t \mid s_{t+1}), & s_t \notin \mathcal{X} \cup \{s_f\}, \\
        \log R(s_t),                & s_t \in \mathcal{X}, \\
        0,                          & s_t = s_f.
    \end{cases}

Then the soft-optimal policy induces the target distribution :math:`R(x) / Z` over
terminating states, and the soft value function coincides with the GFlowNet log-flow:
:math:`V^{\pi^\star}_{\alpha=1}(s) = \log \mathcal{F}(s)`, in particular
:math:`V^{\pi^\star}_{\alpha=1}(s_0) = \log Z`.

This module holds the stateless tensor math built on top of that reward: one-step soft
returns, reward-to-go, soft state values, and Generalized Advantage Estimation. It is
used by :mod:`gfn.gflownet.policy_gradient`, and is deliberately free of module state so
each piece can be tested against its defining equation.

References:
    Zykova-Myzina et al. "Proximal Policy Optimization for Amortized Discrete
    Sampling" (2026, arXiv:2606.15793) — Eqs. 7, 9, 10, 12, 13.

    Tiapkin et al. "Generative Flow Networks as Entropy-Regularized RL"
    (AISTATS 2024, arXiv:2310.12934).

    Schulman et al. "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation" (ICLR 2016, arXiv:1506.02438).

Tensor conventions (matching :class:`~gfn.containers.Trajectories`):
    - ``states`` has batch shape ``(T + 1, N)``, ``actions`` has ``(T, N)``.
    - Per-step tensors (``log_pf``, ``log_pb``, rewards, advantages) are ``(T, N)``.
    - Padding is self-neutralizing: on dummy steps ``log_pf = log_pb = 0``,
      ``is_exit = False`` and the state value is pinned to ``0``, so the one-step soft
      return and the TD residual are both exactly ``0`` there.
"""

from __future__ import annotations

import math

import torch

from gfn.containers import Trajectories
from gfn.estimators import ConditionalScalarEstimator, ScalarEstimator
from gfn.utils.handlers import call_estimator_with_conditions


def soft_step_rewards(
    trajectories: Trajectories,
    log_pb: torch.Tensor,
    log_rewards: torch.Tensor | None = None,
    log_reward_clip_min: float = -float("inf"),
) -> torch.Tensor:
    r"""Builds the soft-MDP per-step reward $r(s_t, s_{t+1})$ (Eq. 7).

    Non-exit transitions are rewarded with $\log P_B(s_t \mid s_{t+1})$; the single exit
    transition $x \to s_f$ is rewarded with $\log R(x)$. This exploits the fact that
    :func:`~gfn.utils.prob_calculations.get_trajectory_pbs` already returns exactly
    $\log P_B(s_t \mid s_{t+1})$ at index ``t``, and zero on exit and dummy steps.

    Args:
        trajectories: The batch of forward trajectories.
        log_pb: Tensor of shape (T, N) of backward logprobs, as returned by
            ``get_trajectory_pbs``.
        log_rewards: Optional log rewards of shape (N,). When None, uses
            ``trajectories.log_rewards``.
        log_reward_clip_min: If finite, clamps the log rewards from below.

    Returns:
        A tensor of shape (T, N) containing the per-step rewards.

    Raises:
        ValueError: If neither the trajectories nor the caller supply log rewards.
    """
    if log_rewards is None:
        log_rewards = trajectories.log_rewards
    if log_rewards is None:
        raise ValueError(
            "Trajectories have no log_rewards; pass `log_rewards` explicitly."
        )
    if math.isfinite(log_reward_clip_min):
        log_rewards = log_rewards.clamp_min(log_reward_clip_min)

    is_exit = trajectories.actions.is_exit  # (T, N)
    return torch.where(is_exit, log_rewards.unsqueeze(0).expand_as(log_pb), log_pb)


def soft_step_returns(log_pf: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    r"""Computes the one-step soft return $g_t = r_t - \log \pi_\theta(s_{t+1} \mid s_t)$.

    This is the entropy-regularized reward: the $-\log \pi$ term is the sample-path form
    of the policy entropy at $\alpha = 1$ (Eq. 9).

    Args:
        log_pf: Tensor of shape (T, N) of forward logprobs.
        rewards: Tensor of shape (T, N) from :func:`soft_step_rewards`.

    Returns:
        A tensor of shape (T, N) containing the one-step soft returns.
    """
    return rewards - log_pf


def reward_to_go(soft_returns: torch.Tensor) -> torch.Tensor:
    r"""Computes the soft reward-to-go $\hat{R}_t = \sum_{k \geq t} g_k$ (Eq. 10).

    Args:
        soft_returns: Tensor of shape (T, N) from :func:`soft_step_returns`.

    Returns:
        A tensor of shape (T, N) containing the reward-to-go at each step. Padded steps
        contribute zero, so their reward-to-go is zero as well.
    """
    return torch.flip(torch.cumsum(torch.flip(soft_returns, [0]), dim=0), [0])


def soft_state_values(
    logV: ScalarEstimator | ConditionalScalarEstimator,
    trajectories: Trajectories,
    debug: bool = False,
) -> torch.Tensor:
    r"""Evaluates the soft value function $\tilde{V}_\varphi(s_t)$ along trajectories.

    The absorbing sink state $s_f$ is pinned to $\tilde{V}_\varphi(s_f) := 0$, which is
    what makes the varying-horizon episodic return finite at $\gamma = 1$. Terminating
    states $x \in \mathcal{X}$ are *not* pinned: they receive a real value estimate,
    which at the soft optimum equals $\log R(x)$.

    Args:
        logV: A ScalarEstimator (or ConditionalScalarEstimator) for the soft value.
        trajectories: The batch of forward trajectories.
        debug: If True, keeps shape assertions active.

    Returns:
        A tensor of shape (T + 1, N) containing the soft state values, with zeros at
        sink states.
    """
    states = trajectories.states
    values = torch.zeros(
        states.batch_shape,
        dtype=torch.get_default_dtype(),
        device=states.device,
    )
    mask = ~states.is_sink_state
    valid_states = states[mask]

    if len(valid_states) == 0:
        return values

    conditions = states.conditions
    if conditions is not None and debug:
        assert conditions.shape[:2] == states.batch_shape
    out = call_estimator_with_conditions(
        logV, "logV", valid_states, None if conditions is None else conditions[mask]
    )

    values[mask] = out.squeeze(-1).to(values.dtype)
    return values


def soft_td_residuals(
    soft_returns: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    r"""Computes the soft TD residual $\delta_t = g_t + V(s_{t+1}) - V(s_t)$ (Eq. 12).

    Args:
        soft_returns: Tensor of shape (T, N) from :func:`soft_step_returns`.
        values: Tensor of shape (T + 1, N) from :func:`soft_state_values`.

    Returns:
        A tensor of shape (T, N) containing the soft TD residuals.
    """
    return soft_returns + values[1:] - values[:-1]


def gae_advantages(
    soft_returns: torch.Tensor,
    values: torch.Tensor,
    lamda: float,
) -> torch.Tensor:
    r"""Computes GAE soft advantages $\hat{A}_t = \sum_k \lambda^k \delta_{t+k}$ (Eq. 13).

    Evaluated with the standard reverse recursion
    $\hat{A}_t = \delta_t + \lambda \hat{A}_{t+1}$. At $\lambda = 1$ this recovers
    reward-to-go minus the baseline (unbiased, full Monte-Carlo variance); at
    $\lambda = 0$ it collapses to the one-step residual $\delta_t$ (low variance, biased
    whenever $V_\varphi \neq \tilde{V}^{\pi_\theta}$).

    Args:
        soft_returns: Tensor of shape (T, N) from :func:`soft_step_returns`.
        values: Tensor of shape (T + 1, N) from :func:`soft_state_values`.
        lamda: The GAE bias-variance trade-off parameter, in [0, 1].

    Returns:
        A tensor of shape (T, N) containing the GAE advantages.

    Raises:
        ValueError: If ``lamda`` is outside [0, 1].
    """
    if not 0.0 <= lamda <= 1.0:
        raise ValueError(f"GAE lamda must be in [0, 1], got {lamda}")

    deltas = soft_td_residuals(soft_returns, values)
    advantages = torch.empty_like(deltas)
    # Shaped from `deltas.shape[1:]`, not `deltas[0]`: an empty selection yields a
    # (0, N) tensor with no row 0 to copy.
    running = torch.zeros(deltas.shape[1:], dtype=deltas.dtype, device=deltas.device)
    for t in range(deltas.shape[0] - 1, -1, -1):
        running = deltas[t] + lamda * running
        advantages[t] = running
    return advantages
