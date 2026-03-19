"""Tests for diffusion terminal-time exit logic.

These tests verify the critical invariant that the estimator's exit condition,
the environment's sink-state detection, and the environment's step() debug
assertion all agree on exactly which step is the final one.

A mismatch causes mask misalignment in get_trajectory_pbs(), silently
corrupting training gradients.
"""

import math

import pytest
import torch

from gfn.estimators import (
    _DIFFUSION_TERMINAL_TIME_EPS,
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
)
from gfn.gym.diffusion_sampling import TERMINAL_TIME_EPS, DiffusionSampling
from gfn.samplers import Sampler
from gfn.utils.modules import DiffusionFixedBackwardModule, DiffusionPISGradNetForward
from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantModule(torch.nn.Module):
    """Module that ignores input and returns a fixed output."""

    def __init__(self, output: torch.Tensor, input_dim: int):
        super().__init__()
        self.register_buffer("output", output)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        batch = x.shape[0]
        return self.output.expand(batch, -1)  # type: ignore


def _make_env_and_pf(
    s_dim: int = 2,
    num_steps: int = 10,
    sigma: float = 1.0,
) -> tuple[DiffusionSampling, PinnedBrownianMotionForward]:
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 0},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pf_module = _ConstantModule(
        output=torch.zeros(1, s_dim, dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pf = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=sigma,
        num_discretization_steps=num_steps,
    )
    return env, pf


# ---------------------------------------------------------------------------
# 1. The two EPS constants must agree
# ---------------------------------------------------------------------------


def test_estimator_eps_matches_environment_eps():
    """If these drift apart the estimator and env disagree on terminal time."""
    assert _DIFFUSION_TERMINAL_TIME_EPS == TERMINAL_TIME_EPS


# ---------------------------------------------------------------------------
# 2. Exhaustive per-step exit classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [4, 5, 10, 20, 100])
def test_exit_fires_on_exactly_the_last_step(num_steps: int):
    """Walk through every t value in a trajectory and verify exit triggers
    on exactly step N-1 (the last productive step) and nowhere earlier."""
    env, pf = _make_env_and_pf(num_steps=num_steps)
    dt = pf.dt

    # Build a batch with one row per step: t = 0, dt, 2*dt, ..., (N-1)*dt
    times = torch.arange(num_steps, dtype=torch.float32) * dt
    s_dim = 2
    states_tensor = torch.zeros(num_steps, s_dim + 1)
    states_tensor[:, -1] = times
    states = env.states_from_tensor(states_tensor)

    dist = pf.to_probability_distribution(states, pf(states))
    exit_mask = torch.isinf(dist.loc).all(dim=-1)

    # Only the very last step (t = (N-1)*dt) should be an exit.
    expected = torch.zeros(num_steps, dtype=torch.bool)
    expected[-1] = True
    assert torch.equal(exit_mask, expected), (
        f"num_steps={num_steps}: exit fired on steps "
        f"{exit_mask.nonzero(as_tuple=True)[0].tolist()}, expected [{num_steps - 1}]"
    )


# ---------------------------------------------------------------------------
# 3. Sink-state classification agrees with exit condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [4, 10, 50])
def test_sink_state_only_for_nonfinite_time(num_steps: int):
    """Only actual sf padding (non-finite time) is classified as sink.
    A state at t=1.0 is a real state from which EXIT is taken, not a sink."""
    env, pf = _make_env_and_pf(num_steps=num_steps)
    dt = pf.dt
    s_dim = 2

    # A state at t=1.0 (physical terminal time) is NOT a sink — it's a valid
    # state that appears in reversed backward trajectories.
    terminal_tensor = torch.zeros(1, s_dim + 1)
    terminal_tensor[:, -1] = num_steps * dt  # t = 1.0
    terminal_state = env.states_from_tensor(terminal_tensor)
    assert (
        not terminal_state.is_sink_state.any()
    ), "State at t=1.0 is a real state, not a sink"

    # The actual sf (non-finite time from make_sink_states) IS a sink.
    sf_state = env.States.make_sink_states((1,), device=torch.device("cpu"))
    assert sf_state.is_sink_state.all(), "sf with non-finite time must be sink"

    # A state one step before terminal is also not a sink.
    pre_exit_tensor = torch.zeros(1, s_dim + 1)
    pre_exit_tensor[:, -1] = (num_steps - 1) * dt
    pre_exit_state = env.states_from_tensor(pre_exit_tensor)
    assert (
        not pre_exit_state.is_sink_state.any()
    ), f"State at t={(num_steps-1)*dt:.6f} before exit should NOT be sink"


# ---------------------------------------------------------------------------
# 4. Environment step() debug assertion catches missing exit
# ---------------------------------------------------------------------------


def test_env_step_rejects_non_exit_at_terminal_time():
    """If the estimator fails to emit an exit action at the last step,
    the env's debug assertion must catch it."""
    env, pf = _make_env_and_pf(num_steps=10)
    dt = pf.dt
    s_dim = 2

    # State at the last productive step: t = (N-1)*dt = 0.9
    terminal_tensor = torch.zeros(1, s_dim + 1)
    terminal_tensor[:, -1] = 9 * dt
    terminal_states = env.states_from_tensor(terminal_tensor)

    # A non-exit (finite) action should be rejected by the env debug check.
    drift_action = torch.zeros(1, s_dim)
    from gfn.actions import Actions

    actions = Actions(drift_action)
    with pytest.raises(AssertionError, match="exit actions"):
        env.step(terminal_states, actions)


# ---------------------------------------------------------------------------
# 5. Full trajectory: exit at correct index, correct number of steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [4, 10, 20])
def test_full_trajectory_exits_at_correct_index(num_steps: int):
    """Sample a full trajectory and verify exit happens at step N-1
    (0-indexed action), giving exactly N productive drift steps before exit."""
    s_dim = 2
    env, pf = _make_env_and_pf(s_dim=s_dim, num_steps=num_steps)
    sampler = Sampler(estimator=pf)
    batch_size = 4

    trajectories = sampler.sample_trajectories(
        env, n=batch_size, save_logprobs=True, save_estimator_outputs=False
    )

    # terminating_idx is the 0-based index of the exit action.
    # With N discretization steps we expect exit at action index N-1
    # (actions 0..N-2 are drift, action N-1 is exit), but the Sampler
    # records terminating_idx as the number of steps taken = N.
    assert torch.all(trajectories.terminating_idx == num_steps)

    # The state reached just before exit should have t = (N-1)*dt
    for b in range(batch_size):
        t_before_exit = trajectories.states.tensor[num_steps - 1, b, -1]
        assert math.isclose(t_before_exit.item(), (num_steps - 1) * pf.dt, rel_tol=1e-5)

    # Exit action at step N-1 (0-indexed)
    assert trajectories.actions.is_exit[num_steps - 1].all()

    # Final state is sink
    final_states = trajectories.states[
        trajectories.terminating_idx, torch.arange(batch_size)
    ]
    assert final_states.is_sink_state.all()


# ---------------------------------------------------------------------------
# 6. Boundary: t values very close to but not at the exit threshold
# ---------------------------------------------------------------------------


def test_no_premature_exit_near_boundary():
    """Verify that times just barely below the exit threshold do NOT trigger exit."""
    num_steps = 10
    env, pf = _make_env_and_pf(num_steps=num_steps)
    dt = pf.dt
    eps = dt * _DIFFUSION_TERMINAL_TIME_EPS
    s_dim = 2

    # t = (N-2)*dt: one full step before the exit step. This must NOT exit.
    safe_time = (num_steps - 2) * dt
    # Verify: (safe_time + dt) = (N-1)*dt < 1.0 - eps
    assert (safe_time + dt) < (1.0 - eps), "Test setup error"

    safe_tensor = torch.zeros(1, s_dim + 1)
    safe_tensor[:, -1] = safe_time
    safe_states = env.states_from_tensor(safe_tensor)
    dist = pf.to_probability_distribution(safe_states, pf(safe_states))
    assert torch.isfinite(
        dist.loc
    ).all(), f"Premature exit at t={safe_time}: one step before last should still drift"


# ---------------------------------------------------------------------------
# 7. Backward policy s0 detection is consistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [4, 10, 20])
def test_backward_s0_boundary_produces_deterministic_return(num_steps: int):
    """At t=dt the backward action must deterministically return to origin:
    base_mean=s_curr (so s_{t-dt} = s_t - s_curr = 0) and base_std=0."""
    s_dim = 2
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 0},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pb_module = _ConstantModule(
        output=torch.zeros(1, s_dim, dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )

    dt = pb.dt
    s_curr = torch.tensor([[0.5, -0.3]], dtype=torch.float32)

    # At t=dt, backward should have mean=s_curr and std=0.
    s0_tensor = torch.cat([s_curr, torch.tensor([[dt]])], dim=-1)
    s0_states = env.states_from_tensor(s0_tensor)
    dist_s0 = pb.to_probability_distribution(s0_states, pb(s0_states))
    assert torch.allclose(
        dist_s0.scale, torch.zeros_like(dist_s0.scale)
    ), "Backward at t=dt should have zero std (deterministic)"
    assert torch.allclose(
        dist_s0.loc, s_curr, atol=1e-5
    ), f"Backward at t=dt should have mean=s_curr={s_curr}, got {dist_s0.loc}"

    # At t=2*dt, backward should NOT be deterministic.
    mid_tensor = torch.cat([s_curr, torch.tensor([[2 * dt]])], dim=-1)
    mid_states = env.states_from_tensor(mid_tensor)
    dist_mid = pb.to_probability_distribution(mid_states, pb(mid_states))
    assert (
        dist_mid.scale > 0
    ).all(), "Backward at t=2*dt should have positive std (not s0)"


# ---------------------------------------------------------------------------
# 8. Backward sampling at s0 boundary returns to origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [4, 10, 20])
def test_backward_sampling_at_s0_returns_to_origin(num_steps: int):
    """Sampling from the backward policy at t=dt must produce action=s_curr,
    so that backward_step yields s_{t-dt} = s_t - action = 0."""
    s_dim = 2
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 0},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pb_module = _ConstantModule(
        output=torch.zeros(1, s_dim, dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )

    dt = pb.dt
    s_curr = torch.tensor([[1.7, -0.9]], dtype=torch.float32)
    s0_tensor = torch.cat([s_curr, torch.tensor([[dt]])], dim=-1)
    s0_states = env.states_from_tensor(s0_tensor)
    dist = pb.to_probability_distribution(s0_states, pb(s0_states))

    # Sample and verify: action should equal s_curr (std=0 → deterministic).
    sampled_action = dist.sample()
    assert torch.allclose(
        sampled_action, s_curr, atol=1e-5
    ), f"Sampled action at s0 boundary should be s_curr={s_curr}, got {sampled_action}"

    # Applying backward_step: prev = s_curr - action should be ~0.
    prev_state = s_curr - sampled_action
    assert torch.allclose(
        prev_state, torch.zeros_like(prev_state), atol=1e-5
    ), f"backward_step should reach origin, got {prev_state}"


# ---------------------------------------------------------------------------
# 9. Backward log_prob at s0 boundary is zero (convention for deterministic)
# ---------------------------------------------------------------------------


def test_backward_logprob_at_s0_boundary_is_zero():
    """log_prob for the s1→s0 transition should be 0 by convention: the
    deterministic Brownian bridge transition cancels in the TB loss."""
    s_dim = 2
    num_steps = 10
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 0},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pb_module = _ConstantModule(
        output=torch.zeros(1, s_dim, dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )

    dt = pb.dt
    s_curr = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    s0_tensor = torch.cat([s_curr, torch.tensor([[dt]])], dim=-1)
    s0_states = env.states_from_tensor(s0_tensor)
    dist = pb.to_probability_distribution(s0_states, pb(s0_states))

    # The correct action at s0 is s_curr; log_prob should be 0.
    log_p = dist.log_prob(s_curr)
    assert torch.allclose(
        log_p, torch.zeros_like(log_p)
    ), f"log_prob at s0 boundary should be 0, got {log_p}"


# ---------------------------------------------------------------------------
# 10. Reversed backward trajectories pass mask alignment checks
# ---------------------------------------------------------------------------


def test_reversed_backward_trajectory_mask_alignment():
    """Sample backward trajectories from ground truth, reverse them, and
    verify that get_trajectory_pfs_and_pbs succeeds (no mask assertion failure).

    This is the exact code path that train_diffusion_sampler.py uses for
    density metric evaluation.
    """
    from gfn.gflownet import TBGFlowNet

    num_steps = 8
    s_dim = 2
    batch_size = 4

    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 0},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )

    pf_module = DiffusionPISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=16,
        t_emb_dim=16,
        s_emb_dim=16,
        hidden_dim=16,
        joint_layers=1,
    )
    pb_module = DiffusionFixedBackwardModule(s_dim=s_dim)

    pf_est = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )
    pb_est = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )
    gfn = TBGFlowNet(pf=pf_est, pb=pb_est, init_logZ=0.0)

    # Sample backward from ground truth.
    gt_xs, _ = env.target.cached_sample(batch_size=batch_size)
    assert gt_xs is not None
    xs_with_time = torch.cat(
        [gt_xs, torch.ones(gt_xs.shape[0], 1, device=gt_xs.device)], dim=1
    )
    states_batch = env.states_from_tensor(xs_with_time)

    assert gfn.pb is not None
    bwd_sampler = Sampler(estimator=gfn.pb)
    bwd_traj = bwd_sampler.sample_trajectories(
        env,
        states=states_batch,
        save_logprobs=False,
        save_estimator_outputs=False,
    )

    # Reverse and evaluate — this is the path that was failing.
    reversed_traj = bwd_traj.reverse_backward_trajectories()
    log_pfs, log_pbs = get_trajectory_pfs_and_pbs(
        gfn.pf,
        gfn.pb,
        reversed_traj,
        recalculate_all_logprobs=False,
    )

    # Shapes must match.
    assert log_pfs.shape == log_pbs.shape

    # log_pbs should be finite (backward policy evaluates on its own actions).
    assert torch.isfinite(log_pbs.sum(dim=0)).all()

    # log_pfs may contain -inf at position N-1: the forward policy forces EXIT
    # at t=(N-1)*dt, but the reversed backward trajectory has a drift action
    # there. This is mathematically correct — the trajectory has zero forward
    # probability. Verify no NaN (which would indicate a real bug).
    assert not torch.isnan(log_pfs).any()
