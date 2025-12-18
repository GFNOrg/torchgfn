import math

import torch

from gfn.estimators import PinnedBrownianMotionBackward, PinnedBrownianMotionForward
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.samplers import Sampler
from gfn.utils.modules import DiffusionPISGradNetBackward


class _Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ConstantJoint(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(
        self, s_emb: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:  # noqa: ARG002
        batch = s_emb.shape[0]
        return self.output.expand(batch, -1)  # type: ignore


class _ConstantModule(torch.nn.Module):
    def __init__(self, output: torch.Tensor, input_dim: int):
        super().__init__()
        self.register_buffer("output", output)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        batch = x.shape[0]
        return self.output.expand(batch, -1)  # type: ignore


def test_diffusion_pis_gradnet_backward_scales_and_clamps_outputs():
    s_dim = 2
    pb_scale_range = 0.2
    log_var_range = 0.05
    model = DiffusionPISGradNetBackward(
        s_dim=s_dim,
        harmonics_dim=4,
        t_emb_dim=4,
        s_emb_dim=4,
        hidden_dim=8,
        joint_layers=1,
        pb_scale_range=pb_scale_range,
        log_var_range=log_var_range,
        learn_variance=True,
    )

    # Replace heavy components with deterministic stubs.
    model.s_model = _Identity()
    model.t_model = _Identity()
    model.joint_model = _ConstantJoint(
        torch.tensor([3.0, -4.0, 50.0], dtype=torch.float32)
    )

    preprocessed = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)
    out = model(preprocessed)

    drift = out[..., :s_dim]
    log_std = out[..., -1]

    assert out.shape == (1, s_dim + 1)
    assert torch.all(torch.abs(drift) <= pb_scale_range + 1e-6)
    assert torch.allclose(
        drift[0, 0],
        torch.tanh(torch.tensor(3.0)) * pb_scale_range,
        atol=1e-4,
    )
    # Log-std correction is tanh-bounded then clamped to log_var_range.
    assert torch.allclose(log_std, torch.full_like(log_std, log_var_range))


def test_pinned_brownian_forward_marks_exit_on_final_step():
    s_dim = 2
    num_steps = 4
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
        sigma=1.0,
        num_discretization_steps=num_steps,
    )

    # t + dt reaches terminal time, so the drift should be converted to exit action (-inf).
    terminal_states = env.states_from_tensor(
        torch.tensor([[0.0, 0.0, 1.0 - pf.dt]], dtype=torch.float32)
    )
    dist = pf.to_probability_distribution(terminal_states, pf(terminal_states))
    assert torch.isinf(dist.loc).all()

    # Earlier times should stay finite.
    mid_states = env.states_from_tensor(
        torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)
    )
    mid_dist = pf.to_probability_distribution(mid_states, pf(mid_states))
    assert torch.isfinite(mid_dist.loc).all()


def test_pinned_brownian_forward_exit_condition_matches_steps():
    """Exit masking triggers only on last step according to is_final_step logic."""
    s_dim = 2
    num_steps = 5
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
        sigma=1.0,
        num_discretization_steps=num_steps,
    )

    dt = pf.dt
    eps = dt * 1e-2  # _DIFFUSION_TERMINAL_TIME_EPS
    times = torch.tensor(
        [
            0.0,  # initial
            dt,  # early
            1.0 - 2 * dt,  # mid
            1.0 - dt - 0.5 * eps,  # should trigger final step mask
            1.0 - dt,  # last step before terminal time
        ],
        dtype=torch.float32,
    )
    states = env.states_from_tensor(
        torch.stack([torch.zeros_like(times), torch.zeros_like(times), times], dim=1)
    )

    dist = pf.to_probability_distribution(states, pf(states))
    exit_mask = torch.isinf(dist.loc).all(dim=-1)
    expected = torch.tensor([False, False, False, True, True])
    assert torch.equal(exit_mask, expected)


def test_pinned_brownian_forward_combines_exploration_variance():
    s_dim = 2
    num_steps = 5
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 1},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pf_module = _ConstantModule(
        output=torch.zeros(1, s_dim + 1, dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pf = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
        n_variance_outputs=1,
    )

    states = env.states_from_tensor(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32))
    base_std = math.sqrt(pf.dt)  # log_std=0 -> exp(0) * sqrt(dt)
    exploration_std = 0.4
    dist = pf.to_probability_distribution(
        states, pf(states), exploration_std=exploration_std
    )

    expected = math.sqrt(base_std**2 + exploration_std**2)
    assert torch.allclose(dist.scale, torch.full_like(dist.scale, expected), atol=1e-6)


def test_pinned_brownian_backward_applies_corrections_and_quadrature():
    s_dim = 2
    num_steps = 4
    pb_scale_range = 0.2
    sigma = 1.5
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 2},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pb_module = _ConstantModule(
        output=torch.tensor([[5.0, -5.0, 1.0]], dtype=torch.float32),
        input_dim=s_dim + 1,
    )
    pb = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=sigma,
        num_discretization_steps=num_steps,
        n_variance_outputs=1,
        pb_scale_range=pb_scale_range,
    )

    t_curr = 0.5
    states = env.states_from_tensor(
        torch.tensor([[0.5, -0.25, t_curr]], dtype=torch.float32)
    )
    dist = pb.to_probability_distribution(states, pb(states))

    dt = pb.dt
    s_curr = states.tensor[:, :-1]
    base_mean = s_curr * dt / t_curr
    base_std = sigma * math.sqrt(dt * (t_curr - dt) / t_curr)

    expected_mean = base_mean + torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    expected_std = math.sqrt(base_std**2 + math.exp(pb_scale_range) ** 2)

    assert torch.allclose(dist.loc, expected_mean, atol=1e-6)
    assert torch.allclose(
        dist.scale, torch.full_like(dist.scale, expected_std), atol=1e-6
    )


def test_diffusion_sampler_completes_after_num_steps():
    num_steps = 6
    batch_size = 3
    s_dim = 2
    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 3},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
        debug=True,
    )
    pf_module = _ConstantModule(
        output=torch.zeros(1, s_dim, dtype=torch.float32), input_dim=s_dim + 1
    )
    pf = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=1.0,
        num_discretization_steps=num_steps,
    )
    sampler = Sampler(estimator=pf)

    trajectories = sampler.sample_trajectories(
        env, n=batch_size, save_logprobs=True, save_estimator_outputs=False
    )

    assert torch.all(trajectories.terminating_idx == num_steps)
    # The sampler uses the estimator output directly (exit action = -inf) so the final
    # state is the sink padding (non-finite). Verify sink detection and exit action.
    final_states = trajectories.states[
        trajectories.terminating_idx, torch.arange(batch_size)
    ]
    assert final_states.is_sink_state.all()
    assert trajectories.actions.is_exit[num_steps - 1].all()
