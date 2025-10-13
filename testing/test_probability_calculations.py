import pytest
import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gym import HyperGrid
from gfn.preprocessors import IdentityPreprocessor
from gfn.samplers import DefaultEstimatorAdapter, Sampler
from gfn.utils.handlers import check_cond_forward
from gfn.utils.prob_calculations import (
    get_trajectory_pbs,
    get_trajectory_pfs,
    get_transition_pbs,
    get_transition_pfs,
)


class NonVectorizedDefaultAdapter(DefaultEstimatorAdapter):
    @property
    def is_vectorized(self) -> bool:  # type: ignore[override]
        return False


def _legacy_get_trajectory_pfs(
    pf: DiscretePolicyEstimator,
    trajectories,
    *,
    fill_value: float = 0.0,
    recalculate_all_logprobs: bool = True,
):
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    state_mask = ~trajectories.states.is_sink_state
    action_mask = ~trajectories.actions.is_dummy

    valid_states = trajectories.states[state_mask]
    valid_actions = trajectories.actions[action_mask]

    if valid_states.batch_shape != valid_actions.batch_shape:
        raise AssertionError("Something wrong happening with log_pf evaluations")

    log_pf_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value,
        dtype=torch.get_default_dtype(),
    )

    if len(valid_states) == 0:
        return log_pf_trajectories

    if trajectories.estimator_outputs is not None and not recalculate_all_logprobs:
        estimator_outputs = trajectories.estimator_outputs[action_mask]
    else:
        masked_cond = None
        if trajectories.conditioning is not None:
            cond_dim = (-1,) * len(trajectories.conditioning.shape)
            traj_len = trajectories.states.tensor.shape[0]
            masked_cond = trajectories.conditioning.unsqueeze(0).expand(
                (traj_len,) + cond_dim
            )[state_mask]

        estimator_outputs = check_cond_forward(pf, "pf", valid_states, masked_cond)

    valid_log_pf_actions = pf.to_probability_distribution(
        valid_states, estimator_outputs
    ).log_prob(valid_actions.tensor)

    log_pf_trajectories[action_mask] = valid_log_pf_actions.to(
        log_pf_trajectories.dtype, copy=False
    )

    assert log_pf_trajectories.shape == (
        trajectories.max_length,
        trajectories.n_trajectories,
    )
    return log_pf_trajectories


def _build_env_and_pf(n: int = 4):
    env = HyperGrid(ndim=2, height=4)
    preprocessor = IdentityPreprocessor(
        output_dim=env.state_shape[-1], target_dtype=torch.get_default_dtype()
    )
    pf_module = torch.nn.Sequential(
        torch.nn.Linear(preprocessor.output_dim, 16),  # type: ignore
        torch.nn.ReLU(),
        torch.nn.Linear(16, env.n_actions),
    )
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    sampler = Sampler(estimator=pf_estimator)

    return env, pf_estimator, sampler


@pytest.mark.parametrize("use_cached_outputs", [True, False])
def test_get_trajectory_pfs_matches_legacy_with_default_adapter(
    use_cached_outputs: bool,
):
    env, pf_estimator, sampler = _build_env_and_pf()

    trajectories = sampler.sample_trajectories(
        env,
        n=5,
        save_estimator_outputs=use_cached_outputs,
        save_logprobs=False,
    )

    # Legacy calculation
    legacy = _legacy_get_trajectory_pfs(
        pf_estimator,
        trajectories,
        fill_value=0.0,
        recalculate_all_logprobs=not use_cached_outputs,
    )

    # Adapter-backed calculation
    adapter = DefaultEstimatorAdapter(pf_estimator)
    modern = get_trajectory_pfs(
        pf_estimator,
        trajectories,
        fill_value=0.0,
        recalculate_all_logprobs=not use_cached_outputs,
        adapter=adapter,
    )

    torch.testing.assert_close(modern, legacy)


def _legacy_get_trajectory_pbs(
    pb: DiscretePolicyEstimator | None,
    trajectories,
    *,
    fill_value: float = 0.0,
):
    if trajectories.is_backward:
        raise ValueError("Backward trajectories are not supported")

    log_pb_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value,
        dtype=torch.get_default_dtype(),
    )

    state_mask = (
        ~trajectories.states.is_sink_state & ~trajectories.states.is_initial_state
    )
    state_mask[0, :] = False
    action_mask = ~trajectories.actions.is_dummy & ~trajectories.actions.is_exit

    valid_states = trajectories.states[state_mask]
    valid_actions = trajectories.actions[action_mask]

    if valid_states.batch_shape != valid_actions.batch_shape:
        raise AssertionError("Something wrong happening with log_pf evaluations")

    if len(valid_states) == 0:
        return log_pb_trajectories

    masked_cond = None
    if trajectories.conditioning is not None:
        masked_cond = trajectories.conditioning[state_mask]

    if pb is not None:
        estimator_outputs = check_cond_forward(pb, "pb", valid_states, masked_cond)
        valid_log_pb_actions = pb.to_probability_distribution(
            valid_states, estimator_outputs
        ).log_prob(valid_actions.tensor)
    else:
        valid_log_pb_actions = torch.zeros_like(valid_actions.tensor)

    log_pb_trajectories[action_mask] = valid_log_pb_actions.to(
        log_pb_trajectories.dtype, copy=False
    )

    assert log_pb_trajectories.shape == (
        trajectories.max_length,
        trajectories.n_trajectories,
    )
    return log_pb_trajectories


def _build_env_pf_pb():
    env = HyperGrid(ndim=2, height=4)
    preprocessor = IdentityPreprocessor(
        output_dim=env.state_shape[-1], target_dtype=torch.get_default_dtype()
    )
    pf_module = torch.nn.Sequential(
        torch.nn.Linear(preprocessor.output_dim, 16),  # type: ignore
        torch.nn.ReLU(),
        torch.nn.Linear(16, env.n_actions),
    )
    pb_module = torch.nn.Sequential(
        torch.nn.Linear(preprocessor.output_dim, 16),  # type: ignore
        torch.nn.ReLU(),
        torch.nn.Linear(16, env.n_actions - 1),
    )
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )
    pf_sampler = Sampler(estimator=pf_estimator)
    return env, pf_estimator, pb_estimator, pf_sampler


def test_get_trajectory_pbs_matches_legacy_with_default_adapter():
    env, _, pb_estimator, pf_sampler = _build_env_pf_pb()

    trajectories = pf_sampler.sample_trajectories(
        env,
        n=6,
        save_estimator_outputs=False,
        save_logprobs=False,
    )

    legacy = _legacy_get_trajectory_pbs(pb_estimator, trajectories, fill_value=0.0)

    adapter = DefaultEstimatorAdapter(pb_estimator)
    modern = get_trajectory_pbs(
        pb_estimator,
        trajectories,
        fill_value=0.0,
        adapter=adapter,
    )

    torch.testing.assert_close(modern, legacy)


@pytest.mark.parametrize("use_cached_outputs", [True, False])
def test_trajectory_pf_vectorized_vs_nonvectorized_parity(use_cached_outputs: bool):
    env, pf_estimator, sampler = _build_env_and_pf()

    trajectories = sampler.sample_trajectories(
        env,
        n=5,
        save_estimator_outputs=use_cached_outputs,
        save_logprobs=False,
    )

    # Vectorized (legacy) path: adapter None triggers vectorized
    vec = get_trajectory_pfs(
        pf_estimator,
        trajectories,
        recalculate_all_logprobs=not use_cached_outputs,
        adapter=None,
    )

    # Non-vectorized path: force via NonVectorizedDefaultAdapter
    nvec = get_trajectory_pfs(
        pf_estimator,
        trajectories,
        recalculate_all_logprobs=not use_cached_outputs,
        adapter=NonVectorizedDefaultAdapter(pf_estimator),
    )

    torch.testing.assert_close(vec, nvec)


def test_trajectory_pb_vectorized_vs_nonvectorized_parity():
    env, _, pb_estimator, pf_sampler = _build_env_pf_pb()

    trajectories = pf_sampler.sample_trajectories(
        env,
        n=6,
        save_estimator_outputs=False,
        save_logprobs=False,
    )

    # Vectorized
    vec = get_trajectory_pbs(pb_estimator, trajectories, adapter=None)
    # Non-vectorized forced
    nvec = get_trajectory_pbs(
        pb_estimator, trajectories, adapter=NonVectorizedDefaultAdapter(pb_estimator)
    )

    torch.testing.assert_close(vec, nvec)


def test_transition_pf_vectorized_vs_nonvectorized_parity():
    env, pf_estimator, _, pf_sampler = _build_env_pf_pb()
    trajectories = pf_sampler.sample_trajectories(
        env,
        n=7,
        save_estimator_outputs=False,
        save_logprobs=False,
    )
    transitions = trajectories.to_transitions()

    vec = get_transition_pfs(
        pf_estimator, transitions, recalculate_all_logprobs=True, adapter=None
    )
    nvec = get_transition_pfs(
        pf_estimator,
        transitions,
        recalculate_all_logprobs=True,
        adapter=NonVectorizedDefaultAdapter(pf_estimator),
    )
    torch.testing.assert_close(vec, nvec)


def test_transition_pb_vectorized_vs_nonvectorized_parity():
    env, _, pb_estimator, pf_sampler = _build_env_pf_pb()
    trajectories = pf_sampler.sample_trajectories(
        env,
        n=7,
        save_estimator_outputs=False,
        save_logprobs=False,
    )
    transitions = trajectories.to_transitions()

    vec = get_transition_pbs(pb_estimator, transitions, adapter=None)
    nvec = get_transition_pbs(
        pb_estimator, transitions, adapter=NonVectorizedDefaultAdapter(pb_estimator)
    )
    torch.testing.assert_close(vec, nvec)


def test_adapter_log_prob_of_actions_precomputed_matches_forward():
    env, pf_estimator, _ = _build_env_and_pf()
    states = env.reset(batch_shape=(5,))

    # Compute estimator outputs once (precomputed path)
    est_out = check_cond_forward(pf_estimator, "pf", states, None)
    dist = pf_estimator.to_probability_distribution(states, est_out)
    with torch.no_grad():
        actions_tensor = dist.sample()

    adapter = DefaultEstimatorAdapter(pf_estimator)
    ctx1 = adapter.init_context(batch_size=5, device=states.device, conditioning=None)
    ctx2 = adapter.init_context(batch_size=5, device=states.device, conditioning=None)
    step_mask = torch.ones(5, dtype=torch.bool, device=states.device)

    # Baseline: adapter recomputes estimator outputs internally
    lp1, _ = adapter.log_prob_of_actions(states, actions_tensor, ctx1, step_mask)

    # Precomputed: adapter uses provided estimator outputs (fast path).
    ctx2.current_estimator_output = est_out
    lp2, _ = adapter.log_prob_of_actions(states, actions_tensor, ctx2, step_mask)

    torch.testing.assert_close(lp1, lp2)


def _legacy_get_transition_pfs(
    pf: DiscretePolicyEstimator,
    transitions,
    *,
    recalculate_all_logprobs: bool = False,
):
    states = transitions.states
    actions = transitions.actions

    if transitions.has_log_probs and recalculate_all_logprobs is False:
        log_pf_actions = transitions.log_probs
        assert log_pf_actions is not None
        return log_pf_actions

    estimator_outputs = check_cond_forward(pf, "pf", states, transitions.conditioning)
    log_pf_actions = pf.to_probability_distribution(states, estimator_outputs).log_prob(
        actions.tensor
    )
    return log_pf_actions


def _legacy_get_transition_pbs(pb: DiscretePolicyEstimator | None, transitions):
    valid_next_states = transitions.next_states[~transitions.is_terminating]
    non_exit_actions = transitions.actions[~transitions.actions.is_exit]
    masked_cond = (
        transitions.conditioning[~transitions.is_terminating]
        if transitions.conditioning is not None
        else None
    )

    log_pb_actions = torch.zeros(
        (transitions.n_transitions,), device=transitions.states.device
    )

    if pb is not None:
        estimator_outputs = check_cond_forward(pb, "pb", valid_next_states, masked_cond)
        valid_log_pb_actions = pb.to_probability_distribution(
            valid_next_states, estimator_outputs
        ).log_prob(non_exit_actions.tensor)
        if len(valid_next_states) != 0:
            log_pb_actions[~transitions.is_terminating] = valid_log_pb_actions

    return log_pb_actions


def test_get_transition_pfs_matches_legacy_with_default_adapter():
    env, pf_estimator, _, pf_sampler = _build_env_pf_pb()
    trajectories = pf_sampler.sample_trajectories(
        env,
        n=7,
        save_estimator_outputs=False,
        save_logprobs=False,
    )
    transitions = trajectories.to_transitions()

    legacy = _legacy_get_transition_pfs(pf_estimator, transitions)
    modern = get_transition_pfs(
        pf_estimator,
        transitions,
        recalculate_all_logprobs=True,
        adapter=DefaultEstimatorAdapter(pf_estimator),
    )
    torch.testing.assert_close(modern, legacy)


def test_get_transition_pbs_matches_legacy_with_default_adapter():
    env, _, pb_estimator, pf_sampler = _build_env_pf_pb()
    trajectories = pf_sampler.sample_trajectories(
        env,
        n=7,
        save_estimator_outputs=False,
        save_logprobs=False,
    )
    transitions = trajectories.to_transitions()

    legacy = _legacy_get_transition_pbs(pb_estimator, transitions)
    modern = get_transition_pbs(
        pb_estimator,
        transitions,
        adapter=DefaultEstimatorAdapter(pb_estimator),
    )
    torch.testing.assert_close(modern, legacy)


if __name__ == "__main__":
    test_trajectory_pb_vectorized_vs_nonvectorized_parity()
