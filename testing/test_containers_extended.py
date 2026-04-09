"""Extended tests for gfn.containers: save/load, replay_buffer, transitions."""

import tempfile

import pytest
import torch

from gfn.containers.replay_buffer import ReplayBuffer, TerminatingStateBuffer
from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.gym import HyperGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return HyperGrid(ndim=2, height=4, validate_modes=False)


@pytest.fixture
def sample_trajectories(env):
    """Sample a small batch of trajectories from the environment."""
    from gfn.estimators import DiscretePolicyEstimator
    from gfn.preprocessors import KHotPreprocessor
    from gfn.samplers import Sampler
    from gfn.utils.modules import MLP

    torch.manual_seed(42)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)
    pf = DiscretePolicyEstimator(
        module=MLP(input_dim=preproc.output_dim, output_dim=env.n_actions),
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    sampler = Sampler(estimator=pf)
    return sampler.sample_trajectories(env, n=8, save_logprobs=True)


# ---------------------------------------------------------------------------
# TensorDict save/load roundtrips
# ---------------------------------------------------------------------------


def test_transitions_save_load_roundtrip(env, sample_trajectories):
    """Transitions should survive a save/load roundtrip via TensorDict."""
    transitions = sample_trajectories.to_transitions()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/transitions.pt"
        transitions.save(path)
        loaded = Transitions.load(env, path)
        assert loaded.n_transitions == transitions.n_transitions
        assert torch.equal(loaded.states.tensor, transitions.states.tensor)
        assert torch.equal(loaded.actions.tensor, transitions.actions.tensor)
        assert torch.equal(loaded.is_terminating, transitions.is_terminating)
        assert torch.equal(loaded.next_states.tensor, transitions.next_states.tensor)
        assert loaded.is_backward == transitions.is_backward


def test_trajectories_save_load_roundtrip(env, sample_trajectories):
    """Trajectories should survive a save/load roundtrip via TensorDict."""
    trajs = sample_trajectories

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/trajectories.pt"
        trajs.save(path)
        loaded = Trajectories.load(env, path)
        assert loaded.batch_size == trajs.batch_size
        assert torch.equal(loaded.states.tensor, trajs.states.tensor)
        assert torch.equal(loaded.actions.tensor, trajs.actions.tensor)
        assert torch.equal(loaded.terminating_idx, trajs.terminating_idx)
        assert loaded.is_backward == trajs.is_backward


def test_trajectories_save_load_preserves_log_probs(env, sample_trajectories):
    """Log probs should survive the roundtrip."""
    trajs = sample_trajectories
    assert trajs.has_log_probs  # save_logprobs=True in fixture

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/trajs.pt"
        trajs.save(path)
        loaded = Trajectories.load(env, path)
        assert loaded.has_log_probs
        assert loaded.log_probs is not None and trajs.log_probs is not None
        assert torch.equal(loaded.log_probs, trajs.log_probs)


def test_states_container_save_load_roundtrip(env, sample_trajectories):
    """StatesContainer should survive a save/load roundtrip."""

    trajs = sample_trajectories
    # Build a StatesContainer from trajectories (like FMGFlowNet does)
    states = trajs.states
    n = states.batch_shape[-1]
    is_terminating = torch.zeros(n, dtype=torch.bool)
    sc = StatesContainer(env=env, states=states[0], is_terminating=is_terminating)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/states.pt"
        sc.save(path)
        loaded = StatesContainer.load(env, path)
        assert len(loaded) == len(sc)


def test_save_load_without_optional_fields(env):
    """Containers with None log_rewards/log_probs should roundtrip cleanly."""
    transitions = Transitions(env=env)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/empty.pt"
        transitions.save(path)
        loaded = Transitions.load(env, path)
        assert loaded.n_transitions == 0


def test_transitions_save_load_preserves_log_rewards(env, sample_trajectories):
    """Log rewards should survive the roundtrip."""
    transitions = sample_trajectories.to_transitions()
    # Force log_rewards computation
    original_lr = transitions.log_rewards
    assert original_lr is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/transitions.pt"
        transitions.save(path)
        loaded = Transitions.load(env, path)
        assert loaded._log_rewards is not None
        assert torch.allclose(loaded._log_rewards, original_lr)


def test_trajectories_save_load_preserves_log_rewards(env, sample_trajectories):
    """Trajectory log rewards should survive the roundtrip."""
    trajs = sample_trajectories
    original_lr = trajs.log_rewards
    assert original_lr is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/trajs.pt"
        trajs.save(path)
        loaded = Trajectories.load(env, path)
        loaded_lr = loaded.log_rewards
        assert loaded_lr is not None
        assert torch.allclose(loaded_lr, original_lr)


def test_to_tensordict_keys(env, sample_trajectories):
    """Verify the TensorDict has the expected keys."""
    transitions = sample_trajectories.to_transitions()
    td = transitions.to_tensordict()
    keys = set(td.keys())
    assert "states" in keys
    assert "actions" in keys
    assert "is_terminating" in keys
    assert "next_states" in keys
    assert "is_backward" in keys


# ---------------------------------------------------------------------------
# Container base — sample, has_log_probs
# ---------------------------------------------------------------------------


def test_container_sample_returns_correct_size(env, sample_trajectories):
    transitions = sample_trajectories.to_transitions()
    n = min(3, len(transitions))
    sampled = transitions.sample(n)
    assert len(sampled) == n


def test_container_has_log_probs_false_when_none(env):
    transitions = Transitions(env=env)
    assert transitions.has_log_probs is False


def test_container_has_log_probs_true_when_present(env, sample_trajectories):
    assert sample_trajectories.has_log_probs is True


# ---------------------------------------------------------------------------
# Transitions.__repr__
# ---------------------------------------------------------------------------


def test_transitions_repr_basic(env, sample_trajectories):
    transitions = sample_trajectories.to_transitions()
    r = repr(transitions)
    assert "Transitions" in r
    assert "n_transitions" in r


# ---------------------------------------------------------------------------
# Transitions.all_log_rewards
# ---------------------------------------------------------------------------


def test_transitions_all_log_rewards(env, sample_trajectories):
    transitions = sample_trajectories.to_transitions()
    log_rewards = transitions.all_log_rewards
    assert log_rewards.shape == (transitions.n_transitions, 2)
    assert log_rewards.is_floating_point()


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


def test_replay_buffer_initialize(env, sample_trajectories):
    rb = ReplayBuffer(env=env, capacity=100)
    result = rb.initialize(sample_trajectories)
    assert result is not None


def test_replay_buffer_add_and_sample(env, sample_trajectories):
    rb = ReplayBuffer(env=env, capacity=100)
    rb.add(sample_trajectories)
    assert len(rb) == len(sample_trajectories)
    sampled = rb.sample(4)
    assert len(sampled) == 4


def test_replay_buffer_save_load(env, sample_trajectories):
    """ReplayBuffer save/load should roundtrip via the new single-file API."""
    rb = ReplayBuffer(env=env, capacity=100)
    rb.add(sample_trajectories)
    n_before = len(rb)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/buffer.pt"
        rb.save(path)

        rb2 = ReplayBuffer(env=env, capacity=100)
        rb2.add(sample_trajectories)  # initialize the container type
        rb2.load(path)
        assert len(rb2) == n_before


# ---------------------------------------------------------------------------
# TerminatingStateBuffer
# ---------------------------------------------------------------------------


def test_terminating_state_buffer_init_and_add(env, sample_trajectories):
    buf = TerminatingStateBuffer(env=env, capacity=100)
    assert len(buf) == 0
    buf.add(sample_trajectories)
    assert len(buf) > 0
    assert buf.training_container is not None
