"""Extended tests for gfn.containers: base, transitions, replay_buffer."""

import os
import tempfile

import pytest
import torch

from gfn.containers.replay_buffer import ReplayBuffer, TerminatingStateBuffer
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
# Container base — save/load (B2 TDD)
# ---------------------------------------------------------------------------


def test_container_save_load_roundtrip(env, sample_trajectories):
    """Trajectories should survive a save/load roundtrip."""
    trajs = sample_trajectories
    transitions = trajs.to_transitions()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "transitions")
        os.makedirs(path)
        transitions.save(path)
        # Create a fresh empty transitions object
        loaded = Transitions(env=env)
        loaded.load(path)
        assert loaded.n_transitions == transitions.n_transitions


def test_container_save_with_none_attributes(env, sample_trajectories):
    """B2: save() should not crash on containers with None/bool/int attrs."""
    trajs = sample_trajectories
    transitions = trajs.to_transitions()
    # transitions has: is_backward (bool), _log_rewards (None or Tensor),
    # log_probs (None or Tensor)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "transitions")
        os.makedirs(path)
        # This should not raise ValueError for non-Tensor attributes
        transitions.save(path)


# ---------------------------------------------------------------------------
# Container base — sample, has_log_probs
# ---------------------------------------------------------------------------


def test_container_sample_returns_correct_size(env, sample_trajectories):
    trajs = sample_trajectories
    transitions = trajs.to_transitions()
    n = min(3, len(transitions))
    sampled = transitions.sample(n)
    assert len(sampled) == n


def test_container_has_log_probs_false_when_none(env):
    transitions = Transitions(env=env)
    assert transitions.has_log_probs is False


def test_container_has_log_probs_true_when_present(env, sample_trajectories):
    trajs = sample_trajectories
    # trajs have log_probs since save_logprobs=True
    assert trajs.has_log_probs is True


# ---------------------------------------------------------------------------
# Transitions.__repr__ (B3 TDD)
# ---------------------------------------------------------------------------


def test_transitions_repr_basic(env, sample_trajectories):
    """Transitions.__repr__ should return a string without crashing."""
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
    """ReplayBuffer.initialize() should return the correct empty container type."""
    rb = ReplayBuffer(env=env, capacity=100)
    result = rb.initialize(sample_trajectories)
    # initialize returns an empty container of the same type
    assert result is not None


def test_replay_buffer_add_and_sample(env, sample_trajectories):
    rb = ReplayBuffer(env=env, capacity=100)
    rb.add(sample_trajectories)
    assert len(rb) == len(sample_trajectories)
    sampled = rb.sample(4)
    assert len(sampled) == 4


# ---------------------------------------------------------------------------
# TerminatingStateBuffer
# ---------------------------------------------------------------------------


def test_terminating_state_buffer_init_and_add(env, sample_trajectories):
    buf = TerminatingStateBuffer(env=env, capacity=100)
    assert len(buf) == 0
    buf.add(sample_trajectories)
    assert len(buf) > 0
    # Should contain terminating states
    assert buf.training_container is not None
