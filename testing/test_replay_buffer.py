from tempfile import TemporaryDirectory

import pytest
import torch

from gfn.containers.replay_buffer import (
    NormBasedDiversePrioritizedReplayBuffer,
    ReplayBuffer,
)
from gfn.containers.state_pairs import StatePairs
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.gym.hypergrid import HyperGrid


@pytest.fixture
def simple_env():
    return HyperGrid()


@pytest.fixture
def trajectories(simple_env):
    # Create a batch of 5 trajectories
    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    # Create some simple trajectories
    traj = Trajectories(
        simple_env,
        states=state_class(torch.randn(10, 5, 2)),
        actions=action_class(torch.ones(10, 5, 1)),
        log_probs=torch.ones(10, 5),
        log_rewards=torch.arange(5, dtype=torch.float),
        terminating_idx=torch.randint(0, 5, (5,), dtype=torch.long),
    )

    return traj


@pytest.fixture
def transitions(simple_env):
    # Create a batch of transitions
    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    trans = Transitions(
        simple_env,
        states=state_class(torch.randn(5, 2)),
        next_states=state_class(torch.randn(5, 2)),
        actions=action_class(torch.ones(5, 1)),
        log_probs=torch.zeros(5),
        log_rewards=torch.ones((5,), dtype=torch.float),
        is_done=torch.zeros(5, dtype=torch.bool),
    )

    return trans


@pytest.fixture
def state_pairs(simple_env):
    # Create a batch of state pairs
    state_class = simple_env.make_states_class()

    pairs = StatePairs(
        simple_env,
        intermediary_states=state_class(torch.randn(5, 2)),
        terminating_states=state_class(torch.randn(5, 2)),
        log_rewards=torch.ones((5), dtype=torch.float),
    )

    return pairs


def test_init_rb(simple_env):
    buffer = ReplayBuffer(simple_env, capacity=100)
    assert buffer.capacity == 100
    assert buffer.env == simple_env
    assert buffer.training_objects is None
    assert len(buffer) == 0
    assert not buffer.prioritized

    # Test representation
    assert "ReplayBuffer" in repr(buffer)
    assert "empty" in repr(buffer)


def test_add_trajectories(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(trajectories)

    assert buffer.training_objects is not None
    assert isinstance(buffer.training_objects, Trajectories)
    assert len(buffer) == 5
    assert "trajectories" in repr(buffer)


def test_add_transitions(simple_env, transitions):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(transitions)

    assert buffer.training_objects is not None
    assert isinstance(buffer.training_objects, Transitions)
    assert len(buffer) == 5
    assert "transitions" in repr(buffer)


def test_add_state_pairs(simple_env, state_pairs):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(state_pairs)

    assert buffer.training_objects is not None
    assert isinstance(buffer.training_objects, StatePairs)
    assert len(buffer) == 5
    assert "statepairs" in repr(buffer)


def test_capacity_limit(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=3)
    buffer.add(trajectories)

    assert len(buffer) == 3
    # Should keep the last 3 trajectories
    assert isinstance(buffer.training_objects, Trajectories)
    assert isinstance(buffer.training_objects.log_rewards, torch.Tensor)
    assert buffer.training_objects.log_rewards[-3:].tolist() == [2.0, 3.0, 4.0]


def test_prioritized(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=5, prioritized=True)
    buffer.add(trajectories)

    assert buffer.prioritized
    assert len(buffer) == 5
    # Should sort by log_rewards in ascending order
    assert isinstance(buffer.training_objects, Trajectories)
    assert isinstance(buffer.training_objects.log_rewards, torch.Tensor)
    assert torch.allclose(
        buffer.training_objects.log_rewards, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    )


def test_sample(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(trajectories)

    # Sample 3 trajectories
    sampled = buffer.sample(3)
    assert len(sampled) == 3
    assert isinstance(sampled, Trajectories)

    # Test sampling with empty buffer
    empty_buffer = ReplayBuffer(simple_env)
    with pytest.raises(ValueError, match="Buffer is empty"):
        empty_buffer.sample(1)


@pytest.mark.skip(reason="Save is broken")  # TODO: Fix saving and loading
def test_save_load(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(trajectories)

    with TemporaryDirectory() as tmpdir:
        # Save the buffer
        buffer.save(tmpdir)

        # Create a new buffer and load
        new_buffer = ReplayBuffer(simple_env, capacity=10)
        new_buffer.add(trajectories[:1])  # Add one trajectory to initialize
        new_buffer.load(tmpdir)

        # Check if loaded correctly
        assert len(new_buffer) == 5
        assert isinstance(new_buffer.training_objects, Trajectories)
        assert isinstance(buffer.training_objects, Trajectories)
        assert isinstance(new_buffer.training_objects.log_rewards, torch.Tensor)
        assert isinstance(buffer.training_objects.log_rewards, torch.Tensor)
        assert torch.allclose(
            new_buffer.training_objects.log_rewards, buffer.training_objects.log_rewards
        )


def test_type_error(simple_env):
    buffer = ReplayBuffer(simple_env)

    # Try to add an invalid type
    with pytest.raises(TypeError, match="Must be a container type"):
        buffer.add("not a container")  # type: ignore


def test_init_ndrb(simple_env):
    buffer = NormBasedDiversePrioritizedReplayBuffer(
        simple_env, capacity=100, cutoff_distance=0.5, p_norm_distance=2.0
    )

    assert buffer.capacity == 100
    assert buffer.cutoff_distance == 0.5
    assert buffer.p_norm_distance == 2.0
    assert buffer.prioritized


def test_add_with_diversity(simple_env, trajectories):
    buffer = NormBasedDiversePrioritizedReplayBuffer(
        simple_env,
        capacity=5,
        cutoff_distance=0.5,
    )

    buffer.add(trajectories)
    assert len(buffer) == 5

    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    # Create similar trajectories
    similar_trajs = Trajectories(simple_env)
    for i in range(3):
        new_traj = Trajectories(
            simple_env,
            states=state_class(torch.randn(1, 1, 2)),
            actions=action_class(torch.ones(1, 1, 1)),
            log_probs=torch.zeros(1, 1),
            log_rewards=torch.tensor([i + 10], dtype=torch.float),
            terminating_idx=torch.tensor([1], dtype=torch.long),
        )
        similar_trajs.extend(new_traj)

    # Add similar trajectories - they should be filtered by diversity
    buffer.add(similar_trajs)

    # The buffer should still prioritize by reward within the diverse set
    assert len(buffer) == 5
    assert isinstance(buffer.training_objects, Trajectories)
    assert isinstance(buffer.training_objects.log_rewards, torch.Tensor)
    assert torch.all(buffer.training_objects.log_rewards >= 0)


def test_skip_diversity_check(simple_env, trajectories):
    # With negative cutoff, diversity check is skipped
    buffer = NormBasedDiversePrioritizedReplayBuffer(
        simple_env,
        capacity=5,
        cutoff_distance=-1.0,  # Negative cutoff skips diversity check
    )

    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    buffer.add(trajectories)
    assert len(buffer) == 5

    # Create trajectories with higher rewards
    better_trajs = Trajectories(simple_env)
    for i in range(3):
        new_traj = Trajectories(
            simple_env,
            states=state_class(torch.randn(1, 1, 2)),
            actions=action_class(torch.ones(1, 1, 1)),
            log_probs=torch.zeros(1, 1),
            log_rewards=torch.tensor([i + 10], dtype=torch.float),
            terminating_idx=torch.tensor([1], dtype=torch.long),
        )
        better_trajs.extend(new_traj)
    # Add better trajectories - they should replace lower reward ones
    buffer.add(better_trajs)

    # The buffer should contain the highest reward trajectories
    assert len(buffer) == 5
    assert isinstance(buffer.training_objects, Trajectories)
    assert isinstance(buffer.training_objects.log_rewards, torch.Tensor)
    assert torch.min(buffer.training_objects.log_rewards) >= 2.0
