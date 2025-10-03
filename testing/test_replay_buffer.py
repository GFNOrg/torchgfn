from tempfile import TemporaryDirectory

import pytest
import torch

from gfn.containers.replay_buffer import (
    NormBasedDiversePrioritizedReplayBuffer,
    ReplayBuffer,
)
from gfn.containers.states_container import StatesContainer
from gfn.containers.trajectories import Trajectories
from gfn.containers.transitions import Transitions
from gfn.gym.hypergrid import ConditionalHyperGrid, HyperGrid


@pytest.fixture
def simple_env():
    return HyperGrid(validate_modes=True)


@pytest.fixture
def trajectories(simple_env):
    # Create a batch of 5 trajectories
    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    # Create some simple trajectories
    traj = Trajectories(
        simple_env,
        states=state_class(torch.randn(10, 5, 2)),
        actions=action_class(torch.ones(9, 5, 1)),
        log_probs=torch.ones(9, 5),
        log_rewards=torch.arange(5, dtype=torch.get_default_dtype()),
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
        log_rewards=torch.ones((5,), dtype=torch.get_default_dtype()),
        is_terminating=torch.zeros(5, dtype=torch.bool),
    )

    return trans


@pytest.fixture
def state_pairs(simple_env):
    # Create a batch of state pairs
    state_class = simple_env.make_states_class()

    pairs = StatesContainer(
        simple_env,
        states=state_class(torch.randn(10, 2)),
        is_terminating=torch.tensor(
            [False, False, False, False, False, True, True, True, True, True],
            dtype=torch.bool,
        ),
        log_rewards=torch.tensor(
            [torch.inf, torch.inf, torch.inf, torch.inf, torch.inf, 1, 1, 1, 1, 1],
            dtype=torch.get_default_dtype(),
        ),
    )

    return pairs


def test_init_rb(simple_env):
    buffer = ReplayBuffer(simple_env, capacity=100)
    assert buffer.capacity == 100
    assert buffer.env == simple_env
    assert buffer.training_container is None
    assert len(buffer) == 0
    assert not buffer.prioritized_capacity
    assert not buffer.prioritized_sampling

    # Test representation
    assert "ReplayBuffer" in repr(buffer)
    assert "empty" in repr(buffer)


def test_add_trajectories(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(trajectories)

    assert buffer.training_container is not None
    assert isinstance(buffer.training_container, Trajectories)
    assert len(buffer) == 5
    assert "Trajectories" in repr(buffer)


def test_add_transitions(simple_env, transitions):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(transitions)

    assert buffer.training_container is not None
    assert isinstance(buffer.training_container, Transitions)
    assert len(buffer) == 5
    assert "Transitions" in repr(buffer)


def test_add_state_pairs(simple_env, state_pairs):
    buffer = ReplayBuffer(simple_env, capacity=10)
    buffer.add(state_pairs)

    assert buffer.training_container is not None
    assert isinstance(buffer.training_container, StatesContainer)
    assert len(buffer) == 10
    assert "StatesContainer" in repr(buffer)


def test_capacity_limit(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=3)
    buffer.add(trajectories)

    assert len(buffer) == 3
    # Should keep the last 3 trajectories
    assert isinstance(buffer.training_container, Trajectories)
    assert isinstance(buffer.training_container.log_rewards, torch.Tensor)
    assert buffer.training_container.log_rewards is not None
    assert buffer.training_container.log_rewards[-3:].tolist() == [2.0, 3.0, 4.0]


def test_prioritized_capacity(simple_env, trajectories):
    buffer = ReplayBuffer(simple_env, capacity=5, prioritized_capacity=True)
    buffer.add(trajectories)

    assert buffer.prioritized_capacity
    assert not buffer.prioritized_sampling
    assert len(buffer) == 5
    # Should sort by log_rewards in ascending order
    assert isinstance(buffer.training_container, Trajectories)
    assert isinstance(buffer.training_container.log_rewards, torch.Tensor)
    assert torch.allclose(
        buffer.training_container.log_rewards, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
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
        assert isinstance(new_buffer.training_container, Trajectories)
        assert isinstance(buffer.training_container, Trajectories)
        assert isinstance(new_buffer.training_container.log_rewards, torch.Tensor)
        assert isinstance(buffer.training_container.log_rewards, torch.Tensor)
        assert torch.allclose(
            new_buffer.training_container.log_rewards,
            buffer.training_container.log_rewards,
        )


def test_type_error(simple_env):
    buffer = ReplayBuffer(simple_env)

    # Try to add an invalid type
    with pytest.raises(AssertionError, match="Must be a container type"):
        buffer.add("not a container")  # type: ignore


def test_init_ndrb(simple_env):
    buffer = NormBasedDiversePrioritizedReplayBuffer(
        simple_env, capacity=100, cutoff_distance=0.5, p_norm_distance=2.0
    )

    assert buffer.capacity == 100
    assert buffer.cutoff_distance == 0.5
    assert buffer.p_norm_distance == 2.0
    assert buffer.prioritized_capacity
    assert not buffer.prioritized_sampling


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
            states=state_class(torch.randn(2, 1, 2)),
            actions=action_class(torch.ones(1, 1, 1)),
            log_probs=torch.zeros(1, 1),
            log_rewards=torch.tensor([i + 10], dtype=torch.get_default_dtype()),
            terminating_idx=torch.tensor([1], dtype=torch.long),
        )
        similar_trajs.extend(new_traj)

    # Add similar trajectories - they should be filtered by diversity
    buffer.add(similar_trajs)

    # The buffer should still prioritize by reward within the diverse set
    assert len(buffer) == 5
    assert isinstance(buffer.training_container, Trajectories)
    assert isinstance(buffer.training_container.log_rewards, torch.Tensor)

    log_rewards = buffer.training_container.log_rewards
    assert log_rewards is not None
    assert torch.all(log_rewards >= 0)


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
            states=state_class(torch.randn(2, 1, 2)),
            actions=action_class(torch.ones(1, 1, 1)),
            log_probs=torch.zeros(1, 1),
            log_rewards=torch.tensor([i + 10], dtype=torch.get_default_dtype()),
            terminating_idx=torch.tensor([1], dtype=torch.long),
        )
        better_trajs.extend(new_traj)

    # Add better trajectories - they should replace lower reward ones.
    buffer.add(better_trajs)

    # The buffer should contain the highest reward trajectories
    assert len(buffer) == 5
    assert isinstance(buffer.training_container, Trajectories)
    assert isinstance(buffer.training_container.log_rewards, torch.Tensor)
    assert torch.min(buffer.training_container.log_rewards) >= 2.0


def test_prioritized_sampling(simple_env, trajectories):
    # Fill a buffer that has prioritised *sampling*.
    buffer = ReplayBuffer(simple_env, capacity=5, prioritized_sampling=True)
    buffer.add(trajectories)

    # Use a fixed RNG seed for deterministic behaviour in the test.
    torch.manual_seed(0)

    n_samples = 1000
    counts = torch.zeros(5, dtype=torch.long)

    # Repeatedly sample a single trajectory and record which reward was drawn.
    for _ in range(n_samples):
        sampled = buffer.sample(1)
        # `log_rewards` is guaranteed to be defined for trajectories stored in the
        # replay buffer when `prioritized_sampling` is enabled.
        assert sampled.log_rewards is not None
        reward_value = int(sampled.log_rewards.item())  # Rewards are 0‒4.
        counts[reward_value] += 1

    # Higher-reward trajectories should have been sampled more often.
    assert torch.all(counts[:-1] <= counts[1:])


# ---------------------------------------------------------------------------
# Replay buffer hardening tests
# ---------------------------------------------------------------------------


def _make_transitions(env, n, log_rewards, log_probs=None):
    """Helper to build n Transitions with controlled log_rewards."""
    state_class = env.make_states_class()
    action_class = env.make_actions_class()
    return Transitions(
        env,
        states=state_class(torch.randn(n, 2)),
        next_states=state_class(torch.randn(n, 2)),
        actions=action_class(torch.ones(n, 1)),
        log_probs=log_probs,
        log_rewards=log_rewards,
        is_terminating=torch.zeros(n, dtype=torch.bool),
    )


def test_prioritized_capacity_evicts_lowest_rewards(simple_env):
    """When buffer overflows with prioritized_capacity, lowest rewards are evicted."""
    buffer = ReplayBuffer(simple_env, capacity=3, prioritized_capacity=True)

    batch1 = _make_transitions(simple_env, 3, torch.tensor([10.0, 20.0, 30.0]))
    buffer.add(batch1)
    assert len(buffer) == 3

    batch2 = _make_transitions(simple_env, 2, torch.tensor([5.0, 25.0]))
    buffer.add(batch2)

    assert len(buffer) == 3
    # Combined [5,10,20,25,30] sorted ascending, keep [-3:] = [20,25,30]
    assert isinstance(buffer.training_container, Transitions)
    rewards = buffer.training_container.log_rewards
    assert rewards is not None
    assert torch.allclose(rewards, torch.tensor([20.0, 25.0, 30.0]))


def test_non_prioritized_capacity_keeps_newest(simple_env):
    """Without prioritized_capacity, buffer keeps newest items (FIFO eviction)."""
    buffer = ReplayBuffer(simple_env, capacity=3, prioritized_capacity=False)

    batch1 = _make_transitions(simple_env, 3, torch.tensor([100.0, 200.0, 300.0]))
    buffer.add(batch1)

    batch2 = _make_transitions(simple_env, 2, torch.tensor([1.0, 2.0]))
    buffer.add(batch2)

    assert len(buffer) == 3
    # [100,200,300] extended with [1,2] = [100,200,300,1,2], keep [-3:] = [300,1,2]
    assert isinstance(buffer.training_container, Transitions)
    rewards = buffer.training_container.log_rewards
    assert rewards is not None
    assert torch.allclose(rewards, torch.tensor([300.0, 1.0, 2.0]))


def test_add_clears_stale_fields_on_trajectories(simple_env):
    """After add(), log_probs and estimator_outputs are cleared on Trajectories."""
    buffer = ReplayBuffer(simple_env, capacity=10)
    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()

    traj = Trajectories(
        simple_env,
        states=state_class(torch.randn(4, 2, 2)),
        actions=action_class(torch.ones(3, 2, 1)),
        log_probs=torch.ones(3, 2),
        log_rewards=torch.tensor([1.0, 2.0]),
        estimator_outputs=torch.randn(3, 2, 4),
        terminating_idx=torch.tensor([2, 3]),
    )
    buffer.add(traj)

    assert isinstance(buffer.training_container, Trajectories)
    assert buffer.training_container.log_probs is None
    assert buffer.training_container.estimator_outputs is None


def test_add_clears_log_probs_on_transitions(simple_env):
    """After add(), log_probs is cleared on Transitions."""
    buffer = ReplayBuffer(simple_env, capacity=10)

    trans = _make_transitions(
        simple_env, 3, torch.tensor([1.0, 2.0, 3.0]), log_probs=torch.zeros(3)
    )
    buffer.add(trans)

    assert isinstance(buffer.training_container, Transitions)
    assert buffer.training_container.log_probs is None


def test_type_consistency_rejects_mismatched_containers(simple_env):
    """Adding a different container type to an occupied buffer raises AssertionError."""
    buffer = ReplayBuffer(simple_env, capacity=10)

    trans = _make_transitions(simple_env, 3, torch.tensor([1.0, 2.0, 3.0]))
    buffer.add(trans)

    state_class = simple_env.make_states_class()
    action_class = simple_env.make_actions_class()
    traj = Trajectories(
        simple_env,
        states=state_class(torch.randn(3, 2, 2)),
        actions=action_class(torch.ones(2, 2, 1)),
        log_rewards=torch.tensor([1.0, 2.0]),
        terminating_idx=torch.tensor([1, 2]),
    )

    with pytest.raises(AssertionError):
        buffer.add(traj)


def test_multiple_add_cycles_maintain_capacity(simple_env):
    """Repeated add cycles never exceed capacity and maintain data integrity."""
    capacity = 5
    buffer = ReplayBuffer(simple_env, capacity=capacity)

    for i in range(20):
        batch = _make_transitions(
            simple_env,
            3,
            torch.tensor([i * 3.0, i * 3.0 + 1, i * 3.0 + 2]),
        )
        buffer.add(batch)
        assert len(buffer) <= capacity

    assert len(buffer) == capacity
    assert isinstance(buffer.training_container, Transitions)
    rewards = buffer.training_container.log_rewards
    assert rewards is not None
    assert len(rewards) == capacity


def test_prioritized_sample_with_replacement(simple_env):
    """Prioritized sampling uses replacement when n_samples > buffer size."""
    buffer = ReplayBuffer(simple_env, capacity=10, prioritized_sampling=True)
    trans = _make_transitions(simple_env, 3, torch.tensor([1.0, 2.0, 3.0]))
    buffer.add(trans)

    sampled = buffer.sample(10)
    assert len(sampled) == 10
    assert isinstance(sampled, Transitions)
    assert sampled.log_rewards is not None
    # All sampled rewards must come from the original set.
    for r in sampled.log_rewards.tolist():
        assert r in [1.0, 2.0, 3.0]


def test_device_property_raises_on_empty_buffer(simple_env):
    """Accessing device on empty buffer raises AssertionError."""
    buffer = ReplayBuffer(simple_env, capacity=10)
    with pytest.raises(AssertionError, match="Buffer is empty"):
        _ = buffer.device


def _make_states_container(
    env, state_tensors, is_terminating, log_rewards, conditions=None
):
    """Helper to build a StatesContainer with controlled data."""
    states = env.states_from_tensor(state_tensors)
    if conditions is not None:
        states.conditions = conditions
    return StatesContainer(
        env=env,
        states=states,
        is_terminating=is_terminating,
        log_rewards=log_rewards,
    )


def test_diverse_buffer_rejects_duplicates():
    """Verify diversity filtering: duplicates rejected, novel states accepted."""
    env = HyperGrid(ndim=2, height=8)
    capacity = 10
    cutoff = 1.0  # L2 distance threshold

    buffer = NormBasedDiversePrioritizedReplayBuffer(
        env, capacity=capacity, cutoff_distance=cutoff, p_norm_distance=2.0
    )

    # Pre-fill buffer: 5 copies of state [1,1] + 5 distinct states spread apart.
    duplicate_state = torch.tensor([1, 1])
    distinct_states = torch.tensor([[3, 3], [5, 5], [7, 7], [3, 7], [7, 3]])
    all_states = torch.cat([duplicate_state.unsqueeze(0).expand(5, -1), distinct_states])
    is_term = torch.ones(10, dtype=torch.bool)
    rewards = torch.ones(10) * 10.0  # High rewards so they won't be filtered by reward.

    initial_data = _make_states_container(env, all_states, is_term, rewards)
    buffer.add(initial_data)
    assert len(buffer) == capacity

    # Try to add more copies of [1,1] — should be rejected (within cutoff of existing).
    dup_states = duplicate_state.unsqueeze(0).expand(3, -1).clone()
    dup_data = _make_states_container(
        env,
        dup_states,
        torch.ones(3, dtype=torch.bool),
        torch.ones(3) * 20.0,  # Even higher reward.
    )
    buffer.add(dup_data)

    # Buffer should still be at capacity (duplicates were filtered out).
    assert len(buffer) == capacity
    # Count how many [1,1] states are in the buffer — should not have grown.
    term_states = buffer.training_container.terminating_states.tensor
    n_duplicates = (term_states == duplicate_state).all(dim=-1).sum().item()
    assert n_duplicates <= 5, f"Expected <=5 copies of [1,1], got {n_duplicates}"

    # Add a truly novel state far from everything — should be accepted.
    novel_state = torch.tensor([[0, 0]])  # Far from all existing states.
    novel_data = _make_states_container(
        env,
        novel_state,
        torch.ones(1, dtype=torch.bool),
        torch.ones(1) * 20.0,
    )
    buffer.add(novel_data)

    # Buffer at capacity, but novel state should have displaced the lowest-reward entry.
    term_states = buffer.training_container.terminating_states.tensor
    has_novel = (term_states == torch.tensor([0, 0])).all(dim=-1).any().item()
    assert has_novel, "Novel state [0,0] should have been accepted into the buffer"


def test_diverse_buffer_conditional_same_state_different_conditions():
    """Same terminal state with different conditions should be treated as distinct."""
    env = ConditionalHyperGrid(ndim=2, height=8)
    capacity = 10
    cutoff = 0.5  # Small enough that condition diff of 1.0 exceeds it.

    buffer = NormBasedDiversePrioritizedReplayBuffer(
        env, capacity=capacity, cutoff_distance=cutoff, p_norm_distance=2.0
    )

    # Pre-fill: 5x state [3,3] with condition=0.0, 5x distinct states with condition=0.0.
    state_a = torch.tensor([3, 3])
    distinct_states = torch.tensor([[1, 1], [5, 5], [7, 7], [1, 7], [7, 1]])
    all_states = torch.cat([state_a.unsqueeze(0).expand(5, -1), distinct_states])
    cond_0 = torch.full((10, 1), 0.0)
    is_term = torch.ones(10, dtype=torch.bool)
    rewards = torch.ones(10) * 10.0

    initial_data = _make_states_container(env, all_states, is_term, rewards, cond_0)
    buffer.add(initial_data)
    assert len(buffer) == capacity

    # Add state [3,3] with condition=1.0 — distance in (state,cond) space is 1.0 > cutoff.
    novel_cond_data = _make_states_container(
        env,
        state_a.unsqueeze(0),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1) * 20.0,
        conditions=torch.tensor([[1.0]]),
    )
    buffer.add(novel_cond_data)

    # Should be accepted — the condition difference exceeds cutoff.
    term_states = buffer.training_container.terminating_states
    state_matches = (term_states.tensor == state_a).all(dim=-1)
    matched_conditions = term_states.conditions[state_matches]
    has_0 = (matched_conditions).abs().lt(0.01).any().item()
    has_1 = (matched_conditions - 1.0).abs().lt(0.01).any().item()
    assert has_0 and has_1, (
        f"Expected both conditions 0.0 and 1.0 for state [3,3], "
        f"got conditions: {matched_conditions.squeeze().tolist()}"
    )

    # Add state [3,3] with condition=0.0 again — should be rejected as duplicate.
    dup_cond_data = _make_states_container(
        env,
        state_a.unsqueeze(0),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1) * 20.0,
        conditions=torch.tensor([[0.0]]),
    )
    len_before = len(buffer)
    buffer.add(dup_cond_data)
    assert len(buffer) == len_before


def test_diversity_repr_shape():
    """Unit test: _diversity_repr includes conditions in the representation."""
    env = HyperGrid(ndim=2, height=4)
    cond_env = ConditionalHyperGrid(ndim=2, height=4)

    # Without conditions: shape is (n_terminating, state_dim).
    states_no_cond = _make_states_container(
        env,
        torch.tensor([[1, 1], [2, 2]]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2),
    )
    repr_no_cond = NormBasedDiversePrioritizedReplayBuffer._diversity_repr(
        states_no_cond
    )
    assert repr_no_cond.shape == (2, 2)

    # With conditions: shape is (n_terminating, state_dim + condition_dim).
    states_with_cond = _make_states_container(
        cond_env,
        torch.tensor([[1, 1], [2, 2]]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2),
        conditions=torch.tensor([[0.5], [0.9]]),
    )
    repr_with_cond = NormBasedDiversePrioritizedReplayBuffer._diversity_repr(
        states_with_cond
    )
    assert repr_with_cond.shape == (2, 3)  # 2 state dims + 1 condition dim
