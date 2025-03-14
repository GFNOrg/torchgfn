import pytest
import torch

from gfn.containers import Transitions, StatePairs, Trajectories
from gfn.gym.discrete_ebm import DiscreteEBM


@pytest.fixture
def simple_env():
    """Create a simple discrete environment for testing."""
    return DiscreteEBM(ndim=2)


@pytest.fixture
def transitions_fixture(simple_env):
    """Create a fixture for Transitions container."""
    env = simple_env
    
    # Create some states
    states1 = env.states_from_tensor(torch.tensor([[0, 0], [1, 1]]))
    actions1 = env.actions_from_tensor(torch.tensor([[1], [0]]))
    is_done1 = torch.tensor([False, True])
    next_states1 = env.states_from_tensor(torch.tensor([[1, 0], [1, 1]]))
    log_probs1 = torch.tensor([-0.5, -0.3])
    log_rewards1 = torch.tensor([0.0, -1.0])
    
    transitions1 = Transitions(
        env=env,
        states=states1,
        actions=actions1,
        is_done=is_done1,
        next_states=next_states1,
        log_probs=log_probs1,
        log_rewards=log_rewards1,
    )
    
    # Create another set of transitions
    states2 = env.states_from_tensor(torch.tensor([[2, 1], [0, 2]]))
    actions2 = env.actions_from_tensor(torch.tensor([[0], [1]]))
    is_done2 = torch.tensor([True, False])
    next_states2 = env.states_from_tensor(torch.tensor([[2, 1], [1, 2]]))
    log_probs2 = torch.tensor([-0.2, -0.7])
    log_rewards2 = torch.tensor([0.0, -1.0])
    
    transitions2 = Transitions(
        env=env,
        states=states2,
        actions=actions2,
        is_done=is_done2,
        next_states=next_states2,
        log_probs=log_probs2,
        log_rewards=log_rewards2,
    )
    
    return transitions1, transitions2


@pytest.fixture
def state_pairs_fixture(simple_env):
    """Create a fixture for StatePairs container."""
    env = simple_env
    
    # Create first set of state pairs
    intermediary_states1 = env.states_from_tensor(torch.tensor([[0, 1], [1, 0]]))
    terminating_states1 = env.states_from_tensor(torch.tensor([[2, 2], [1, 2]]))
    log_rewards1 = torch.tensor([-1.0, -2.0])
    
    state_pairs1 = StatePairs(
        env=env,
        intermediary_states=intermediary_states1,
        terminating_states=terminating_states1,
        log_rewards=log_rewards1,
    )
    
    # Create second set of state pairs
    intermediary_states2 = env.states_from_tensor(torch.tensor([[1, 1], [0, 0]]))
    terminating_states2 = env.states_from_tensor(torch.tensor([[2, 1], [2, 0]]))
    log_rewards2 = torch.tensor([-1.5, -0.5])
    
    state_pairs2 = StatePairs(
        env=env,
        intermediary_states=intermediary_states2,
        terminating_states=terminating_states2,
        log_rewards=log_rewards2,
    )
    
    return state_pairs1, state_pairs2


@pytest.fixture
def trajectories_fixture(simple_env):
    """Create a fixture for Trajectories container."""
    env = simple_env
    
    # Create first set of trajectories
    states1 = env.states_from_tensor(torch.tensor([
        [[0, 0], [0, 0]],  # Initial states
        [[1, 0], [0, 1]],  # Step 1
        [[2, 0], [0, 2]],  # Step 2
        [[2, 0], [0, 2]],  # Padding for trajectory 1
    ]))
    
    actions1 = env.actions_from_tensor(torch.tensor([
        [[0], [1]],  # Step 0
        [[0], [1]],  # Step 1
        [[2], [2]],  # Exit action for both trajectories
    ]))
    
    when_is_done1 = torch.tensor([2, 2])
    log_rewards1 = torch.tensor([-2.0, -2.0])
    
    trajectories1 = Trajectories(
        env=env,
        states=states1,
        actions=actions1,
        when_is_done=when_is_done1,
        log_rewards=log_rewards1,
    )
    
    # Create second set of trajectories
    states2 = env.states_from_tensor(torch.tensor([
        [[0, 0], [0, 0]],  # Initial states
        [[1, 0], [0, 1]],  # Step 1
        [[1, 1], [1, 1]],  # Step 2
        [[2, 1], [1, 1]],  # Step 3 (only for trajectory 1)
    ]))
    
    actions2 = env.actions_from_tensor(torch.tensor([
        [[0], [1]],  # Step 0
        [[1], [1]],  # Step 1
        [[0], [2]],  # Step 2 (exit for trajectory 2)
        [[2], [2]],  # Step 3 (exit for trajectory 1, padding for trajectory 2)
    ]))
    
    when_is_done2 = torch.tensor([3, 2])
    log_rewards2 = torch.tensor([-3.0, -2.0])
    
    trajectories2 = Trajectories(
        env=env,
        states=states2,
        actions=actions2,
        when_is_done=when_is_done2,
        log_rewards=log_rewards2,
    )
    
    return trajectories1, trajectories2


@pytest.mark.parametrize("container", ["transitions_fixture", "state_pairs_fixture", "trajectories_fixture"])
def test_containers(container, request):
    container1, container2 = request.getfixturevalue(container)
    initial_len = len(container1)

    # Test extending container1 with container2
    container1.extend(container2)
    
    # Check that the length of container1 is now the sum of both containers
    assert len(container1) == initial_len + len(container2)
    
    # Check that the elements from container2 are correctly added to container1
    if isinstance(container1, Transitions):
        for i in range(len(container2)):
            assert torch.equal(container1[i + initial_len].states.tensor, container2[i].states.tensor)
            assert torch.equal(container1[i + initial_len].actions.tensor, container2[i].actions.tensor)
            assert torch.equal(container1[i + initial_len].is_done, container2[i].is_done)
            assert torch.equal(container1[i + initial_len].next_states.tensor, container2[i].next_states.tensor)
            assert torch.equal(container1[i + initial_len].log_probs, container2[i].log_probs)
            assert torch.equal(container1[i + initial_len].log_rewards, container2[i].log_rewards)
    elif isinstance(container1, StatePairs):
        for i in range(len(container2)):
            assert torch.equal(container1[i + initial_len].intermediary_states.tensor, container2[i].intermediary_states.tensor)
            assert torch.equal(container1[i + initial_len].terminating_states.tensor, container2[i].terminating_states.tensor)
            assert torch.equal(container1[i + initial_len].log_rewards, container2[i].log_rewards)
    elif isinstance(container1, Trajectories):
        for i in range(len(container2)):
            assert torch.equal(container1[i + initial_len].states.tensor, container2[i].states.tensor)
            assert torch.equal(container1[i + initial_len].actions.tensor, container2[i].actions.tensor)
            assert torch.equal(container1[i + initial_len].when_is_done, container2[i].when_is_done)
            assert torch.equal(container1[i + initial_len].log_rewards, container2[i].log_rewards)
