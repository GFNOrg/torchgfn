import pytest
import torch

from gfn.containers import StatesContainer, Trajectories, Transitions
from gfn.gym.discrete_ebm import DiscreteEBM


def transitions_containers(env):
    """Creates two Transitions containers with valid DiscreteEBM(ndim=2) data.

    For ndim=2: states in {-1,0,1}^2, s0=[-1,-1], sf=[2,2].
    Actions: 0=s[0]→0, 1=s[1]→0, 2=s[0]→1, 3=s[1]→1, 4=exit.
    Forward actions require s[i]==-1 at the target position.
    """
    # Set 1: one intermediate transition, one terminating.
    states1 = env.states_from_tensor(torch.tensor([[-1, -1], [0, 1]]))
    actions1 = env.actions_from_tensor(torch.tensor([[0], [4]]))
    next_states1 = env.states_from_tensor(torch.tensor([[0, -1], [2, 2]]))
    is_terminating1 = torch.tensor([False, True])
    log_probs1 = torch.tensor([-0.5, -0.3])
    log_rewards1 = torch.tensor([0.0, -1.0])

    transitions1 = Transitions(
        env=env,
        states=states1,
        actions=actions1,
        is_terminating=is_terminating1,
        next_states=next_states1,
        log_probs=log_probs1,
        log_rewards=log_rewards1,
    )

    # Set 2: one intermediate transition, one terminating.
    states2 = env.states_from_tensor(torch.tensor([[-1, -1], [1, 0]]))
    actions2 = env.actions_from_tensor(torch.tensor([[2], [4]]))
    next_states2 = env.states_from_tensor(torch.tensor([[1, -1], [2, 2]]))
    is_terminating2 = torch.tensor([False, True])
    log_probs2 = torch.tensor([-0.2, -0.7])
    log_rewards2 = torch.tensor([0.0, -1.0])

    transitions2 = Transitions(
        env=env,
        states=states2,
        actions=actions2,
        is_terminating=is_terminating2,
        next_states=next_states2,
        log_probs=log_probs2,
        log_rewards=log_rewards2,
    )

    return transitions1, transitions2


def state_containers(env):
    """Creates two StatesContainer instances with valid DiscreteEBM(ndim=2) data."""
    # Set 1: 2 intermediary + 2 terminating states.
    states1 = env.states_from_tensor(torch.tensor([[-1, -1], [0, -1], [0, 1], [1, 0]]))
    is_terminating1 = torch.tensor([False, False, True, True])
    log_rewards1 = torch.tensor([0.0, 0.0, -1.0, -2.0])

    container1 = StatesContainer(
        env=env,
        states=states1,
        is_terminating=is_terminating1,
        log_rewards=log_rewards1,
    )

    # Set 2: 2 intermediary + 2 terminating states.
    states2 = env.states_from_tensor(torch.tensor([[-1, -1], [1, -1], [1, 1], [0, 0]]))
    is_terminating2 = torch.tensor([False, False, True, True])
    log_rewards2 = torch.tensor([0.0, 0.0, -1.5, -0.5])

    container2 = StatesContainer(
        env=env,
        states=states2,
        is_terminating=is_terminating2,
        log_rewards=log_rewards2,
    )

    return container1, container2


def trajectories_containers(env):
    """Creates two Trajectories containers with valid DiscreteEBM(ndim=2) data.

    Each trajectory: 2 fill actions + 1 exit = 3 steps, terminating at index 2.
    States shape: (max_length+1, n_trajectories, state_shape) = (4, 2, 2).
    Actions shape: (max_length, n_trajectories, action_shape) = (3, 2, 1).
    """
    # Batch 1: two trajectories of length 3.
    # Traj A: [-1,-1] →(a=0)→ [0,-1] →(a=3)→ [0,1] →(a=4)→ sf
    # Traj B: [-1,-1] →(a=2)→ [1,-1] →(a=1)→ [1,0] →(a=4)→ sf
    states1 = env.states_from_tensor(
        torch.tensor(
            [
                [[-1, -1], [-1, -1]],
                [[0, -1], [1, -1]],
                [[0, 1], [1, 0]],
                [[2, 2], [2, 2]],  # sf padding
            ],
        )
    )
    actions1 = env.actions_from_tensor(
        torch.tensor(
            [
                [[0], [2]],
                [[3], [1]],
                [[4], [4]],
            ],
        )
    )
    terminating_idx1 = torch.tensor([2, 2])
    log_rewards1 = torch.tensor([-2.0, -2.0])

    trajectories1 = Trajectories(
        env=env,
        states=states1,
        actions=actions1,
        terminating_idx=terminating_idx1,
        log_rewards=log_rewards1,
    )

    # Batch 2: two trajectories of length 3.
    # Traj C: [-1,-1] →(a=1)→ [-1,0] →(a=2)→ [1,0] →(a=4)→ sf
    # Traj D: [-1,-1] →(a=3)→ [-1,1] →(a=0)→ [0,1] →(a=4)→ sf
    states2 = env.states_from_tensor(
        torch.tensor(
            [
                [[-1, -1], [-1, -1]],
                [[-1, 0], [-1, 1]],
                [[1, 0], [0, 1]],
                [[2, 2], [2, 2]],
            ],
        )
    )
    actions2 = env.actions_from_tensor(
        torch.tensor(
            [
                [[1], [3]],
                [[2], [0]],
                [[4], [4]],
            ],
        )
    )
    terminating_idx2 = torch.tensor([2, 2])
    log_rewards2 = torch.tensor([-3.0, -1.0])

    trajectories2 = Trajectories(
        env=env,
        states=states2,
        actions=actions2,
        terminating_idx=terminating_idx2,
        log_rewards=log_rewards2,
    )

    return trajectories1, trajectories2


@pytest.mark.parametrize(
    "container_type", ["transitions", "states_container", "trajectories"]
)
def test_extend(container_type: str):
    """Test that extending a container appends elements correctly."""
    env = DiscreteEBM(ndim=2)
    if container_type == "transitions":
        container1, container2 = transitions_containers(env)
    elif container_type == "states_container":
        container1, container2 = state_containers(env)
    else:
        container1, container2 = trajectories_containers(env)

    initial_len = len(container1)
    len2 = len(container2)

    container1.extend(container2)  # type: ignore[arg-type]

    assert len(container1) == initial_len + len2

    if isinstance(container1, Transitions):
        for i in range(len2):
            c1 = container1[i + initial_len]
            c2 = container2[i]
            assert torch.equal(c1.states.tensor, c2.states.tensor)
            assert torch.equal(c1.actions.tensor, c2.actions.tensor)
            assert torch.equal(c1.is_terminating, c2.is_terminating)
            assert torch.equal(c1.next_states.tensor, c2.next_states.tensor)
            assert c1.log_probs is not None and c2.log_probs is not None
            assert torch.equal(c1.log_probs, c2.log_probs)
            assert isinstance(c1.log_rewards, torch.Tensor)
            assert isinstance(c2.log_rewards, torch.Tensor)
            assert torch.equal(c1.log_rewards, c2.log_rewards)

    elif isinstance(container1, StatesContainer):
        for i in range(len2):
            c1 = container1[i + initial_len]
            c2 = container2[i]
            assert torch.equal(
                c1.intermediary_states.tensor, c2.intermediary_states.tensor
            )
            assert torch.equal(
                c1.terminating_states.tensor, c2.terminating_states.tensor
            )
            assert isinstance(c1.log_rewards, torch.Tensor)
            assert isinstance(c2.log_rewards, torch.Tensor)
            assert torch.equal(c1.log_rewards, c2.log_rewards)

    elif isinstance(container1, Trajectories):
        for i in range(len2):
            c1 = container1[i + initial_len]
            c2 = container2[i]
            assert torch.equal(c1.states.tensor, c2.states.tensor)
            assert torch.equal(c1.actions.tensor, c2.actions.tensor)
            assert torch.equal(c1.terminating_idx, c2.terminating_idx)
            assert isinstance(c1.log_rewards, torch.Tensor)
            assert isinstance(c2.log_rewards, torch.Tensor)
            assert torch.equal(c1.log_rewards, c2.log_rewards)
