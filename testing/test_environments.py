from typing import Literal

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gfn.actions import GraphActions, GraphActionType
from gfn.env import NonValidActionsError
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.graph_building import GraphBuilding
from gfn.gym.perfect_tree import PerfectBinaryTree
from gfn.gym.set_addition import SetAddition
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.states import GraphStates


# Utilities.
def format_tensor(list_, discrete=True):
    """
    If discrete, returns a long tensor with a singleton batch dimension from list
    ``list_``. Otherwise, casts list to a float tensor without unsqueezing
    """
    if discrete:
        return torch.tensor(list_, dtype=torch.long).unsqueeze(-1)
    else:
        return torch.tensor(list_, dtype=torch.float)


def format_random_tensor(env, n, h):
    """Returns a long tensor w/ a singleton batch dimension & random actions."""
    return torch.randint(0, env.n_actions - 1, (n, h), dtype=torch.long).unsqueeze(-1)


# Tests.
@pytest.mark.parametrize("preprocessor_name", ["Identity", "OneHot", "KHot"])
def test_HyperGrid_preprocessors(
    preprocessor_name: Literal["Identity", "OneHot", "KHot"],
):
    NDIM = 2
    ENV_HEIGHT = 3
    BATCH_SHAPE = 100  # Sufficiently large so all permutations always found.
    ND_BATCH_SHAPE = (4, 2)
    SEED = 1234

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT)

    if preprocessor_name == "Identity":
        preprocessor = IdentityPreprocessor(output_dim=NDIM)
        expected_shape = BATCH_SHAPE, NDIM
    elif preprocessor_name == "OneHot":
        preprocessor = OneHotPreprocessor(
            n_states=env.n_states, get_states_indices=env.get_states_indices
        )
        expected_shape = BATCH_SHAPE, ENV_HEIGHT**NDIM
    elif preprocessor_name == "KHot":
        preprocessor = KHotPreprocessor(ndim=NDIM, height=ENV_HEIGHT)
        expected_shape = BATCH_SHAPE, ENV_HEIGHT * NDIM

    # Test with a 1-d batch size.
    random_states = env.reset(batch_shape=BATCH_SHAPE, random=True, seed=SEED)
    preprocessed_grid = preprocessor.preprocess(random_states)
    assert tuple(preprocessed_grid.shape) == expected_shape

    # Test with a n-d batch size.
    random_states = env.reset(batch_shape=ND_BATCH_SHAPE, random=True, seed=SEED)
    preprocessed_grid = preprocessor.preprocess(random_states)
    assert tuple(preprocessed_grid.shape) == ND_BATCH_SHAPE + expected_shape[1:]


def test_HyperGrid_fwd_step():
    NDIM = 2
    ENV_HEIGHT = BATCH_SIZE = 3

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT)
    states = env.reset(batch_shape=BATCH_SIZE)  # Instantiate a batch of initial states
    assert (states.batch_shape[0], states.state_shape[0]) == (BATCH_SIZE, NDIM)

    # Trying the step function starting from 3 instances of s_0
    passing_actions_lists = [
        [0, 1, 2],
        [2, 0, 1],
        [2, 0, 1],
    ]

    failing_actions_list = [2, 0, 1]

    for actions_list in passing_actions_lists:
        actions = env.actions_from_tensor(format_tensor(actions_list))
        states = env._step(states, actions)

    # Step 4 fails due an invalid input action.
    actions = env.actions_from_tensor(format_tensor(failing_actions_list))
    with pytest.raises(NonValidActionsError):
        states = env._step(states, actions)

    expected_rewards = torch.tensor([0.6, 0.1, 0.6])
    assert (torch.round(env.reward(states), decimals=7) == expected_rewards).all()


def test_HyperGrid_bwd_step():
    NDIM = 2
    ENV_HEIGHT = 3
    SEED = 1234

    # Testing the backward method from a batch of random (seeded) state.
    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT)
    states = env.reset(batch_shape=(NDIM, ENV_HEIGHT), random=True, seed=SEED)

    passing_actions_lists = [
        [[0, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [1, 0, 1]],
        [[0, 0, 1], [0, 2, 1]],
        [[2, 1, 1], [2, 1, 2]],
        [[3, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
    ]

    failing_actions_list = [[1, 0, 0], [0, 0, 1]]

    # All passing actions complete sucessfully.
    for passing_actions_list in passing_actions_lists:
        actions = env.actions_from_tensor(format_tensor(passing_actions_list))
        states = env._backward_step(states, actions)

    # Fails due to an invalid input action.
    states = env.reset(batch_shape=(NDIM, ENV_HEIGHT), random=True, seed=SEED)
    failing_actions = env.actions_from_tensor(format_tensor(failing_actions_list))
    with pytest.raises(NonValidActionsError):
        states = env._backward_step(states, failing_actions)


def test_DiscreteEBM_fwd_step():
    NDIM = 2
    BATCH_SIZE = 4

    env = DiscreteEBM(ndim=NDIM)
    states = env.reset(
        batch_shape=BATCH_SIZE, seed=1234
    )  # Instantiate a batch of initial states
    assert (states.batch_shape[0], states.state_shape[0]) == (BATCH_SIZE, NDIM)

    # Trying the step function starting from 3 instances of s_0
    passing_actions_lists = [
        [0, 1, 0, 1],
        [3, 2, 1, 2],
    ]  # Only next possible move is [4, 4, 4, 4],

    for actions_list in passing_actions_lists:
        actions = env.actions_from_tensor(format_tensor(actions_list))
        states = env._step(states, actions)

    # Step 4 fails due an invalid input action (15 is not possible).
    actions = env.actions_from_tensor(format_tensor([4, 15, 4, 4]))
    with pytest.raises(RuntimeError):
        states = env._step(states, actions)

    # Step 5 fails due an invalid input action (1 is possible but not in this state).
    actions = env.actions_from_tensor(format_tensor([1, 4, 4, 4]))
    with pytest.raises(NonValidActionsError):
        states = env._step(states, actions)

    expected_rewards = torch.tensor([1, 1, 54.5982, 1])
    assert (torch.round(env.reward(states), decimals=4) == expected_rewards).all()


def test_DiscreteEBM_bwd_step():
    NDIM = 2
    BATCH_SIZE = 3
    SEED = 1234

    # Testing the backward method from a batch of random (seeded) state.
    env = DiscreteEBM(ndim=NDIM)
    states = env.reset(batch_shape=BATCH_SIZE, random=True, seed=SEED)

    passing_actions_lists = [
        [2, 3, 1],
        [2, 2, 2],
    ]
    # All passing actions complete sucessfully.
    for passing_actions_list in passing_actions_lists:
        actions = env.actions_from_tensor(format_tensor(passing_actions_list))
        states = env._backward_step(states, actions)

    # Fails due to an invalid input action.
    failing_actions_list = [0, 0, 0]
    states = env.reset(batch_shape=BATCH_SIZE, random=True, seed=SEED)
    failing_actions = env.actions_from_tensor(format_tensor(failing_actions_list))
    with pytest.raises(NonValidActionsError):
        states = env._backward_step(states, failing_actions)


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0])
def test_box_fwd_step(delta: float):
    env = Box(delta=delta)
    BATCH_SIZE = 3

    states = env.reset(batch_shape=BATCH_SIZE)  # Instantiate a batch of initial states
    assert (states.batch_shape[0], states.state_shape[0]) == (BATCH_SIZE, 2)

    failing_actions_lists_at_s0 = [
        [[delta, delta], [0.01, 0.01], [0.01, 0.01]],
        [[0.01, 0.01], [0.01, 0.01], [1.0, delta]],
        [[0.01, 0.01], [delta, 1.0], [0.01, 0.01]],
    ]

    for failing_actions_list in failing_actions_lists_at_s0:
        actions = env.actions_from_tensor(
            format_tensor(failing_actions_list, discrete=False)
        )
        with pytest.raises(NonValidActionsError):
            states = env._step(states, actions)

    # Trying the step function starting from 3 instances of s_0
    A, B = None, None
    for i in range(3):
        if i == 0:
            # The following element contains 3 actions within the quarter disk that could be taken from s0
            actions_list = [
                [delta / 2 * np.cos(np.pi / 4), delta / 2 * np.sin(np.pi / 4)],
                [delta / 3 * np.cos(np.pi / 3), delta / 3 * np.sin(np.pi / 3)],
                [delta / np.sqrt(2), delta / np.sqrt(2)],
            ]
        else:
            assert A is not None and B is not None
            # The following contains 3 actions within the corresponding quarter circles
            actions_tensor = torch.tensor([0.2, 0.3, 0.4]) * (B - A) + A
            actions_tensor *= np.pi / 2
            actions_tensor = (
                torch.stack(
                    [torch.cos(actions_tensor), torch.sin(actions_tensor)], dim=1
                )
                * env.delta
            )
            actions_tensor[B - A < 0] = torch.tensor([-float("inf"), -float("inf")])
            actions_list = actions_tensor.tolist()

        actions = env.actions_from_tensor(format_tensor(actions_list, discrete=False))
        states = env._step(states, actions)
        states_tensor = states.tensor

        # The following evaluate the maximum angles of the possible actions
        A = torch.where(
            states_tensor[:, 0] <= 1 - env.delta,
            0.0,
            2.0 / torch.pi * torch.arccos((1 - states_tensor[:, 0]) / env.delta),
        )
        B = torch.where(
            states_tensor[:, 1] <= 1 - env.delta,
            1.0,
            2.0 / torch.pi * torch.arcsin((1 - states_tensor[:, 1]) / env.delta),
        )


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_states_getitem(ndim: int, env_name: str):
    ND_BATCH_SHAPE = (2, 3)

    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=8)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=ndim)
    elif env_name == "Box":
        env = Box(delta=1.0 / ndim)
    else:
        raise ValueError(f"Unknown env_name {env_name}")

    states = env.reset(batch_shape=ND_BATCH_SHAPE, random=True)

    # Boolean selector to index batch elements.
    selections = torch.randint(0, 2, ND_BATCH_SHAPE, dtype=torch.bool)
    n_selections = int(torch.sum(selections))
    selected_states = states[selections]

    assert selected_states.tensor.shape == (
        n_selections,
        ndim if env_name != "Box" else 2,
    )

    # Boolean selector off of only the first batch dimension.
    selections = torch.randint(0, 2, (ND_BATCH_SHAPE[0],), dtype=torch.bool)
    n_selections = int(torch.sum(selections))
    selected_states = states[selections]

    assert selected_states.tensor.shape == (
        n_selections,
        ND_BATCH_SHAPE[1],
        ndim if env_name != "Box" else 2,
    )


def test_get_grid():
    HEIGHT = 8
    NDIM = 2

    env = HyperGrid(
        height=HEIGHT, ndim=NDIM, calculate_all_states=True, calculate_partition=True
    )
    all_states = env.all_states

    assert all_states.batch_shape == (HEIGHT**2,)
    assert all_states.state_shape == (NDIM,)

    rewards = env.reward(all_states)
    assert tuple(rewards.shape) == all_states.batch_shape

    # All rewards are positive.
    assert torch.sum(rewards > 0) == HEIGHT**2

    # log(Z) should equal the environment log_partition.
    Z = rewards.sum()
    assert env.log_partition is not None
    assert np.isclose(Z.log().item(), env.log_partition)

    # State indices of the grid are ordered from 0:HEIGHT**2.
    assert (env.get_states_indices(all_states).ravel() == torch.arange(HEIGHT**2)).all()


def test_graph_env():
    BATCH_SIZE = 3
    NUM_NODES = 5

    env = GraphBuilding(
        state_evaluator=lambda s: torch.zeros(s.batch_shape),
        num_node_classes=10,
        num_edge_classes=10,
    )
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.batch_shape == (BATCH_SIZE,)
    action_cls = env.make_actions_class()

    # We can't add an edge without nodes.
    with pytest.raises(IndexError):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_EDGE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._step(states, actions)

    # Add nodes.
    for _ in range(NUM_NODES):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._step(states, actions)

    assert states.tensor.x.shape == (BATCH_SIZE * NUM_NODES, 1)

    # We can't add a node with the same features.
    with pytest.raises(NonValidActionsError):
        first_node_mask = torch.arange(len(states.tensor.x)) // BATCH_SIZE == 0
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: states.tensor.x[first_node_mask],
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._step(states, actions)

    # Add edges.
    for i in range(NUM_NODES**2 - NUM_NODES):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_EDGE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.tensor([i] * BATCH_SIZE),
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._step(states, actions)

    actions = action_cls.from_tensor_dict(
        TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: torch.full(
                    (BATCH_SIZE,), GraphActionType.EXIT
                ),
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_INDEX_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
            },
            batch_size=BATCH_SIZE,
        )
    )

    sf_states = env._step(states, actions)
    assert torch.all(sf_states.is_sink_state)
    assert isinstance(sf_states, GraphStates)
    env.reward(sf_states)

    # Remove edges.
    for i in reversed(range(NUM_NODES**2 - NUM_NODES)):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_EDGE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.tensor([i] * BATCH_SIZE),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._backward_step(states, actions)

    # We can't remove edges that don't exist.
    with pytest.raises(NonValidActionsError):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_EDGE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._backward_step(states, actions)

    # Remove nodes.
    for i in reversed(range(1, NUM_NODES + 1)):
        edge_idx = torch.arange(BATCH_SIZE) * i
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: states.tensor.x[edge_idx],
                    GraphActions.EDGE_CLASS_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._backward_step(states, actions)

    assert states.tensor.x.shape == (0, 1)

    # Add one random node again
    actions = action_cls.from_tensor_dict(
        TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: torch.full(
                    (BATCH_SIZE,), GraphActionType.ADD_NODE
                ),
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_INDEX_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
            },
            batch_size=BATCH_SIZE,
        )
    )
    states = env._step(states, actions)

    # We can't remove nodes that don't exist.
    with pytest.raises(NonValidActionsError):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.ones(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_CLASS_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.EDGE_INDEX_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env._backward_step(states, actions)

    # Remove the node.
    actions = action_cls.from_tensor_dict(
        TensorDict(
            {
                GraphActions.ACTION_TYPE_KEY: torch.full(
                    (BATCH_SIZE,), GraphActionType.ADD_NODE
                ),
                GraphActions.NODE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_CLASS_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.EDGE_INDEX_KEY: torch.zeros(
                    (BATCH_SIZE,), dtype=torch.long
                ),
            },
            batch_size=BATCH_SIZE,
        )
    )
    states = env._backward_step(states, actions)
    assert states.tensor.x.shape == (0, 1)


def test_set_addition_fwd_step():
    N_ITEMS = 4
    MAX_ITEMS = 3
    BATCH_SIZE = 2

    env = SetAddition(
        n_items=N_ITEMS, max_items=MAX_ITEMS, reward_fn=lambda s: s.sum(-1)
    )
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.tensor.shape == (BATCH_SIZE, N_ITEMS)

    # Add item 0 and 1
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)

    # Add item 2 and 3
    actions = env.actions_from_tensor(format_tensor([2, 3]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)

    # Try adding existing items (invalid)
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    with pytest.raises(NonValidActionsError):
        env._step(states, actions)

    # Add item 3 and 0
    actions = env.actions_from_tensor(format_tensor([3, 0]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)  # Now has 3 items

    # Try adding another item (invalid, max_items reached)
    actions = env.actions_from_tensor(format_tensor([1, 2]))
    with pytest.raises(NonValidActionsError):
        env._step(states, actions)

    # Exit action (valid)
    actions = env.actions_from_tensor(format_tensor([N_ITEMS, N_ITEMS]))
    final_states = env._step(states, actions)
    assert torch.all(final_states.is_sink_state)

    # Check rewards
    rewards = env.reward(states)
    expected_rewards = torch.tensor([3.0, 3.0])
    assert torch.allclose(rewards, expected_rewards)


def test_set_addition_bwd_step():
    N_ITEMS = 5
    MAX_ITEMS = 4
    BATCH_SIZE = 2

    env = SetAddition(
        n_items=N_ITEMS, max_items=MAX_ITEMS, reward_fn=lambda s: s.sum(-1)
    )

    # Start from a state with 3 items
    initial_tensor = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float)
    states = env.states_from_tensor(initial_tensor)

    # Remove item 1 and 2
    actions = env.actions_from_tensor(format_tensor([1, 2]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)

    # Try removing non-existent item (invalid)
    actions = env.actions_from_tensor(format_tensor([2, 0]))
    with pytest.raises(NonValidActionsError):
        env._backward_step(states, actions)

    # Remove item 0 and 4
    actions = env.actions_from_tensor(format_tensor([0, 4]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)

    # Remove item 3 and 1 (last items)
    actions = env.actions_from_tensor(format_tensor([3, 1]))
    states = env._backward_step(states, actions)
    expected_states = torch.zeros((BATCH_SIZE, N_ITEMS), dtype=torch.float)
    assert torch.equal(states.tensor, expected_states)
    assert torch.all(states.is_initial_state)


def test_perfect_binary_tree_fwd_step():
    DEPTH = 3
    BATCH_SIZE = 2
    N_ACTIONS = 3  # 0=left, 1=right, 2=exit

    env = PerfectBinaryTree(depth=DEPTH, reward_fn=lambda s: s.float() + 1)
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.tensor.shape == (BATCH_SIZE, 1)
    assert torch.all(states.tensor == 0)

    # Go left, Go right
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[1], [2]], dtype=torch.long)
    assert torch.equal(states.tensor, expected_states)

    # Go right, Go left
    actions = env.actions_from_tensor(format_tensor([1, 0]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[4], [5]], dtype=torch.long)
    assert torch.equal(states.tensor, expected_states)

    # Go left, Go left
    actions = env.actions_from_tensor(format_tensor([0, 0]))
    states = env._step(states, actions)
    expected_states = torch.tensor([[9], [11]], dtype=torch.long)  # Leaf nodes
    assert torch.equal(states.tensor, expected_states)
    assert torch.all(torch.isin(states.tensor, env.terminating_states.tensor))

    # Try moving from leaf node (invalid)
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    with pytest.raises(NonValidActionsError):
        env._step(states, actions)

    # Exit action (valid)
    actions = env.actions_from_tensor(format_tensor([N_ACTIONS - 1, N_ACTIONS - 1]))
    final_states = env._step(states, actions)
    assert torch.all(final_states.is_sink_state)

    # Check rewards
    rewards = env.reward(states)
    expected_rewards = torch.tensor([[10.0], [12.0]])
    assert torch.allclose(rewards, expected_rewards)


def test_perfect_binary_tree_bwd_step():
    DEPTH = 3

    env = PerfectBinaryTree(depth=DEPTH, reward_fn=lambda s: s.float() + 1)

    # Start from leaf nodes 8 and 12
    initial_tensor = torch.tensor([[8], [12]], dtype=torch.long)
    states = env.states_from_tensor(initial_tensor)

    # Try backward exit action (invalid)
    actions = env.actions_from_tensor(format_tensor([2, 2]))
    with pytest.raises(RuntimeError):
        env._backward_step(states, actions)

    # Go up (from right child, from left child)
    # Node 8 is right child of 3 (action 1). Node 12 is left child of 5 (action 0)
    actions = env.actions_from_tensor(format_tensor([1, 0]))
    # Go up (Node 8 is right child of 3 -> bwd action 1; Node 12 is right child of 5 -> bwd action 1)
    actions = env.actions_from_tensor(format_tensor([1, 1]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor([[3], [5]], dtype=torch.long)
    assert torch.equal(states.tensor, expected_states)

    # Go up (from left child, from right child)
    # Node 3 is left child of 1 (action 0). Node 5 is right child of 2 (action 1)
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    # Go up (Node 3 is left child of 1 -> bwd action 0; Node 5 is left child of 2 -> bwd action 0)
    actions = env.actions_from_tensor(format_tensor([0, 0]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor([[1], [2]], dtype=torch.long)
    assert torch.equal(states.tensor, expected_states)

    # Go up to root (from left child, from right child)
    # Node 1 is left child of 0 (action 0). Node 2 is right child of 0 (action 1)
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor([[0], [0]], dtype=torch.long)
    assert torch.equal(states.tensor, expected_states)
    assert torch.all(states.is_initial_state)
