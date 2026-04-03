from typing import Literal, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gfn.actions import GraphActions, GraphActionType
from gfn.env import DiscreteEnv, Env, NonValidActionsError
from gfn.estimators import (
    DiscretePolicyEstimator,
    PinnedBrownianMotionBackward,
    PinnedBrownianMotionForward,
)
from gfn.gflownet import TBGFlowNet
from gfn.gym import Box, ChipDesign, ConditionalHyperGrid, DiscreteEBM, HyperGrid
from gfn.gym.chip_design import ChipDesignStates
from gfn.gym.diffusion_sampling import DiffusionSampling
from gfn.gym.graph_building import GraphBuilding
from gfn.gym.perfect_tree import PerfectBinaryTree
from gfn.gym.set_addition import SetAddition
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import GraphStates
from gfn.utils.modules import DiffusionFixedBackwardModule, DiffusionPISGradNetForward


# Utilities.
def format_tensor(list_, discrete=True):
    """
    If discrete, returns a long tensor with a singleton batch dimension from list
    ``list_``. Otherwise, casts list to a float tensor without unsqueezing
    """
    if discrete:
        return torch.tensor(list_, dtype=torch.long).unsqueeze(-1)
    else:
        return torch.tensor(list_, dtype=torch.get_default_dtype())


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

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, debug=True, validate_modes=False)

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

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, debug=True, validate_modes=False)
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
    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, debug=True, validate_modes=False)
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


def test_ConditionalHyperGrid():
    NDIM = 2
    ENV_HEIGHT = 3
    BATCH_SIZE = 5

    env = ConditionalHyperGrid(
        ndim=NDIM, height=ENV_HEIGHT, store_all_states=True, validate_modes=False
    )

    # Condition Sampling
    conditions = env.sample_conditions(BATCH_SIZE)
    assert conditions.shape == (BATCH_SIZE, 1)
    assert (conditions >= 0).all() and (conditions <= 1).all()

    # Reset with automatic condition sampling
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.conditions is not None
    assert states.conditions.shape == (BATCH_SIZE, 1)

    # Reset with provided conditions
    fixed_cond = torch.rand((BATCH_SIZE, 1))
    states = env.reset(batch_shape=BATCH_SIZE, conditions=fixed_cond)
    assert states.conditions is not None
    assert torch.equal(states.conditions, fixed_cond)


def test_DiscreteEBM_fwd_step():
    NDIM = 2
    BATCH_SIZE = 4

    env = DiscreteEBM(ndim=NDIM, debug=True)
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
    env = DiscreteEBM(ndim=NDIM, debug=True)
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
    """Test Box environment forward step with Cartesian semantics.

    Cartesian semantics:
    - From s0: actions should be in [0, 1] per dimension (full space coverage)
    - From non-s0: actions should be in [delta, 1-state] per dimension
    """
    env = Box(delta=delta, debug=True)
    BATCH_SIZE = 3

    states = env.reset(batch_shape=BATCH_SIZE)  # Instantiate a batch of initial states
    assert (states.batch_shape[0], states.state_shape[0]) == (BATCH_SIZE, 2)

    # Test invalid actions from s0 (Cartesian: any component > 1 is invalid)
    failing_actions_lists_at_s0 = [
        # One action has component > 1
        [[0.01, 0.01], [0.01, 0.01], [1.1, 0.01]],
        [[0.01, 0.01], [0.01, 1.1], [0.01, 0.01]],
        # Negative actions are invalid
        [[-0.01, 0.01], [0.01, 0.01], [0.01, 0.01]],
    ]

    for failing_actions_list in failing_actions_lists_at_s0:
        actions = env.actions_from_tensor(
            format_tensor(failing_actions_list, discrete=False)
        )
        with pytest.raises(NonValidActionsError):
            env._step(states, actions)

    # Test valid actions from s0 (Cartesian: all components in [0, 1])
    valid_actions_at_s0 = [
        [0.3, 0.4],
        [0.7, 0.2],
        [0.99, 0.99],  # Near maximum valid action per dimension
    ]
    actions = env.actions_from_tensor(format_tensor(valid_actions_at_s0, discrete=False))
    next_states = env._step(states, actions)

    # Verify next states are in valid range
    assert (next_states.tensor >= 0).all()
    assert (next_states.tensor <= 1 + 1e-6).all()

    # Test from non-s0 states: actions must be >= delta and not exceed boundary
    # Only test if delta < 0.5 (otherwise non-s0 states can't take valid non-exit actions)
    if delta < 0.5:
        # Choose states that have enough room for action of size delta
        # state + delta <= 1, so state <= 1 - delta
        max_state = 1.0 - delta - 0.01
        non_s0_states = env.States(
            torch.tensor(
                [
                    [max_state * 0.3, max_state * 0.3],
                    [max_state * 0.5, max_state * 0.5],
                    [max_state * 0.4, max_state * 0.4],
                ]
            )
        )

        # Valid actions: minimum valid is delta
        valid_non_s0_actions = [
            [delta, delta],
            [delta, delta],
            [delta, delta],
        ]
        actions = env.actions_from_tensor(
            format_tensor(valid_non_s0_actions, discrete=False)
        )
        final_states = env._step(non_s0_states, actions)

        # Verify final states don't exceed boundary
        assert (final_states.tensor <= 1 + 1e-6).all()
        assert (final_states.tensor >= 0).all()


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "ConditionalHyperGrid"]
)
def test_states_getitem(ndim: int, env_name: str):
    ND_BATCH_SHAPE = (2, 3)

    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=8, debug=True)
    elif env_name == "ConditionalHyperGrid":
        env = ConditionalHyperGrid(ndim=ndim, height=8, debug=True)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=ndim, debug=True)
    elif env_name == "Box":
        env = Box(delta=1.0 / ndim, debug=True)
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
        height=HEIGHT,
        ndim=NDIM,
        store_all_states=True,
        calculate_partition=True,
        debug=True,
    )
    all_states = env.all_states
    assert all_states is not None

    assert all_states.batch_shape == (HEIGHT**2,)
    assert all_states.state_shape == (NDIM,)

    rewards = env.reward(all_states)
    assert tuple(rewards.shape) == all_states.batch_shape

    # All rewards are positive.
    assert torch.sum(rewards > 0) == HEIGHT**2

    # log(Z) should equal the environment log_partition.
    Z = rewards.sum()
    true_logZ = env.log_partition()
    assert true_logZ is not None
    assert np.isclose(Z.log().item(), true_logZ)

    # State indices of the grid are ordered from 0:HEIGHT**2.
    assert (env.get_states_indices(all_states).ravel() == torch.arange(HEIGHT**2)).all()


def test_graph_env():
    BATCH_SIZE = 3
    NUM_NODES = 5

    env = GraphBuilding(
        state_evaluator=lambda s: torch.zeros(s.batch_shape),
        num_node_classes=10,
        num_edge_classes=10,
        debug=True,
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
                    GraphActions.NODE_INDEX_KEY: torch.zeros(
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
        states = env._step(states, actions)

    # Add nodes.
    for i in range(NUM_NODES):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.NODE_INDEX_KEY: torch.tensor([i] * BATCH_SIZE),
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
                    GraphActions.NODE_INDEX_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
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
                GraphActions.NODE_INDEX_KEY: torch.zeros(
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
                    GraphActions.NODE_INDEX_KEY: torch.zeros(
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
                    GraphActions.NODE_INDEX_KEY: torch.zeros(
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
    for i in reversed(range(NUM_NODES)):
        actions = action_cls.from_tensor_dict(
            TensorDict(
                {
                    GraphActions.ACTION_TYPE_KEY: torch.full(
                        (BATCH_SIZE,), GraphActionType.ADD_NODE
                    ),
                    GraphActions.NODE_CLASS_KEY: torch.zeros(
                        (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.NODE_INDEX_KEY: torch.tensor([i] * BATCH_SIZE),
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
                GraphActions.NODE_CLASS_KEY: torch.randint(
                    0, 10, (BATCH_SIZE,), dtype=torch.long
                ),
                GraphActions.NODE_INDEX_KEY: torch.zeros(
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
                    GraphActions.NODE_CLASS_KEY: torch.randint(
                        0, 10, (BATCH_SIZE,), dtype=torch.long
                    ),
                    GraphActions.NODE_INDEX_KEY: torch.ones(
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
                GraphActions.NODE_INDEX_KEY: torch.zeros(
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
        n_items=N_ITEMS, max_items=MAX_ITEMS, reward_fn=lambda s: s.sum(-1), debug=True
    )
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.tensor.shape == (BATCH_SIZE, N_ITEMS)

    # Add item 0 and 1
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    states = env._step(states, actions)
    expected_states = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.get_default_dtype()
    )
    assert torch.equal(states.tensor, expected_states)

    # Add item 2 and 3
    actions = env.actions_from_tensor(format_tensor([2, 3]))
    states = env._step(states, actions)
    expected_states = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.get_default_dtype()
    )
    assert torch.equal(states.tensor, expected_states)

    # Try adding existing items (invalid)
    actions = env.actions_from_tensor(format_tensor([0, 1]))
    with pytest.raises(NonValidActionsError):
        env._step(states, actions)

    # Add item 3 and 0
    actions = env.actions_from_tensor(format_tensor([3, 0]))
    states = env._step(states, actions)
    expected_states = torch.tensor(
        [[1, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.get_default_dtype()
    )
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
        n_items=N_ITEMS, max_items=MAX_ITEMS, reward_fn=lambda s: s.sum(-1), debug=True
    )

    # Start from a state with 3 items
    initial_tensor = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.get_default_dtype()
    )
    states = env.states_from_tensor(initial_tensor)

    # Remove item 1 and 2
    actions = env.actions_from_tensor(format_tensor([1, 2]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor(
        [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.get_default_dtype()
    )
    assert torch.equal(states.tensor, expected_states)

    # Try removing non-existent item (invalid)
    actions = env.actions_from_tensor(format_tensor([2, 0]))
    with pytest.raises(NonValidActionsError):
        env._backward_step(states, actions)

    # Remove item 0 and 4
    actions = env.actions_from_tensor(format_tensor([0, 4]))
    states = env._backward_step(states, actions)
    expected_states = torch.tensor(
        [[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=torch.get_default_dtype()
    )
    assert torch.equal(states.tensor, expected_states)

    # Remove item 3 and 1 (last items)
    actions = env.actions_from_tensor(format_tensor([3, 1]))
    states = env._backward_step(states, actions)
    expected_states = torch.zeros((BATCH_SIZE, N_ITEMS), dtype=torch.get_default_dtype())
    assert torch.equal(states.tensor, expected_states)
    assert torch.all(states.is_initial_state)


def test_perfect_binary_tree_fwd_step():
    DEPTH = 3
    BATCH_SIZE = 2
    N_ACTIONS = 3  # 0=left, 1=right, 2=exit

    env = PerfectBinaryTree(
        depth=DEPTH,
        reward_fn=lambda s: s.to(torch.get_default_dtype()) + 1,
        debug=True,
    )
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

    env = PerfectBinaryTree(depth=DEPTH, reward_fn=lambda s: s.float() + 1, debug=True)

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


def test_chip_design():
    BATCH_SIZE = 2

    env = ChipDesign(device="cpu")
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.tensor.shape == (BATCH_SIZE, env.n_macros)
    assert torch.all(states.tensor == -1)

    # Place macros
    for i in range(env.n_macros):
        actions = env.actions_from_tensor(format_tensor([i] * BATCH_SIZE))
        expected_tensor = states.tensor.clone()
        states = env._step(states, actions)
        expected_tensor[..., i] = i
        assert torch.equal(states.tensor, expected_tensor)

    # Exit action (valid)
    actions = env.actions_from_tensor(format_tensor([env.n_actions - 1] * BATCH_SIZE))
    final_states = env._step(states, actions)
    assert torch.all(final_states.is_sink_state)

    # Check rewards
    assert isinstance(final_states, ChipDesignStates)
    rewards = env.log_reward(final_states)
    assert torch.all(rewards == rewards[0])



# -----------------------------------------------------------------------------
# Tests for default sf fill value based on dtype
# -----------------------------------------------------------------------------


class _DummyEnv(Env):
    def step(self, states, actions):  # pragma: no cover - not used in this test
        return states

    def backward_step(self, states, actions):  # pragma: no cover - not used
        return states

    def is_action_valid(
        self, states, actions, backward: bool = False
    ) -> bool:  # noqa: ARG002
        return True


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_env_default_sf_float_dtypes(dtype: torch.dtype):
    state_shape = (2, 3)
    s0 = torch.zeros(state_shape, dtype=dtype)
    dummy_action = torch.zeros((1,), dtype=torch.long)
    exit_action = torch.zeros((1,), dtype=torch.long)
    env = _DummyEnv(
        s0=s0,
        state_shape=state_shape,
        action_shape=(1,),
        dummy_action=dummy_action,
        exit_action=exit_action,
        sf=None,
        debug=True,
    )
    assert env.sf.dtype == dtype
    assert isinstance(env.sf, torch.Tensor)
    assert torch.isinf(env.sf).all()
    assert (env.sf < 0).all()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_env_default_sf_complex_dtypes(dtype: torch.dtype):
    state_shape = (2, 2)
    s0 = torch.zeros(state_shape, dtype=dtype)
    dummy_action = torch.zeros((1,), dtype=torch.long)
    exit_action = torch.zeros((1,), dtype=torch.long)
    env = _DummyEnv(
        s0=s0,
        state_shape=state_shape,
        action_shape=(1,),
        dummy_action=dummy_action,
        exit_action=exit_action,
        sf=None,
        debug=True,
    )
    assert env.sf.dtype == dtype
    assert isinstance(env.sf, torch.Tensor)
    # -inf + 0j
    assert torch.isinf(env.sf).all()
    assert (env.sf.real < 0).all()
    assert torch.equal(env.sf.imag, torch.zeros_like(env.sf.imag))


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
)
def test_env_default_sf_integer_dtypes(dtype: torch.dtype):
    state_shape = (3, 1)
    s0 = torch.zeros(state_shape, dtype=dtype)
    dummy_action = torch.zeros((1,), dtype=torch.long)
    exit_action = torch.zeros((1,), dtype=torch.long)
    env = _DummyEnv(
        s0=s0,
        state_shape=state_shape,
        action_shape=(1,),
        dummy_action=dummy_action,
        exit_action=exit_action,
        sf=None,
        debug=True,
    )
    assert env.sf.dtype == dtype
    assert isinstance(env.sf, torch.Tensor)
    expected_min = torch.iinfo(dtype).min
    assert torch.equal(env.sf, torch.full(state_shape, expected_min, dtype=dtype))


def test_env_default_sf_bool_dtype():
    state_shape = (1, 4)
    s0 = torch.zeros(state_shape, dtype=torch.bool)
    dummy_action = torch.zeros((1,), dtype=torch.long)
    exit_action = torch.zeros((1,), dtype=torch.long)
    env = _DummyEnv(
        s0=s0,
        state_shape=state_shape,
        action_shape=(1,),
        dummy_action=dummy_action,
        exit_action=exit_action,
        sf=None,
        debug=True,
    )
    assert env.sf.dtype == torch.bool
    assert isinstance(env.sf, torch.Tensor)
    assert torch.equal(env.sf, torch.zeros(state_shape, dtype=torch.bool))


def test_diffusion_trajectory_mask_alignment():
    """Test that diffusion trajectory masks align correctly for PB calculation.

    This verifies that the estimator's exit action detection matches the environment's
    terminal state detection, ensuring valid_states and valid_actions have the same
    count in get_trajectory_pbs. A mismatch would cause an AssertionError.

    The key invariant is: for each trajectory step where we compute PB, we need
    exactly one valid state (at t+1) and one valid action (at t). Exit actions
    must be properly marked so they're excluded from the action mask.
    """
    # Use small config for fast testing.
    num_steps = 8
    batch_size = 16
    s_dim = 2

    env = DiffusionSampling(
        target_str="gmm2",
        target_kwargs={"seed": 42},
        num_discretization_steps=num_steps,
        device=torch.device("cpu"),
    )

    pf_module = DiffusionPISGradNetForward(
        s_dim=s_dim,
        harmonics_dim=16,
        t_emb_dim=16,
        s_emb_dim=16,
        hidden_dim=32,
        joint_layers=1,
    )
    pb_module = DiffusionFixedBackwardModule(s_dim=s_dim)

    pf_estimator = PinnedBrownianMotionForward(
        s_dim=s_dim,
        pf_module=pf_module,
        sigma=5.0,
        num_discretization_steps=num_steps,
    )
    pb_estimator = PinnedBrownianMotionBackward(
        s_dim=s_dim,
        pb_module=pb_module,
        sigma=5.0,
        num_discretization_steps=num_steps,
    )

    sampler = Sampler(estimator=pf_estimator)

    # Sample trajectories.
    trajectories = sampler.sample_trajectories(
        env,
        n=batch_size,
        save_logprobs=True,
        save_estimator_outputs=False,
    )

    # Compute masks the same way get_trajectory_pbs does.
    state_mask = (
        ~trajectories.states.is_sink_state & ~trajectories.states.is_initial_state
    )
    state_mask[0, :] = False  # Can't compute PB for first state row.
    action_mask = ~trajectories.actions.is_dummy & ~trajectories.actions.is_exit

    valid_states_count = int(state_mask.sum())
    valid_actions_count = int(action_mask.sum())
    exit_count = int(trajectories.actions.is_exit.sum())

    # Key assertions:
    # 1. Exit actions should be detected (one per trajectory for fixed-length diffusion).
    assert exit_count == batch_size, (
        f"Expected {batch_size} exit actions (one per trajectory), got {exit_count}. "
        "The estimator may not be marking exit actions correctly."
    )

    # 2. Valid states and actions must match for PB calculation.
    assert valid_states_count == valid_actions_count, (
        f"Mask mismatch: {valid_states_count} valid states vs {valid_actions_count} valid actions. "
        f"Exit count: {exit_count}. This would cause get_trajectory_pbs to fail."
    )

    # 3. Verify get_trajectory_pbs runs without error (the actual alignment check).
    from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs

    log_pfs, log_pbs = get_trajectory_pfs_and_pbs(
        pf_estimator,
        pb_estimator,
        trajectories,
        recalculate_all_logprobs=False,
    )
    # Shape is (T, N) = (num_steps, batch_size) - per-step log probs for each trajectory.
    assert log_pfs.shape == (num_steps, batch_size)
    assert log_pbs.shape == (num_steps, batch_size)


# ---------------------------------------------------------------------------
# Box backward_step tests
# ---------------------------------------------------------------------------


def test_box_bwd_step():
    """Test Box environment backward step arithmetic."""
    env = Box(delta=0.1, debug=True)

    # Backward step is subtraction: result = state - action
    # Actions must produce valid states (result >= 0, and result either >= delta or == 0)
    state_tensor = torch.tensor([[0.5, 0.5], [0.7, 0.3], [0.9, 0.9]])
    states = env.States(state_tensor.clone())
    # These actions produce states [0.3, 0.4], [0.4, 0.15], [0.5, 0.5] — all >= delta
    action_tensor = torch.tensor([[0.2, 0.1], [0.3, 0.15], [0.4, 0.4]])
    actions = env.actions_from_tensor(action_tensor)

    result = env._backward_step(states, actions)
    expected = state_tensor - action_tensor
    assert torch.allclose(result.tensor, expected, atol=1e-6)


def test_box_bwd_step_roundtrip():
    """Forward step then backward step returns to original state."""
    env = Box(delta=0.1, debug=True)

    # Start from s0, take a forward step
    s0 = env.reset(batch_shape=(3,))
    action_tensor = torch.tensor([[0.3, 0.4], [0.5, 0.2], [0.1, 0.8]])
    actions = env.actions_from_tensor(action_tensor)

    stepped = env._step(s0, actions)
    # Now backward step with the same action
    restored = env._backward_step(stepped, actions)

    assert torch.allclose(restored.tensor, s0.tensor, atol=1e-6)


def test_box_bwd_is_action_valid():
    """Test backward action validation for BoxCartesian."""
    env = Box(delta=0.1, debug=True)

    state = env.States(torch.tensor([[0.5, 0.5]]))

    # Valid backward action: state - action >= 0
    valid_action = env.actions_from_tensor(torch.tensor([[0.3, 0.2]]))
    assert env.is_action_valid(state, valid_action, backward=True)

    # Invalid backward action: would produce negative coordinates
    invalid_action = env.actions_from_tensor(torch.tensor([[0.6, 0.2]]))
    assert not env.is_action_valid(state, invalid_action, backward=True)

    # Can't go backward from s0
    s0 = env.reset(batch_shape=(1,))
    any_action = env.actions_from_tensor(torch.tensor([[0.1, 0.1]]))
    assert not env.is_action_valid(s0, any_action, backward=True)


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------


def _make_tb_gflownet(env, debug=True):
    """Helper to build a TBGFlowNet for validation tests."""
    from gfn.preprocessors import KHotPreprocessor
    from gfn.utils.modules import MLP

    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    pf_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    pb_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1)
    pf = DiscretePolicyEstimator(
        pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )
    return TBGFlowNet(pf=pf, pb=pb, debug=debug)


def test_validate_hypergrid():
    """validate() returns L1 distance and logZ_diff for HyperGrid."""
    env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
    gflownet = _make_tb_gflownet(env)

    torch.manual_seed(42)
    info, states = env.validate(gflownet, n_validation_samples=200)

    assert "l1_dist" in info
    assert 0 <= info["l1_dist"] <= 2.0  # L1 between prob dists is in [0, 2]
    assert "logZ_diff" in info
    assert info["logZ_diff"] >= 0
    assert states is not None
    assert len(states) > 0


def test_validate_discrete_ebm():
    """validate() works for DiscreteEBM."""
    env = DiscreteEBM(ndim=4)

    from gfn.utils.modules import MLP

    preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
    input_dim = cast(int, preprocessor.output_dim)
    pf_module = MLP(input_dim=input_dim, output_dim=env.n_actions)
    pb_module = MLP(input_dim=input_dim, output_dim=env.n_actions - 1)
    pf = DiscretePolicyEstimator(
        pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )
    gflownet = TBGFlowNet(pf=pf, pb=pb)

    torch.manual_seed(42)
    info, states = env.validate(gflownet, n_validation_samples=200)

    assert "l1_dist" in info
    assert 0 <= info["l1_dist"] <= 2.0
    assert "logZ_diff" in info
    assert states is not None


@pytest.mark.skip(reason="validate() doesn't propagate condition to sampling yet")
def test_validate_conditional_hypergrid():
    """validate() works with a condition tensor for ConditionalHyperGrid."""
    from gfn.estimators import ConditionalDiscretePolicyEstimator
    from gfn.utils.modules import MLP

    env = ConditionalHyperGrid(
        ndim=2, height=4, store_all_states=True, validate_modes=False
    )
    preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
    input_dim = cast(int, preprocessor.output_dim)
    hidden = 16

    pf = ConditionalDiscretePolicyEstimator(
        state_module=MLP(input_dim=input_dim, output_dim=hidden),
        condition_module=MLP(input_dim=env.condition_dim, output_dim=hidden),
        final_module=MLP(input_dim=hidden * 2, output_dim=env.n_actions),
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    )
    pb = ConditionalDiscretePolicyEstimator(
        state_module=MLP(input_dim=input_dim, output_dim=hidden),
        condition_module=MLP(input_dim=env.condition_dim, output_dim=hidden),
        final_module=MLP(input_dim=hidden * 2, output_dim=env.n_actions - 1),
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    )
    gflownet = TBGFlowNet(pf=pf, pb=pb)

    torch.manual_seed(42)
    condition = torch.tensor([0.5])
    info, states = env.validate(
        gflownet,
        n_validation_samples=200,
        validate_condition=condition,
    )

    assert "l1_dist" in info
    assert 0 <= info["l1_dist"] <= 2.0
    assert "logZ_diff" in info
    assert states is not None


def test_validate_visited_states_deprecated():
    """validate() emits DeprecationWarning when visited_terminating_states is passed."""
    env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
    gflownet = _make_tb_gflownet(env)

    torch.manual_seed(42)
    sampled = gflownet.sample_terminating_states(env, 100)

    from gfn.states import DiscreteStates

    with pytest.warns(DeprecationWarning, match="visited_terminating_states"):
        info, returned_states = env.validate(
            gflownet,
            n_validation_samples=50,
            visited_terminating_states=cast(DiscreteStates, sampled),
        )

    assert "l1_dist" in info
    assert returned_states is not None


def test_validate_no_true_dist_raises():
    """validate() raises ValueError when true_dist is not implemented."""
    env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
    gflownet = _make_tb_gflownet(env)

    # Temporarily make true_dist raise NotImplementedError.
    original = env.true_dist
    env.true_dist = lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError)

    with pytest.raises(ValueError, match="does not implement true_dist"):
        env.validate(gflownet, n_validation_samples=10)

    env.true_dist = original


def test_validate_zero_samples_raises():
    """validate() raises ValueError with n_validation_samples=0."""
    env = HyperGrid(ndim=2, height=4, store_all_states=True, validate_modes=False)
    gflownet = _make_tb_gflownet(env)

    with pytest.raises(ValueError, match="must be > 0"):
        env.validate(gflownet, n_validation_samples=0)


# --- JSD correctness tests ---


class TestJSD:
    """Verify DiscreteEnv._jsd is unbiased across distribution shapes."""

    @staticmethod
    def _jsd_numpy(p, q):
        """Reference JSD using scipy-style masking for 0·log(0)=0."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        m = 0.5 * (p + q)
        kl_pm = np.where(p > 0, p * np.log(p / np.where(m > 0, m, 1.0)), 0.0).sum()
        kl_qm = np.where(q > 0, q * np.log(q / np.where(m > 0, m, 1.0)), 0.0).sum()
        return 0.5 * (kl_pm + kl_qm)

    def test_identical_distributions(self):
        """JSD(p, p) == 0 for any p."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert DiscreteEnv._jsd(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_disjoint_distributions(self):
        """JSD is ln(2) when supports are completely disjoint."""
        p = torch.tensor([0.5, 0.5, 0.0, 0.0])
        q = torch.tensor([0.0, 0.0, 0.5, 0.5])
        assert DiscreteEnv._jsd(p, q) == pytest.approx(np.log(2), abs=1e-7)

    def test_uniform_vs_peaked(self):
        """JSD between uniform and peaked matches numpy reference."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.97, 0.01, 0.01, 0.01])
        expected = self._jsd_numpy(p.numpy(), q.numpy())
        assert DiscreteEnv._jsd(p, q) == pytest.approx(expected, abs=1e-7)

    def test_extreme_sparsity(self):
        """JSD is unbiased when most bins are zero (sparse distributions).

        This is the case that eps-clamping gets wrong: with 10000 bins and
        only a few nonzero, clamping adds eps to every bin, inflating total
        mass and biasing the result.
        """
        n_bins = 10000
        p = torch.zeros(n_bins)
        q = torch.zeros(n_bins)
        # p has mass on 3 bins, q has mass on 3 different bins
        p[:3] = torch.tensor([0.5, 0.3, 0.2])
        q[5000:5003] = torch.tensor([0.6, 0.3, 0.1])
        expected = self._jsd_numpy(p.numpy(), q.numpy())
        result = DiscreteEnv._jsd(p, q)
        assert result == pytest.approx(expected, abs=1e-7)
        # Fully disjoint support → must equal ln(2)
        assert result == pytest.approx(np.log(2), abs=1e-7)

    def test_partial_overlap_sparse(self):
        """JSD with partial overlap on a large sparse vector."""
        n_bins = 10000
        p = torch.zeros(n_bins)
        q = torch.zeros(n_bins)
        # Shared bin 0; different bins 1 and 5000
        p[0] = 0.5
        p[1] = 0.5
        q[0] = 0.5
        q[5000] = 0.5
        expected = self._jsd_numpy(p.numpy(), q.numpy())
        result = DiscreteEnv._jsd(p, q)
        assert result == pytest.approx(expected, abs=1e-7)
        # Must be strictly between 0 and ln(2)
        assert 0 < result < np.log(2)

    def test_symmetry(self):
        """JSD(p, q) == JSD(q, p)."""
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.3, 0.6])
        assert DiscreteEnv._jsd(p, q) == pytest.approx(DiscreteEnv._jsd(q, p), abs=1e-10)

    def test_bounded(self):
        """JSD is always in [0, ln(2)]."""
        rng = torch.Generator().manual_seed(42)
        for _ in range(20):
            p = torch.rand(50, generator=rng)
            p = p / p.sum()
            q = torch.rand(50, generator=rng)
            q = q / q.sum()
            jsd = DiscreteEnv._jsd(p, q)
            assert 0 <= jsd <= np.log(2) + 1e-10
