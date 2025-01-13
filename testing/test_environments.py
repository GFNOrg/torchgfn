import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gfn.actions import GraphActionType
from gfn.env import NonValidActionsError
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.graph_building import GraphBuilding


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
@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_HyperGrid_preprocessors(
    preprocessor: str,
):
    NDIM = 2
    ENV_HEIGHT = 3
    BATCH_SHAPE = 100  # Sufficiently large so all permutations always found.
    ND_BATCH_SHAPE = (4, 2)
    SEED = 1234

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, preprocessor_name=preprocessor)

    # Test with a 1-d batch size.
    random_states = env.reset(batch_shape=BATCH_SHAPE, random=True, seed=SEED)
    preprocessed_grid = env.preprocessor.preprocess(random_states)

    if preprocessor == "Identity":
        assert tuple(preprocessed_grid.shape) == (BATCH_SHAPE, NDIM)
    elif preprocessor == "OneHot":
        assert tuple(preprocessed_grid.shape) == (BATCH_SHAPE, ENV_HEIGHT**NDIM)
    elif preprocessor == "KHot":
        assert tuple(preprocessed_grid.shape) == (BATCH_SHAPE, ENV_HEIGHT * NDIM)

    # Test with a n-d batch size.
    random_states = env.reset(batch_shape=ND_BATCH_SHAPE, random=True, seed=SEED)
    preprocessed_grid = env.preprocessor.preprocess(random_states)

    if preprocessor == "Identity":
        assert tuple(preprocessed_grid.shape) == ND_BATCH_SHAPE + tuple([NDIM])
    elif preprocessor == "OneHot":
        assert tuple(preprocessed_grid.shape) == ND_BATCH_SHAPE + tuple(
            [ENV_HEIGHT**NDIM]
        )
    elif preprocessor == "KHot":
        assert tuple(preprocessed_grid.shape) == ND_BATCH_SHAPE + tuple(
            [ENV_HEIGHT * NDIM]
        )


@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_HyperGrid_fwd_step_with_preprocessors(
    preprocessor: str,
):
    NDIM = 2
    ENV_HEIGHT = BATCH_SIZE = 3

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, preprocessor_name=preprocessor)
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


@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_HyperGrid_bwd_step_with_preprocessors(
    preprocessor: str,
):
    NDIM = 2
    ENV_HEIGHT = 3
    SEED = 1234

    # Testing the backward method from a batch of random (seeded) state.
    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, preprocessor_name=preprocessor)
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

    env = HyperGrid(height=HEIGHT, ndim=NDIM)
    grid = env.build_grid()

    assert grid.batch_shape == (HEIGHT, HEIGHT)
    assert grid.state_shape == (NDIM,)

    rewards = env.reward(grid)
    assert tuple(rewards.shape) == grid.batch_shape

    # All rewards are positive.
    assert torch.sum(rewards > 0) == HEIGHT**2

    # log(Z) should equal the environment log_partition.
    Z = rewards.sum()
    assert Z.log().item() == env.log_partition

    # State indices of the grid are ordered from 0:HEIGHT**2.
    assert (env.get_states_indices(grid).ravel() == torch.arange(HEIGHT**2)).all()


def test_graph_env():
    FEATURE_DIM = 8
    BATCH_SIZE = 3
    NUM_NODES = 5

    env = GraphBuilding(
        feature_dim=FEATURE_DIM, state_evaluator=lambda s: torch.zeros(s.batch_shape)
    )
    states = env.reset(batch_shape=BATCH_SIZE)
    assert states.batch_shape == (BATCH_SIZE,)
    action_cls = env.make_actions_class()

    with pytest.raises(NonValidActionsError):
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_EDGE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                    "edge_index": torch.randint(
                        0, 10, (BATCH_SIZE, 2), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.step(states, actions)

    for _ in range(NUM_NODES):
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_NODE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.step(states, actions)
        states = env.States(states)

    assert states.tensor["node_feature"].shape == (BATCH_SIZE * NUM_NODES, FEATURE_DIM)

    with pytest.raises(NonValidActionsError):
        first_node_mask = (
            torch.arange(len(states.tensor["node_feature"])) // BATCH_SIZE == 0
        )
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_NODE),
                    "features": states.tensor["node_feature"][first_node_mask],
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.step(states, actions)

    with pytest.raises(NonValidActionsError):
        edge_index = torch.randint(0, 3, (BATCH_SIZE,), dtype=torch.long)
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_EDGE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                    "edge_index": torch.stack([edge_index, edge_index], dim=1),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.step(states, actions)

    for i in range(NUM_NODES - 1):
        node_is = torch.full((BATCH_SIZE,), i)
        node_js = torch.full((BATCH_SIZE,), i + 1)
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_EDGE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                    "edge_index": torch.stack([node_is, node_js], dim=1),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.step(states, actions)
        states = env.States(states)

    actions = action_cls(
        TensorDict(
            {
                "action_type": torch.full((BATCH_SIZE,), GraphActionType.EXIT),
            },
            batch_size=BATCH_SIZE,
        )
    )
    sf_states = env.step(states, actions)
    sf_states = env.States(sf_states)
    assert torch.all(sf_states.is_sink_state)
    env.reward(states)

    # with pytest.raises(NonValidActionsError):
    #     node_idx = torch.arange(0, BATCH_SIZE)
    #     actions = action_cls(
    #         GraphActionType.ADD_NODE,
    #         states.data.x[node_idxs],
    #     )
    #     states = env.backward_step(states, actions)

    num_edges_per_batch = len(states.tensor["edge_feature"]) // BATCH_SIZE
    for i in reversed(range(num_edges_per_batch)):
        edge_idx = torch.arange(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_EDGE),
                    "features": states.tensor["edge_feature"][edge_idx],
                    "edge_index": states.tensor["edge_index"][edge_idx] - states.tensor["batch_ptr"][:-1][:, None],
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.backward_step(states, actions)
        states = env.States(states)

    with pytest.raises(NonValidActionsError):
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_EDGE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                    "edge_index": torch.randint(
                        0, 10, (BATCH_SIZE, 2), dtype=torch.long
                    ),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.backward_step(states, actions)

    for i in reversed(range(1, NUM_NODES + 1)):
        edge_idx = torch.arange(BATCH_SIZE) * i
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_NODE),
                    "features": states.tensor["node_feature"][edge_idx],
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.backward_step(states, actions)
        states = env.States(states)

    assert states.tensor["node_feature"].shape == (0, FEATURE_DIM)

    with pytest.raises(NonValidActionsError):
        actions = action_cls(
            TensorDict(
                {
                    "action_type": torch.full((BATCH_SIZE,), GraphActionType.ADD_NODE),
                    "features": torch.rand((BATCH_SIZE, FEATURE_DIM)),
                },
                batch_size=BATCH_SIZE,
            )
        )
        states = env.backward_step(states, actions)
