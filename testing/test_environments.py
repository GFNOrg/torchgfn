import pytest
import torch

from gfn.envs import DiscreteEBMEnv, HyperGrid
from gfn.envs.env import NonValidActionsError


# Utilities.
def format_actions(a, env):
    """Returns a Actions instance from a [batch_size, 1] tensor of actions."""
    return env.Actions(a)


def format_tensor(l):
    """Returns a long tensor with a singleton batch dimension from list l."""
    return torch.tensor(l, dtype=torch.long).unsqueeze(-1)


def format_random_tensor(env, n, h):
    """Returns a long tensor w/ a singleton batch dimension & random actions."""
    return torch.randint(0, env.n_actions - 1, (n, h), dtype=torch.long).unsqueeze(-1)


# Tests.
@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_hypergrid_preprocessors(
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
def test_hypergrid_fwd_step_with_preprocessors(
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
        actions = format_actions(format_tensor(actions_list), env)
        states = env.step(states, actions)

    # Step 4 fails due an invalid input action.
    actions = format_actions(format_tensor(failing_actions_list), env)
    with pytest.raises(NonValidActionsError):
        states = env.step(states, actions)

    expected_rewards = torch.tensor([0.6, 0.1, 0.6])
    assert (torch.round(env.reward(states), decimals=7) == expected_rewards).all()


@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_hypergrid_bwd_step_with_preprocessors(
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
        actions = format_actions(format_tensor(passing_actions_list), env)
        states = env.backward_step(states, actions)

    # Fails due to an invalid input action.
    states = env.reset(batch_shape=(NDIM, ENV_HEIGHT), random=True, seed=SEED)
    failing_actions = format_actions(format_tensor(failing_actions_list), env)
    with pytest.raises(NonValidActionsError):
        states = env.backward_step(states, failing_actions)


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_states_getitem(ndim: int, env_name: str):
    ND_BATCH_SHAPE = (2, 3)

    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=8)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(ndim=ndim)
    else:
        raise ValueError(f"Unknown env_name {env_name}")

    states = env.reset(batch_shape=ND_BATCH_SHAPE, random=True)

    # Boolean selector to index batch elements.
    selections = torch.randint(0, 2, ND_BATCH_SHAPE, dtype=torch.bool)
    n_selections = int(torch.sum(selections))
    selected_states = states[selections]

    assert selected_states.states_tensor.shape == (n_selections, ndim)

    # Boolean selector off of only the first batch dimension.
    selections = torch.randint(0, 2, (ND_BATCH_SHAPE[0],), dtype=torch.bool)
    n_selections = int(torch.sum(selections))
    selected_states = states[selections]

    assert selected_states.states_tensor.shape == (
        n_selections,
        ND_BATCH_SHAPE[1],
        ndim,
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
