import pytest
import torch

from gfn.envs import DiscreteEBMEnv, HyperGrid
from gfn.envs.env import NonValidActionsError


@pytest.mark.parametrize("preprocessor", ["Identity", "OneHot", "KHot"])
def test_hypergrid_and_preprocessors(
    preprocessor: str,
):

    def format_actions(a, env):
        """Returns a Actions instance from a [batch_size, 1] tensor of actions."""
        return env.Actions(a)

    def format_tensor(l):
        """Returns a long tensor with a singleton batch dimension from list l."""
        return torch.tensor(l, dtype=torch.long).unsqueeze(-1)

    def format_random_tensor(env, n, h):
        """Returns a long tensor w/ a singleton batch dimension & random actions."""
        return torch.randint(
            0, env.n_actions - 1, (n, h), dtype=torch.long).unsqueeze(-1)

    NDIM = 2
    ENV_HEIGHT = 3
    BATCH_SIZE = 3

    env = HyperGrid(ndim=NDIM, height=ENV_HEIGHT, preprocessor_name=preprocessor)
    print(env)

    print("\nInstantiating a linear batch of initial states")
    states = env.reset(batch_shape=BATCH_SIZE)
    print("States:", states)

    print("\nTrying the step function starting from 3 instances of s_0")
    actions = format_actions(format_tensor([0, 1, 2]), env)
    states = env.step(states, actions)

    print("After one step:", states)
    actions = format_actions(format_tensor([2, 0, 1]), env)
    states = env.step(states, actions)

    print("After two steps:", states)
    actions = format_actions(format_tensor([2, 0, 1]), env)
    states = env.step(states, actions)

    print("After three steps:", states)
    try:
        actions = format_actions(format_tensor([2, 0, 1]), env)
        states = env.step(states, actions)
    except NonValidActionsError:
        print("NonValidActionsError raised as expected because of invalid actions")

    print(states)
    print("Final rewards:", env.reward(states))
    print("\nTrying the backward step function starting from a batch of random states")
    print("\nInstantiating a two-dimensional batch of random states")

    states = env.reset(batch_shape=(NDIM, ENV_HEIGHT), random=True)

    print("States:", states)
    backward_step_ok = False
    while not backward_step_ok:
        actions = format_actions(format_random_tensor(env, NDIM, ENV_HEIGHT), env)
        print("Actions: ", actions)
        try:
            states = env.backward_step(states, actions)
            print("States:", states)
        except NonValidActionsError:
            print("NonValidActionsError raised as expected because of invalid actions")
        backward_step_ok = True

    print("\nTrying the preprocessors")
    random_states = env.reset(batch_shape=10, random=True)
    preprocessed_grid = env.preprocessor.preprocess(random_states)
    print("Preprocessed Grid: ", preprocessed_grid)

    random_states = env.reset(batch_shape=(4, 2), random=True)
    preprocessed_grid = env.preprocessor.preprocess(random_states)
    print("Preprocessed Grid: ", preprocessed_grid)


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_states_getitem(ndim: int, env_name: str):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=ndim, height=8)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(ndim=ndim)
    else:
        raise ValueError(f"Unknown env_name {env_name}")

    states = env.reset(batch_shape=(2, 3), random=True)
    print("States:", states)
    print("\nTesting subscripting with boolean tensors")

    selections = torch.randint(0, 2, (2, 3), dtype=torch.bool)
    print("Selections:", selections)
    print("States[selections]:", states[selections])
    selections = torch.randint(0, 2, (2,), dtype=torch.bool)
    print("Selections:", selections)
    print("States[selections]:", states[selections])


def test_get_grid(plot=False):
    env = HyperGrid(height=8, ndim=2)
    grid = env.build_grid()
    print("Shape of the grid: ", grid.batch_shape, grid.state_shape)
    rewards = env.reward(grid)

    Z = rewards.sum()

    if Z.log().item() != env.log_partition:
        raise ValueError("Something is wrong")

    if plot:
        import matplotlib.pyplot as plt

        plt.imshow(rewards)
        plt.colorbar()
        plt.show()

    print(env.get_states_indices(grid))

    print(env.reward(grid))


# if __name__ == "__main__":
#     test_hypergrid_and_preprocessors(preprocessor="Identity")