import torch
from einops import rearrange

from gfn.containers import States
from gfn.envs import HyperGrid


def build_grid(env: HyperGrid) -> States:
    H = env.height
    ndim = env.ndim
    grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
    grid = torch.zeros(grid_shape)
    for i in range(ndim):
        grid_i = torch.linspace(start=0, end=H - 1, steps=H)
        for _ in range(i):
            grid_i = grid_i.unsqueeze(1)
        grid[..., i] = grid_i

    rearrange_string = " ".join([f"n{i}" for i in range(1, ndim + 1)])
    rearrange_string += " ndim -> "
    rearrange_string += " ".join([f"n{i}" for i in range(ndim, 0, -1)])
    rearrange_string += " ndim"
    grid = rearrange(grid, rearrange_string)
    return env.States(grid)


def get_flat_grid(env: HyperGrid) -> States:
    grid = build_grid(env)
    flat_grid = rearrange(grid.states, "... ndim -> (...) ndim")
    return env.States(flat_grid)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = HyperGrid(height=4, ndim=3)
    grid = get_flat_grid(env)
    print("Shape of the grid: ", grid.batch_shape, grid.state_shape)
    print(grid)
    print("All rewards: ", env.reward(grid))

    env = HyperGrid(height=8, ndim=2)
    grid = build_grid(env)
    flat_grid = get_flat_grid(env)
    print("Shape of the grid: ", grid.batch_shape, grid.state_shape)
    rewards = env.reward(grid)

    Z = rewards.sum()

    if Z != env.reward(flat_grid).sum():
        print("Something is wrong")

    plt.imshow(rewards)
    plt.colorbar()
    plt.show()

    print(env.get_states_indices(grid))
    print(env.get_states_indices(flat_grid))

    print(env.reward(grid))
    print(env.reward(flat_grid))
