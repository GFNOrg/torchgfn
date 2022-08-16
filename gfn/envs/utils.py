from distutils.command.build import build

import torch
from einops import rearrange

from gfn.envs import HyperGrid


def build_grid(env: HyperGrid):
    H = env.height
    ndim = env.ndim
    grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
    grid = torch.zeros(grid_shape)
    for i in range(ndim):
        grid_i = torch.linspace(start=0, end=H - 1, steps=H)
        for _ in range(i):
            grid_i = grid_i.unsqueeze(1)
        grid[..., i] = grid_i
    # return grid.view((H)**ndim,-1) # ((H)*ndim, ndim)
    return grid


def get_flat_grid(env: HyperGrid):
    grid = build_grid(env)
    grid.transpose_(0, 1)
    return rearrange(grid, "... ndim -> (...) ndim")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = HyperGrid(height=4, ndim=3)
    grid = get_flat_grid(env)
    print("Shape of the grid: ", grid.shape)
    print("All rewards: ", env.reward(grid))

    env = HyperGrid(height=8, ndim=2)
    grid = build_grid(env)
    flat_grid = get_flat_grid(env)
    print("Shape of the grid: ", grid.shape)
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
