import torch
from einops import rearrange

from gfn.containers import States

from .hypergrid import HyperGrid


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


def get_true_dist_pmf(env: HyperGrid) -> torch.Tensor:
    "Returns a one-dimensional tensor representing the true distribution."
    flat_grid = get_flat_grid(env)
    flat_grid_indices = env.get_states_indices(flat_grid)
    true_dist = env.reward(flat_grid)
    true_dist = torch.tensor(
        [true_dist[flat_grid_indices[i]] for i in range(len(flat_grid_indices))]
    )
    true_dist /= true_dist.sum()
    return true_dist
