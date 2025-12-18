import itertools
from typing import TYPE_CHECKING

import torch
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from gfn.gym.diffusion_sampling import BaseTarget


def sliced_log_reward(
    x: torch.Tensor, target: "BaseTarget", dims: tuple
) -> torch.Tensor:
    _x = torch.zeros((x.shape[0], target.dim))
    _x[:, dims] = x
    return target.log_reward(_x.to(target.device)).detach().cpu()


def viz_2d_slice(
    ax: Axes,
    target: "BaseTarget",
    dims: tuple,
    samples: torch.Tensor | None,
    plot_border: tuple[float, float, float, float],
    alpha=0.5,
    n_contour_levels=50,
    grid_width_n_points=200,
    log_reward_clamp_min=-10000.0,
    use_log_reward=False,
    max_n_samples: int | None = None,
) -> None:
    x_points_dim1 = torch.linspace(plot_border[0], plot_border[1], grid_width_n_points)
    x_points_dim2 = torch.linspace(plot_border[2], plot_border[3], grid_width_n_points)
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_r_x = sliced_log_reward(x=x_points, target=target, dims=dims)
    log_r_x = torch.clamp_min(log_r_x, log_reward_clamp_min)
    log_r_x = log_r_x.reshape((grid_width_n_points, grid_width_n_points))
    if not use_log_reward:
        log_r_x = torch.exp(log_r_x)

    x_points_dim1 = (
        x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )
    x_points_dim2 = (
        x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    )
    ax.contour(x_points_dim1, x_points_dim2, log_r_x, levels=n_contour_levels)

    if samples is not None:
        if max_n_samples is not None:
            samples = samples[:max_n_samples]
        samples = samples[:, dims].detach().cpu()
        samples[:, 0] = torch.clamp(samples[:, 0], plot_border[0], plot_border[1])
        samples[:, 1] = torch.clamp(samples[:, 1], plot_border[2], plot_border[3])
        ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, c="r", marker="x")
