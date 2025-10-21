import os
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wishart

from gfn.actions import Actions
from gfn.env import Env
from gfn.states import States
from gfn.utils.common import set_seed


class DiffusionSampling(Env):
    """Diffusion sampling environment.

    Attributes:
        target: The target distribution.
        device: The device to use.
        check_action_validity: Whether to check the action validity.
    """

    def __init__(
        self,
        target_str: str,
        target_kwargs: dict[str, Any],
        device: torch.device = torch.device("cpu"),
        check_action_validity: bool = False,
    ) -> None:
        """Initialize the DiffusionSampling environment.

        Args:
            target_str: The string identifier for the target.
            target_kwargs: The keyword arguments for the target.
            device: The device to use.
            check_action_validity: Whether to check the action validity.
        """
        DIFFUSION_TARGETS = {
            "simple_gmm": SimpleGaussianMixtureTarget,
            # TODO: add more targets here
        }
        self.target = DIFFUSION_TARGETS[target_str](device=device, **target_kwargs)
        self.dim = self.target.dim

        default_dtype = torch.get_default_dtype()

        s0 = torch.zeros((self.dim + 1,), device=device)  # + 1 for time

        super().__init__(
            s0=s0,
            state_shape=(self.dim + 1,),
            action_shape=(self.dim + 1,),
            # dummy action is never used since all trajectories are terminated at
            # time == 1 (i.e., we don't need to pad shorter trajectories)
            dummy_action=torch.full(
                (self.dim + 1,), float("inf"), device=device, dtype=default_dtype
            ),
            exit_action=torch.full(
                (self.dim + 1,), -float("inf"), device=device, dtype=default_dtype
            ),
            check_action_validity=check_action_validity,
        )

    def step(self, states: States, actions: Actions) -> States:
        """Step function for the SimpleGaussianMixtureModel environment.

        Args:
            states: The current states (not used here).
            actions: The actions, which correspond to the next states.

        Returns:
            The next states.
        """
        return self.States(actions.tensor)

    def backward_step(self, states: States, actions: Actions) -> States:
        """Backward step function for the SimpleGaussianMixtureModel environment.

        Args:
            states: The current states (not used here).
            actions: The actions, which correspond to the previous states.

        Returns:
            The previous states.
        """
        return self.States(actions.tensor)

    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Check if the actions are valid.

        Args:
            states: The current states (not used here).
            actions: The actions to check.

        Returns:
            True if the actions are valid, False otherwise.
        """
        assert len(states.batch_shape) == 1, "States must have a batch_shape of length 1"
        time = states.tensor[0, -1].item()
        # TODO: support randomized discretization
        assert (
            states.tensor[:, -1] == time
        ).all(), "Time must be the same for all states in the batch"

        if not backward and time == 1.0:  # Terminate if time == 1.0 for forward steps
            return bool((actions.tensor == self.sf).all().item())
        elif backward and time == 0.0:  # Return to s0 if time == 0.0 for backward steps
            return bool((actions.tensor == self.s0).all().item())
        else:
            return True

    def log_reward(self, states: States) -> torch.Tensor:
        """Log reward function for the DiffusionSampling environment.

        Args:
            states: The current states.

        Returns:
            The log rewards for the input states.
        """
        # Remove the last index, which encodes the time step.
        return self.target.log_reward(states.tensor[..., :-1])


class BaseTarget(ABC):
    """Base class for all target distributions for diffusion sampling.

    Attributes:
        device: The device on which the target is stored.
        dim: The dimension of the target.
        gt_xs: The ground truth samples.
        gt_xs_log_rewards: The log rewards of the ground truth samples.
    """

    def __init__(
        self,
        device: torch.device,
        dim: int,
        n_gt_xs: int,
        seed: int | None = None,
    ) -> None:
        """Initialize the target.

        Args:
            device: The device on which the target is stored.
            dim: The dimension of the target.
            n_gt_xs: The number of ground truth samples to sample.
            seed: The seed for the random number generator.
        """
        self.device = device
        self.dim = dim
        try:
            self.gt_xs = self.sample(n_gt_xs, seed)
            self.gt_xs_log_rewards = self.log_reward(self.gt_xs)
        except NotImplementedError:
            self.gt_xs = None
            self.gt_xs_log_rewards = None

    @abstractmethod
    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Log reward function.

        Args:
            x: The input tensor.

        Returns:
            The log rewards for the input tensor.
        """
        raise NotImplementedError

    def grad_log_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of the log reward function.

        Args:
            x: The input tensor.

        Returns:
            The gradient of the log reward function.
        """
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.log_reward(copy_x).sum().backward()
                lgv = copy_x.grad
                assert lgv is not None
        return lgv.data

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        """Sample from the target.

        Args:
            batch_size: The number of samples to sample.
            seed: The seed for the random number generator.

        Returns:
            The samples.
        """
        raise NotImplementedError

    def gt_logz(self) -> float:
        """Log partition function of the target.

        Returns:
            The log partition function.
        """
        raise NotImplementedError

    def visualize(self, samples: torch.Tensor | None = None, show: bool = False) -> None:
        """Visualize the target.

        Args:
            samples: The samples to visualize.
            show: Whether to show the plot.
        """
        raise NotImplementedError

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached sample from the target.

        Args:
            batch_size: The number of samples to sample.
            seed: The seed for the random number generator.

        Returns:
            The samples and the log rewards.
        """
        if self.gt_xs is None or batch_size != self.gt_xs.size(0):
            self.gt_xs = self.sample(batch_size, seed)
            self.gt_xs_log_rewards = self.log_reward(self.gt_xs)
        assert self.gt_xs_log_rewards is not None
        return self.gt_xs, self.gt_xs_log_rewards


class SimpleGaussianMixtureTarget(BaseTarget):
    """Simple Gaussian Mixture Target distribution.

    This target distribution is adapted from https://github.com/DenisBless/variational_sampling_methods/blob/main/targets/gaussian_mixture.py.

    Attributes:
        ...
    """

    def __init__(
        self,
        num_components: int = 2,
        dim: int = 2,
        mean_val_range: tuple[float, float] = (-10.0, 10.0),
        mixture_weight_range: tuple[float, float] = (0.3, 0.7),
        degree_of_freedom_adjustment: int = 2,
        seed: int = 3,
        locs: np.ndarray | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        degree_of_freedom_wishart = dim + degree_of_freedom_adjustment

        rng = np.random.default_rng(seed)
        if locs is None:
            locs = rng.uniform(
                mean_val_range[0], mean_val_range[1], size=(num_components, dim)
            )
        elif isinstance(locs, np.ndarray):
            assert locs.shape == (num_components, dim)
            assert (locs >= mean_val_range[0]).all() and (
                locs <= mean_val_range[1]
            ).all(), f"Locs must be within the mean value range {mean_val_range}"

        covariances = []
        for _ in range(num_components):
            cov_matrix = wishart.rvs(
                df=degree_of_freedom_wishart, scale=np.eye(dim), random_state=rng
            )
            covariances.append(cov_matrix)
        mixture_weights = rng.uniform(
            mixture_weight_range[0], mixture_weight_range[1], size=num_components
        )
        mixture_weights = mixture_weights / mixture_weights.sum()

        # Convert to torch tensors
        print("+ Gaussian Mixture Target initialization:")
        print("+ num_components : ", num_components)
        print("+ locs : ", locs)
        print("+ covariances : ", covariances)
        print("+ mixture_weights : ", mixture_weights)

        locs = torch.tensor(locs, device=device)  # type: ignore
        covariances = torch.tensor(covariances, device=device)
        mixture_weights = torch.tensor(mixture_weights, device=device)

        # Define the distribution
        self.distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=mixture_weights),
            torch.distributions.MultivariateNormal(
                loc=locs,  # type: ignore
                covariance_matrix=covariances,
            ),
        )

        self.plot_border = [1.5 * mean_val_range[0], 1.5 * mean_val_range[1]]

        super().__init__(device=device, dim=dim, n_gt_xs=10_000, seed=seed)

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Log reward function for the SimpleGaussianMixtureTarget.

        Args:
            x: The input tensor.

        Returns:
            The log rewards for the input tensor.
        """
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        log_probs = self.distribution.log_prob(x)
        if not batched:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        """Sample from the SimpleGaussianMixtureTarget.

        Args:
            batch_size: The number of samples to sample.
            seed: The seed for the random number generator.

        Returns:
            The samples.
        """
        if seed is not None:
            set_seed(seed)
        return self.distribution.sample((batch_size,))

    def visualize(
        self,
        samples: torch.Tensor | None = None,
        show: bool = False,
        prefix: str = "",
        linspace_n_steps: int = 100,
        max_n_samples: int = 500,
    ) -> None:
        """Visualize the distribution.

        Args:
            samples: The samples to visualize.
            show: Whether to show the plot.
            prefix: The prefix for the plot file name.
            linspace_n_steps: The number of steps in the linspace.
            max_n_samples: The maximum number of samples to visualize.
        """

        if self.dim != 2:
            raise ValueError(
                f"Visualization is only supported for 2D, but got {self.dim}D"
            )

        fig = plt.figure()
        ax = fig.add_subplot()

        x, y = torch.meshgrid(
            torch.linspace(
                self.plot_border[0],
                self.plot_border[1],
                linspace_n_steps,
                device=self.device,
            ),
            torch.linspace(
                self.plot_border[0],
                self.plot_border[1],
                linspace_n_steps,
                device=self.device,
            ),
        )
        grid = torch.stack([x.ravel(), y.ravel()], dim=1)
        pdf_values = torch.exp(self.distribution.log_prob(grid)).reshape(x.shape)
        ax.contourf(x, y, pdf_values, levels=20)  # , cmap='viridis')
        if samples is not None:
            plt.scatter(
                samples[:max_n_samples, 0].clamp(
                    self.plot_border[0], self.plot_border[1]
                ),
                samples[:max_n_samples, 1].clamp(
                    self.plot_border[0], self.plot_border[1]
                ),
                c="r",
                alpha=0.5,
                marker="x",
            )

        # Add dashed lines at 0
        ax.axhline(
            y=0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="y=0"
        )
        ax.axvline(
            x=0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="x=0"
        )

        # Add dashed lines at each mode
        modes = self.distribution.component_distribution.loc
        for i, mode in enumerate(modes):
            mode_x = mode[0].item()
            mode_y = mode[1].item()
            ax.axhline(
                y=mode_y,
                color="yellow",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"mode {i+1}: y={mode_y:.2f}",
            )
            ax.axvline(
                x=mode_x,
                color="yellow",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"mode {i+1}: x={mode_x:.2f}",
            )

        # Set x-ticks and y-ticks to show extremes and special values
        x_tick_positions = [self.plot_border[0], self.plot_border[1]]
        y_tick_positions = [self.plot_border[0], self.plot_border[1]]
        x_tick_labels = [f"{self.plot_border[0]:.1f}", f"{self.plot_border[1]:.1f}"]
        y_tick_labels = [f"{self.plot_border[0]:.1f}", f"{self.plot_border[1]:.1f}"]

        plt.xticks(x_tick_positions, x_tick_labels)
        plt.yticks(y_tick_positions, y_tick_labels)

        # Add legend
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)

        if show:
            plt.show()
        else:
            os.makedirs("viz", exist_ok=True)
            plt.savefig(f"viz/{prefix}simple_gmm.png")

        plt.close()


if __name__ == "__main__":
    env = SimpleGaussianMixtureTarget()
    env.visualize()
