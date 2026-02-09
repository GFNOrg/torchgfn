import logging
import math
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
from scipy.stats import wishart

from gfn.actions import Actions
from gfn.env import Env
from gfn.gym.helpers.diffusion_utils import viz_2d_slice
from gfn.states import States
from gfn.utils.common import filter_kwargs_for_callable, temporarily_set_seed

logger = logging.getLogger(__name__)

# Lightweight typing alias for the target registry entries.
TargetEntry = tuple[type["BaseTarget"], dict[str, Any]]


###############################
### Target energy functions ###
###############################


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
        seed: int = 0,
        plot_border: float | tuple[float, float, float, float] | None = None,
    ) -> None:
        """Initialize the target.

        Args:
            device: The device on which the target is stored.
            dim: The dimension of the target.
            n_gt_xs: The number of ground truth samples to sample.
            seed: The seed for the random number generator.
            plot_border: The border for the plotting. (left, right, bottom, top)
        """
        self.device = device
        self.dim = dim
        if isinstance(plot_border, float):
            plot_border = (-plot_border, plot_border, -plot_border, plot_border)
        self.plot_border = cast(tuple[float, float, float, float] | None, plot_border)
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
            copy_x = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                log_reward = self.log_reward(copy_x).sum()
                log_reward.backward()
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

    def visualize(
        self,
        samples: torch.Tensor | None = None,
        show: bool = False,
        prefix: str = "",
        linspace_n_steps: int = 100,
        max_n_samples: int = 1000,
    ) -> None:
        """Visualize the target.

        Args:
            samples: The samples to visualize.
            show: Whether to show the plot.
            prefix: The prefix for the plot file name.
            linspace_n_steps: The number of steps in the linspace.
            max_n_samples: The maximum number of samples to visualize.
        """
        raise NotImplementedError

    def cached_sample(
        self, batch_size: int, seed: int | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Cached sample from the target.

        Args:
            batch_size: The number of samples to sample.
            seed: The seed for the random number generator.

        Returns:
            The samples and the log rewards.
        """

        if (
            self.gt_xs is not None
            and self.gt_xs_log_rewards is not None
            and self.gt_xs.size(0) >= batch_size
        ):
            return self.gt_xs[:batch_size], self.gt_xs_log_rewards[:batch_size]
        else:
            if self.gt_xs is None or self.gt_xs_log_rewards is None:
                try:
                    self.gt_xs = self.sample(batch_size, seed)
                    self.gt_xs_log_rewards = self.log_reward(self.gt_xs)
                except NotImplementedError:
                    self.gt_xs = None
                    self.gt_xs_log_rewards = None
            elif batch_size > self.gt_xs.size(0):
                gt_xs_new = self.sample(batch_size - self.gt_xs.size(0), seed)
                gt_xs_log_rewards_new = self.log_reward(gt_xs_new)
                self.gt_xs = torch.cat([self.gt_xs, gt_xs_new], dim=0)
                self.gt_xs_log_rewards = torch.cat(
                    [self.gt_xs_log_rewards, gt_xs_log_rewards_new], dim=0
                )
            return self.gt_xs, self.gt_xs_log_rewards


class SimpleGaussianMixture(BaseTarget):
    """Simple Gaussian Mixture Target distribution.

    This target distribution is adapted from
    https://github.com/DenisBless/variational_sampling_methods/blob/main/targets/gaussian_mixture.py.

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

        logger.info("+ Gaussian Mixture Target initialization:")
        logger.info(f"+ num_components: {num_components}")
        logger.info(f"+ mixture_weights: {mixture_weights}")
        for i, (loc, cov) in enumerate(zip(locs, covariances)):
            loc_str = np.array2string(loc, precision=2, separator=", ").replace(
                "\n", " "
            )
            cov_str = np.array2string(cov, precision=2, separator=", ").replace(
                "\n", " "
            )
            logger.info(f"\tComponent {i+1}: loc={loc_str}, cov={cov_str}")

        # Convert to torch tensors
        dtype = torch.get_default_dtype()
        locs_tsr = torch.tensor(locs, device=device, dtype=dtype)
        covariances_tsr = torch.tensor(covariances, device=device, dtype=dtype)
        mixture_weights_tsr = torch.tensor(mixture_weights, device=device, dtype=dtype)

        # Define the distribution
        self.distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=mixture_weights_tsr),
            torch.distributions.MultivariateNormal(
                loc=locs_tsr,
                covariance_matrix=covariances_tsr,
            ),
        )

        super().__init__(
            device=device,
            dim=dim,
            n_gt_xs=10_000,
            plot_border=(
                1.5 * mean_val_range[0],
                1.5 * mean_val_range[1],
                1.5 * mean_val_range[0],
                1.5 * mean_val_range[1],
            ),
            seed=seed,
        )

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Log reward function for the SimpleGaussianMixtureTarget.

        Args:
            x: The input tensor.

        Returns:
            The log rewards for the input tensor.
        """
        not_batched = x.ndim == 1
        if not_batched:
            x = x.unsqueeze(0)

        log_probs = self.distribution.log_prob(x)
        if not_batched:
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
        with temporarily_set_seed(seed) if seed is not None else nullcontext():
            samples = self.distribution.sample((batch_size,))
        return samples

    def gt_logz(self) -> float:
        """Log partition function of the target.

        Returns:
            The log partition function.
        """
        return 0.0

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
        assert self.plot_border is not None, "Visualization requires a plot border."

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
                self.plot_border[2],
                self.plot_border[3],
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
                    self.plot_border[2], self.plot_border[3]
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
        y_tick_positions = [self.plot_border[2], self.plot_border[3]]
        x_tick_labels = [f"{self.plot_border[0]:.1f}", f"{self.plot_border[1]:.1f}"]
        y_tick_labels = [f"{self.plot_border[2]:.1f}", f"{self.plot_border[3]:.1f}"]

        plt.xticks(x_tick_positions, x_tick_labels)
        plt.yticks(y_tick_positions, y_tick_labels)

        # Add legend
        ax.legend(fontsize="small", framealpha=0.8)

        if show:
            plt.show()
        else:
            os.makedirs("viz", exist_ok=True)
            plt.savefig(f"viz/{prefix}simple_gmm.png")

        plt.close()


class Funnel(BaseTarget):
    """Neal's funnel distribution target.

    x0 ~ Normal(0, std^2), and for i >= 1: xi | x0 ~ Normal(0, exp(x0)).

    Args:
        dim: Total dimensionality (x0 plus dim-1 conditional coordinates).
        std: Standard deviation for the marginal prior on x0.
        device: Torch device.
        seed: RNG seed.
    """

    def __init__(
        self,
        dim: int = 10,
        std: float = 1.0,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
    ) -> None:
        dtype = torch.get_default_dtype()
        self.dist_dominant = D.Normal(
            torch.tensor([0.0], device=device, dtype=dtype),
            torch.tensor([std], device=device, dtype=dtype),
        )
        super().__init__(
            device=device, dim=dim, n_gt_xs=10_000, plot_border=10.0, seed=seed
        )

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density of Neal's funnel distribution.

        Returns log p(x0) + sum_i log p(xi | x0), i=1..dim-1.
        """
        not_batched = x.ndim == 1
        if not_batched:
            x = x.unsqueeze(0)

        dominant_x = x[:, 0]
        log_prob_x0 = self.dist_dominant.log_prob(dominant_x)

        log_sigma = 0.5 * x[:, 0:1]
        sigma2 = torch.exp(x[:, 0:1])
        neg_log_prob_other = (
            0.5 * np.log(2 * np.pi) + log_sigma + 0.5 * x[:, 1:] ** 2 / sigma2
        )
        log_prob_other = torch.sum(-neg_log_prob_other, dim=-1)

        log_prob = log_prob_x0 + log_prob_other
        if not_batched:
            log_prob = log_prob.squeeze(0)
        return log_prob

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temporarily_set_seed(seed) if seed is not None else nullcontext():
            # Sample x0 ~ Normal(0, std^2)
            x0 = self.dist_dominant.sample((batch_size,))

            # Sample xs | x0 with variance exp(x0) => std = exp(0.5 * x0)
            eps = torch.randn(batch_size, self.dim - 1, device=self.device)

        xs = eps * torch.exp(0.5 * x0)
        return torch.cat([x0, xs], dim=-1)

    def gt_logz(self) -> float:
        """Log partition function of the target.

        Returns:
            The log partition function.
        """
        return 0.0

    def visualize(
        self,
        samples: torch.Tensor | None = None,
        show: bool = False,
        prefix: str = "",
        linspace_n_steps: int = 100,
        max_n_samples: int = 500,
    ) -> None:
        """Visualize only supported for 2D (x0, x1)."""
        assert self.plot_border is not None, "Visualization requires a plot border."

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i in range(0, 2):
            viz_2d_slice(
                axes[i],
                self,
                (0, i + 1),
                samples,
                plot_border=self.plot_border,
                use_log_reward=True,
            )

        plt.tight_layout()
        if show:
            plt.show()
        else:
            os.makedirs("viz", exist_ok=True)
            fig.savefig(f"viz/{prefix}funnel.png")

        plt.close()


class ManyWell(BaseTarget):
    """Many-well target distribution.

    The 32D (default) instance is the concatenation of 16 identical 2D double-well
    components. Each 2D block (x1, x2) has unnormalized log-density
        log p(x1, x2) = -x1^4 + 6 x1^2 + 0.5 x1 - 0.5 x2^2 + C
    The overall log-density is the sum over all independent 2D blocks.

    Sampling uses rejection sampling for the x1 coordinate in each block with a
    simple Gaussian mixture proposal, and standard Normal for x2.
    """

    def __init__(
        self,
        dim: int = 32,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
    ) -> None:
        assert (
            dim % 2 == 0
        ), "ManyWellTarget requires an even dimension (pairs of coordinates)."

        # Simple mixture proposal for x1: 3 equally weighted Normals
        self.component_mix = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=device)
        self.means = torch.tensor([-2.0, 0.0, 2.0], device=device)
        self.scales = torch.tensor([1.0, 1.0, 1.0], device=device)

        super().__init__(
            device=device,
            dim=dim,
            n_gt_xs=10_000,
            plot_border=3.0,
            seed=seed,
        )

        self.Z_x1 = 11784.50927
        self.Z_x2 = np.sqrt(2 * np.pi)

    @staticmethod
    def _block_log_density(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Per 2D block log p(x1, x2) up to an additive constant
        return -(x1**4) + 6.0 * (x1**2) + 0.5 * x1 - 0.5 * (x2**2)

    def log_reward(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        if not batched:
            x = x.unsqueeze(0)

        assert (
            self.dim % 2 == 0
        ), "ManyWellTarget requires an even dimension (pairs of coordinates)."

        # Reshape into (..., n_blocks, 2)
        n_blocks = self.dim // 2
        x_pairs = x.view(x.shape[0], n_blocks, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        block_logs = self._block_log_density(x1, x2)
        logp = block_logs.sum(dim=-1)
        if not batched:
            logp = logp.squeeze(0)

        return logp

    def _make_proposal(self) -> D.MixtureSameFamily:
        mix = D.Categorical(self.component_mix)
        com = D.Normal(self.means, self.scales)

        return D.MixtureSameFamily(mixture_distribution=mix, component_distribution=com)

    def _compute_envelope_k(self, proposal: D.Distribution) -> float:
        # Coarse grid-based envelope to upper bound target/proposal ratio
        grid = torch.linspace(-6.0, 6.0, 201, device=self.device)
        target_log = -(grid**4) + 6.0 * (grid**2) + 0.5 * grid
        prop_log = proposal.log_prob(grid)
        k = torch.exp(target_log - prop_log).max().item()

        return float(1.2 * k)  # small safety margin

    @staticmethod
    def _rejection_sampling_x1(
        n_samples: int, proposal: D.Distribution, k: float
    ) -> torch.Tensor:
        # Basic rejection sampler with vectorized batches and refill loop
        collected: list[torch.Tensor] = []
        remaining = n_samples
        while remaining > 0:
            # Oversample for higher acceptance rate
            z = proposal.sample((remaining * 10,))
            u = torch.rand_like(z) * (k * torch.exp(proposal.log_prob(z)))
            accept = torch.exp(-(z**4) + 6.0 * (z**2) + 0.5 * z) > u
            accepted = z[accept]
            if accepted.shape[0] == 0:
                continue
            if accepted.shape[0] >= remaining:
                collected.append(accepted[:remaining])
                remaining = 0
            else:
                collected.append(accepted)
                remaining -= accepted.shape[0]

        return torch.cat(collected, dim=0)

    def sample(self, batch_size: int, seed: int | None = None) -> torch.Tensor:
        with temporarily_set_seed(seed) if seed is not None else nullcontext():
            n_blocks = self.dim // 2

            proposal = self._make_proposal()
            k = self._compute_envelope_k(proposal)

            xs = torch.empty(batch_size, self.dim, device=self.device)
            standard_normal = D.Normal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            for b in range(n_blocks):
                x1 = self._rejection_sampling_x1(batch_size, proposal, k)
                x2 = standard_normal.sample((batch_size,))
                xs[:, 2 * b] = x1
                xs[:, 2 * b + 1] = x2
        return xs

    def gt_logz(self) -> float:
        return (self.dim // 2) * (np.log(self.Z_x1) + np.log(self.Z_x2))

    def visualize(
        self,
        samples: torch.Tensor | None = None,
        show: bool = False,
        prefix: str = "",
        linspace_n_steps: int = 100,
        max_n_samples: int = 500,
    ) -> None:
        assert self.plot_border is not None, "Visualization requires a plot border."

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (idx1, idx2) in enumerate([(0, 2), (1, 2)]):
            viz_2d_slice(
                axes[i],
                self,
                (idx1, idx2),
                samples,
                plot_border=self.plot_border,
                use_log_reward=True,
            )

        plt.tight_layout()
        if show:
            plt.show()
        else:
            os.makedirs("viz", exist_ok=True)
            fig.savefig(f"viz/{prefix}manywell.png")

        plt.close()


######################################
### Diffusion Sampling Environment ###
######################################


class DiffusionSampling(Env):
    """Diffusion sampling environment.

    Attributes:
        target: The target distribution.
        device: The device to use.
    """

    # Registry of available targets.
    DIFFUSION_TARGETS: dict[str, TargetEntry] = {
        "gmm2": (SimpleGaussianMixture, {"num_components": 2}),  # 2D
        "gmm4": (SimpleGaussianMixture, {"num_components": 4}),  # 2D
        "gmm8": (SimpleGaussianMixture, {"num_components": 8}),  # 2D
        "easy_funnel": (Funnel, {"std": 1.0}),  # 10D
        "hard_funnel": (Funnel, {"std": 3.0}),  # 10D
        "many_well": (ManyWell, {}),  # 32D
    }

    def __init__(
        self,
        target_str: str,
        target_kwargs: dict[str, Any] | None,
        num_discretization_steps: float,
        device: torch.device = torch.device("cpu"),
        debug: bool = False,
    ) -> None:
        """Initialize the DiffusionSampling environment.

        Args:
            target_str: The string identifier for the target.
            target_kwargs: The keyword arguments for the target, overriding the
                defaults.
            num_discretization_steps: The number of discretization steps.
            device: The device to use.
            debug: If True, emit States with debug guards (not compile-friendly).
        """

        # Initalize the target.
        if target_str not in DiffusionSampling.DIFFUSION_TARGETS:
            DiffusionSampling.list_available_targets()
            raise ValueError(f"Invalid target: {target_str}")

        target_cls, default_kwargs = DiffusionSampling.DIFFUSION_TARGETS[target_str]
        merged_kwargs = filter_kwargs_for_callable(
            target_cls.__init__,
            {**default_kwargs, **(target_kwargs or {})},
        )
        logger.info("DiffusionSampling:")
        logger.info(
            f"+ Initalizing target {target_cls.__name__} with kwargs: {merged_kwargs}"
        )
        self.target = target_cls(device=device, **merged_kwargs)

        self.dim = self.target.dim
        self.dt = 1.0 / num_discretization_steps

        # Note that all states in this environment contain a time (last) dimension.
        # This is crucial to prevent cycles in the DAG of the GFlowNet.
        s0 = torch.zeros((self.dim + 1,), device=device)  # + 1 for time

        super().__init__(
            s0=s0,
            state_shape=(self.dim + 1,),
            action_shape=(self.dim,),
            # dummy action is never used since all trajectories are terminated at
            # time == 1 (i.e., we don't need to pad shorter trajectories)
            dummy_action=torch.full(
                (self.dim,), 0.0, device=device, dtype=torch.get_default_dtype()
            ),
            exit_action=torch.full(
                (self.dim,),
                -float("inf"),
                device=device,
                dtype=torch.get_default_dtype(),
            ),
            debug=debug,
        )

    def make_states_class(self) -> type[States]:
        """Returns the States class for diffusion sampling."""
        env = self

        class DiffusionSamplingStates(States):
            """States for diffusion sampling."""

            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf

            @property
            def is_initial_state(self) -> torch.Tensor:
                """Returns a tensor that is True for states that are s0

                When time is close enought to 0.0 (considering floating point errors),
                the state is s0.
                """
                return (self.tensor[..., -1] - 0.0) < env.dt * 1e-2

        return DiffusionSamplingStates

    @classmethod
    def list_available_targets(cls) -> dict[str, dict[str, Any]]:
        """Return metadata about available targets and their default kwargs.

        This helper allows users to easily see which kwargs are provided by default
        for each alias. Note that accepted/required kwargs are determined by each
        target class's constructor signature.
        """
        out = {}
        logger.info("Available DiffusionSampling targets:")
        for alias, (cls, defaults) in cls.DIFFUSION_TARGETS.items():
            logger.info(f"+ {alias}: {cls.__name__} with kwargs: {defaults}")
            out[alias] = {"class": cls.__name__, "defaults": dict(defaults)}

        return out

    def step(self, states: States, actions: Actions) -> States:
        """Step function for the SimpleGaussianMixtureModel environment.

        Args:
            states: The current states.
            actions: The actions, which correspond to the changes to the states.

        Returns:
            The next states.
        """
        next_states_tensor = states.tensor.clone()
        next_states_tensor[..., :-1] = next_states_tensor[..., :-1] + actions.tensor
        next_states_tensor[..., -1] = next_states_tensor[..., -1] + self.dt
        return self.States(next_states_tensor)

    def backward_step(self, states: States, actions: Actions) -> States:
        """Backward step function for the SimpleGaussianMixtureModel environment.

        Args:
            states: The current states.
            actions: The actions, which correspond to the changes to the states.

        Returns:
            The previous states.
        """
        prev_states_tensor = states.tensor.clone()
        prev_states_tensor[..., :-1] = prev_states_tensor[..., :-1] - actions.tensor
        prev_states_tensor[..., -1] = prev_states_tensor[..., -1] - self.dt
        return self.States(prev_states_tensor)

    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Check if the actions are valid.

        Args:
            states: The current states.
            actions: The actions to check.

        Returns:
            True if the actions are valid, False otherwise.
        """
        time = states.tensor[..., -1].flatten()[0].item()
        # TODO: support randomized discretization
        assert (
            states.tensor[..., -1] == time
        ).all(), "Time must be the same for all states in the batch"

        if not backward and time == 1.0:  # Terminate if time == 1.0 for forward steps
            sf = cast(torch.Tensor, self.sf)
            return bool((actions.tensor == sf[:-1]).all().item())
        elif backward and time == 0.0:  # Return to s0 if time == 0.0 for backward steps
            s0 = cast(torch.Tensor, self.s0)
            return bool((actions.tensor == s0[:-1]).all().item())
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

    @staticmethod
    def density_metrics(
        fwd_log_pfs: torch.Tensor,
        fwd_log_pbs: torch.Tensor,
        fwd_log_rewards: torch.Tensor,
        log_Z_learned: float,
        bwd_log_pfs: torch.Tensor | None = None,
        bwd_log_pbs: torch.Tensor | None = None,
        bwd_log_rewards: torch.Tensor | None = None,
        gt_log_Z: float | None = None,
    ) -> dict:
        bsz = fwd_log_pfs.shape[1]
        assert bsz == fwd_log_pbs.shape[1] == fwd_log_rewards.shape[0]
        assert fwd_log_pfs.ndim == fwd_log_pbs.ndim == 2

        log_weights = fwd_log_rewards + fwd_log_pbs.sum(0) - fwd_log_pfs.sum(0)
        iw_elbo = torch.logsumexp(log_weights, dim=0) - math.log(bsz)
        elbo = log_weights.mean().item()

        # EUBO, if the ground truth samples are available
        if (
            bwd_log_rewards is not None
            and bwd_log_pfs is not None
            and bwd_log_pbs is not None
        ):
            gt_bsz = bwd_log_pfs.shape[1]
            assert gt_bsz == bwd_log_pbs.shape[1] == bwd_log_rewards.shape[0]
            assert bwd_log_pfs.ndim == bwd_log_pbs.ndim == 2
            eubo = (
                (bwd_log_rewards + bwd_log_pbs.sum(0) - bwd_log_pfs.sum(0)).mean().item()
            )
        else:
            eubo = float("nan")

        ess = 1.0 / (log_weights.softmax(0) ** 2).sum().item()

        metrics = {
            "log_Z_learned": log_Z_learned,
            "elbo": elbo,
            "eubo": eubo,
            "iw_elbo": iw_elbo,
            "Δ_elbo": (gt_log_Z - elbo) if gt_log_Z is not None else float("nan"),
            "Δ_eubo": (gt_log_Z - eubo) if gt_log_Z is not None else float("nan"),
            "Δ_iw_elbo": (gt_log_Z - iw_elbo) if gt_log_Z is not None else float("nan"),
            "ess(%)": ess / bsz * 100,
        }
        return metrics
