"""This file contains utilitary functions for the Box environment."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Beta, Categorical, Distribution, MixtureSameFamily

from gfn.estimators import Estimator
from gfn.gym import Box
from gfn.states import States
from gfn.utils.modules import MLP

PI_2_INV: float = 2.0 / torch.pi
PI_2: float = torch.pi / 2.0
CLAMP: float = torch.finfo(torch.float).eps


class QuarterCircle(Distribution):
    """Represents distributions on quarter circles.

    The distributions are Mixture of Beta distributions on the possible angle range.

    When a state is of norm <= delta, and northeastern=False, then the distribution
    is a Dirac at the state (i.e. the only possible parent is s_0).

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment.

    Attributes:
        delta (float): The radius of the quarter disk.
        northeastern (bool): Whether the quarter disk is northeastern or southwestern.
        n_states (int): The number of states.
        n_components (int): The number of components in the mixture.
        centers (States): The centers of the distribution.
        base_dist (MixtureSameFamily): The base distribution.
        min_angles (Tensor): The minimum angles.
        max_angles (Tensor): The maximum angles.
    """

    delta: float
    northeastern: bool
    n_states: int
    n_components: int
    centers: States
    base_dist: MixtureSameFamily
    min_angles: Tensor
    max_angles: Tensor

    def __init__(
        self,
        delta: float,
        northeastern: bool,
        centers: States,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            northeastern: whether the quarter disk is northeastern or southwestern.
            centers: the centers of the distribution with shape `(n_states, 2)`.
            mixture_logits: Tensor of shape `(n_states, n_components)` containing the
                logits of the mixture of Beta distributions.
            alpha: Tensor of shape `(n_states, n_components)` containing the alpha
                parameters of the Beta distributions.
            beta: Tensor of shape `(n_states, n_components)` containing the beta
                parameters of the Beta distributions.
        """
        self.delta = delta
        self.northeastern = northeastern
        self.n_states, self.n_components = mixture_logits.shape

        assert centers.tensor.shape == (self.n_states, 2)
        self.centers = centers

        assert alpha.shape == (self.n_states, self.n_components)
        assert beta.shape == (self.n_states, self.n_components)
        self.base_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )

        self.min_angles, self.max_angles = self.get_min_and_max_angles()

    def get_min_and_max_angles(self) -> Tuple[Tensor, Tensor]:
        """Computes the minimum and maximum angles for the distribution.

        Returns:
            A tuple of two tensors of shape `(n_states,)` containing the minimum and
            maximum angles, respectively.
        """
        if self.northeastern:
            min_angles = torch.where(
                self.centers.tensor[..., 0] <= 1 - self.delta,
                0.0,
                PI_2_INV * torch.arccos((1 - self.centers.tensor[..., 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers.tensor[..., 1] <= 1 - self.delta,
                1.0,
                PI_2_INV * torch.arcsin((1 - self.centers.tensor[..., 1]) / self.delta),
            )
        else:
            min_angles = torch.where(
                self.centers.tensor[..., 0] >= self.delta,
                0.0,
                PI_2_INV * torch.arccos((self.centers.tensor[..., 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers.tensor[..., 1] >= self.delta,
                1.0,
                PI_2_INV * torch.arcsin((self.centers.tensor[..., 1]) / self.delta),
            )

        assert min_angles.shape == (self.n_states,)
        assert max_angles.shape == (self.n_states,)
        return min_angles, max_angles

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            The sampled actions of shape `(sample_shape, n_states, 2)`.
        """
        base_01_samples = self.base_dist.sample(sample_shape=sample_shape)

        sampled_angles = (
            self.min_angles + (self.max_angles - self.min_angles) * base_01_samples
        )
        sampled_angles = PI_2 * sampled_angles

        sampled_actions = self.delta * torch.stack(
            [torch.cos(sampled_angles), torch.sin(sampled_angles)],
            dim=-1,
        )

        if not self.northeastern:
            # when centers are of norm <= delta, the distribution is a Dirac at the center
            centers_in_quarter_disk = (
                torch.norm(self.centers.tensor, dim=-1) <= self.delta
            )
            # repeat the centers_in_quarter_disk tensor to be of shape (*centers.batch_shape, 2)
            centers_in_quarter_disk = centers_in_quarter_disk.unsqueeze(-1).repeat(
                *([1] * len(self.centers.batch_shape)), 2
            )
            sampled_actions = torch.where(
                centers_in_quarter_disk,
                self.centers.tensor,
                sampled_actions,
            )

            # Sometimes, when a point is at the border of the square (e.g. (1e-8, something) or (something, 1e-9))
            # Then the approximation errors lead to the sampled_actions being slightly larger than the state or slightly
            # negative at the low coordinate. So what we do is we set the sampled_action to be half that coordinate

            sampled_actions = torch.where(
                sampled_actions > self.centers.tensor,
                self.centers.tensor / 2,
                sampled_actions,
            )

            sampled_actions = torch.where(
                sampled_actions < 0,
                self.centers.tensor / 2,
                sampled_actions,
            )
        else:
            # When at the border of the square,
            # the approximation errors might lead to the sampled_actions being slightly negative at the high coordinate
            # We set the sampled_action to be 0
            # This is actually of no impact, given that the true action that would be sampled is the exit action
            sampled_actions = torch.where(
                sampled_actions < 0,
                0,
                sampled_actions,
            )
            if torch.any(
                torch.abs(torch.norm(sampled_actions, dim=-1) - self.delta) > 1e-5
            ):
                raise ValueError("Sampled actions should be of norm delta ish")

        assert sampled_actions.shape == sample_shape + (self.n_states, 2)
        return sampled_actions

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        assert sampled_actions.shape[-1] == 2
        batch_shape = sampled_actions.shape[:-1]

        sampled_actions = sampled_actions.to(
            torch.double
        )  # Arccos is very brittle, so we use double precision
        sampled_actions.clamp_(
            min=0.0, max=self.delta
        )  # Should be the case already - just to avoid numerical issues
        sampled_angles = torch.arccos(sampled_actions[..., 0] / self.delta) / (PI_2)

        base_01_samples = (sampled_angles - self.min_angles) / (
            self.max_angles - self.min_angles
        ).clamp_(min=CLAMP, max=1 - CLAMP)

        if not self.northeastern:
            # Ideally, we shouldn't need this
            # But it is used in the original implementation, so we keep it. It helps with numerical issues.
            base_01_samples = base_01_samples.clamp(1e-4, 1 - 1e-4)

        # Ugly hack: when some of the sampled actions are -infinity (exit action), the corresponding value is nan
        # And we don't really care about the log prob of the exit action
        # So we first need to replace nans by anything between 0 and 1, say 0.5
        base_01_samples = torch.where(
            torch.isnan(base_01_samples),
            torch.ones_like(base_01_samples) * 0.5,
            base_01_samples,
        ).clamp_(min=CLAMP, max=1 - CLAMP)

        # Another hack: when backward (northeastern=False), sometimes the sampled_actions are equal to the centers
        # In this case, the base_01_samples are close to 0 because of approximations errors. But they do not count
        # when evaluating the logpros, so we just bump them to CLAMP so that Beta.log_prob does not throw an error
        if not self.northeastern:
            base_01_samples = torch.where(
                torch.norm(self.centers.tensor, dim=-1) <= self.delta,
                torch.ones_like(base_01_samples) * CLAMP,
                base_01_samples,
            ).clamp_(min=CLAMP, max=1 - CLAMP)
        base_01_samples = base_01_samples.to(torch.float)
        base_01_logprobs = self.base_dist.log_prob(base_01_samples)

        if not self.northeastern:
            base_01_logprobs = base_01_logprobs.clamp_max(100)

        logprobs = (
            base_01_logprobs
            - np.log(self.delta)
            - np.log(np.pi / 2)
            - torch.log((self.max_angles - self.min_angles).clamp_(min=CLAMP))
            # The clamp doesn't really matter, because if we need to clamp, it means the actual action is exit action
        )

        if not self.northeastern:
            # when centers are of norm <= delta, the distribution is a Dirac at the center
            logprobs = torch.where(
                torch.norm(self.centers.tensor, dim=-1) <= self.delta,
                torch.zeros_like(logprobs),
                logprobs,
            )

        if torch.any(torch.isinf(logprobs)) or torch.any(torch.isnan(logprobs)):
            raise ValueError("logprobs contains inf or nan")

        assert logprobs.shape == batch_shape
        return logprobs


class QuarterDisk(Distribution):
    """Represents a distribution on the northeastern quarter disk.

    The radius and the angle follow Mixture of Betas distributions.

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment.

    Attributes:
        delta (float): The radius of the quarter disk.
        mixture_logits (Tensor): The logits of the mixture of Beta distributions.
        base_r_dist (MixtureSameFamily): The base distribution for the radius.
        base_theta_dist (MixtureSameFamily): The base distribution for the angle.
        n_components (int): The number of components in the mixture.
    """

    delta: float
    mixture_logits: Tensor
    base_r_dist: MixtureSameFamily
    base_theta_dist: MixtureSameFamily
    n_components: int

    def __init__(
        self,
        delta: float,
        mixture_logits: Tensor,
        alpha_r: Tensor,
        beta_r: Tensor,
        alpha_theta: Tensor,
        beta_theta: Tensor,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            mixture_logits: Tensor of shape `(n_components,)` containing the logits of
                the mixture of Beta distributions.
            alpha_r: Tensor of shape `(n_components,)` containing the alpha parameters
                of the Beta distributions for the radius.
            beta_r: Tensor of shape `(n_components,)` containing the beta parameters of
                the Beta distributions for the radius.
            alpha_theta: Tensor of shape `(n_components,)` containing the alpha
                parameters of the Beta distributions for the angle.
            beta_theta: Tensor of shape `(n_components,)` containing the beta
                parameters of the Beta distributions for the angle.
        """
        self.delta = delta
        self.mixture_logits = mixture_logits
        (self.n_components,) = mixture_logits.shape

        assert alpha_r.shape == (self.n_components,)
        assert beta_r.shape == (self.n_components,)
        assert alpha_theta.shape == (self.n_components,)
        assert beta_theta.shape == (self.n_components,)

        self.base_r_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha_r, beta_r),
        )

        self.base_theta_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha_theta, beta_theta),
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            The sampled actions of shape `(sample_shape, 2)`.
        """
        base_r_01_samples = self.base_r_dist.sample(sample_shape=sample_shape)
        base_theta_01_samples = self.base_theta_dist.sample(sample_shape=sample_shape)

        sampled_actions = self.delta * (
            torch.stack(
                [
                    base_r_01_samples * torch.cos(PI_2 * base_theta_01_samples),
                    base_r_01_samples * torch.sin(PI_2 * base_theta_01_samples),
                ],
                dim=-1,
            )
        )

        assert sampled_actions.shape == sample_shape + (2,)
        return sampled_actions

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        assert sampled_actions.shape[-1] == 2
        batch_shape = sampled_actions.shape[:-1]

        sampled_actions = sampled_actions.to(
            torch.double
        )  # Arccos is very brittle, so we use double precision
        base_r_01_samples = (
            torch.sqrt(torch.sum(sampled_actions**2, dim=-1))
            / self.delta  # changes from 0 to 1.
        )
        # Debugging, I changed from -1 to 0 in the following line
        base_theta_01_samples = (
            torch.arccos(sampled_actions[..., 0] / (base_r_01_samples * self.delta))
            / PI_2
        ).clamp_(CLAMP, 1 - CLAMP)

        base_r_01_samples = base_r_01_samples.to(torch.float)
        base_theta_01_samples = base_theta_01_samples.to(torch.float)
        logprobs = (
            self.base_r_dist.log_prob(base_r_01_samples)
            + self.base_theta_dist.log_prob(base_theta_01_samples)
            - np.log(self.delta)
            - np.log(PI_2)
            - torch.log(base_r_01_samples * self.delta)
        )

        if torch.any(torch.isinf(logprobs)):
            raise ValueError("logprobs contains inf")

        assert logprobs.shape == batch_shape
        return logprobs


class QuarterCircleWithExit(Distribution):
    """Extends `QuarterCircle` with an exit action.

    When sampling, with probability `exit_probability`, the `exit_action`
    `[-inf, -inf]` is sampled. The `log_prob` function is adjusted accordingly.

    Attributes:
        delta (float): The radius of the quarter disk.
        epsilon (float): The epsilon value to consider the state as being at the
            border of the square.
        centers (States): The centers of the distribution.
        dist_without_exit (QuarterCircle): The distribution without the exit action.
        exit_probability (Tensor): The probability of exiting.
        exit_action (Tensor): The exit action.
        n_states (int): The number of states.
    """

    delta: float
    epsilon: float
    centers: States
    dist_without_exit: QuarterCircle
    exit_probability: Tensor
    exit_action: Tensor
    n_states: int

    def __init__(
        self,
        delta: float,
        centers: States,
        exit_probability: Tensor,
        mixture_logits: Tensor,
        alpha: Tensor,
        beta: Tensor,
        epsilon: float = 1e-4,
    ) -> None:
        """Initializes the distribution.

        Args:
            delta: the radius of the quarter disk.
            centers: the centers of the distribution with shape `(n_states, 2)`.
            exit_probability: Tensor of shape `(n_states,)` containing the
                probability of exiting the quarter disk.
            mixture_logits: Tensor of shape `(n_states, n_components)` containing the
                logits of the mixture of Beta distributions.
            alpha: Tensor of shape `(n_states, n_components)` containing the alpha
                parameters of the Beta distributions.
            beta: Tensor of shape `(n_states, n_components)` containing the beta
                parameters of the Beta distributions.
            epsilon: the epsilon value to consider the state as being at the border
                of the square.
        """
        self.n_states, n_components = mixture_logits.shape
        assert centers.tensor.shape == (self.n_states, 2)
        assert exit_probability.shape == (self.n_states,)
        assert alpha.shape == (self.n_states, n_components)
        assert beta.shape == (self.n_states, n_components)

        self.delta = delta
        self.epsilon = epsilon
        self.centers = centers
        self.dist_without_exit = QuarterCircle(
            delta=delta,
            northeastern=True,
            centers=centers,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
        )
        self.exit_probability = exit_probability
        self.exit_action = torch.FloatTensor([-float("inf"), -float("inf")]).to(
            centers.device
        )

    def sample(self) -> Tensor:
        """Samples from the distribution.

        Returns:
            The sampled actions with shape `(n_states, 2)`.
        """
        actions = self.dist_without_exit.sample()
        repeated_exit_probability = self.exit_probability.repeat((1,))
        exit_mask = torch.bernoulli(repeated_exit_probability).bool()

        # When torch.norm(1 - states, dim=-1) <= env.delta or
        # torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1), we have to exit
        exit_mask[torch.norm(1 - self.centers.tensor, dim=-1) <= self.delta] = True
        exit_mask[torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1)] = True
        actions[exit_mask] = self.exit_action

        assert actions.shape == self.centers.tensor.shape == (self.n_states, 2)

        return actions

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            The log probability of the sampled actions as a tensor of shape `batch_shape`.
        """
        exit = torch.all(
            sampled_actions == torch.full_like(sampled_actions[0], -float("inf")), 1
        )
        logprobs = torch.full_like(self.exit_probability, fill_value=-float("inf"))
        logprobs[~exit] = self.dist_without_exit.log_prob(sampled_actions)[~exit]
        logprobs[~exit] = logprobs[~exit] + torch.log(1 - self.exit_probability)[~exit]
        logprobs[exit] = torch.log(self.exit_probability[exit])
        # When torch.norm(1 - states, dim=-1) <= env.delta, logprobs should be 0
        # When torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1), logprobs should be 0
        logprobs[torch.norm(1 - self.centers.tensor, dim=-1) <= self.delta] = 0.0
        logprobs[torch.any(self.centers.tensor >= 1 - self.epsilon, dim=-1)] = 0.0
        return logprobs


class DistributionWrapper(Distribution):
    """A wrapper that combines `QuarterDisk` and `QuarterCircleWithExit`.

    Attributes:
        idx_is_initial (Tensor): The indices of the initial states.
        idx_not_initial (Tensor): The indices of the non-initial states.
        quarter_disk (Optional[QuarterDisk]): The `QuarterDisk` distribution.
        quarter_circ (Optional[QuarterCircleWithExit]): The `QuarterCircleWithExit`
            distribution.
    """

    idx_is_initial: Tensor
    idx_not_initial: Tensor
    _output_shape: tuple[int, ...]
    quarter_disk: Optional[QuarterDisk]
    quarter_circ: Optional[QuarterCircleWithExit]

    def __init__(
        self,
        states: States,
        delta: float,
        epsilon: float,
        mixture_logits: Tensor,
        alpha_r: Tensor,
        beta_r: Tensor,
        alpha_theta: Tensor,
        beta_theta: Tensor,
        exit_probability: Tensor,
        n_components: int,
        n_components_s0: int,
    ) -> None:
        """Initializes the distribution.

        Args:
            states: The states.
            delta: The radius of the quarter disk.
            epsilon: The epsilon value.
            mixture_logits: The logits of the mixture of Beta distributions.
            alpha_r: The alpha parameters of the Beta distributions for the radius.
            beta_r: The beta parameters of the Beta distributions for the radius.
            alpha_theta: The alpha parameters of the Beta distributions for the angle.
            beta_theta: The beta parameters of the Beta distributions for the angle.
            exit_probability: The probability of exiting.
            n_components: The number of components in the mixture.
            n_components_s0: The number of components in the mixture for s0.
        """
        self.idx_is_initial = torch.where(torch.all(states.tensor == 0, 1))[0]
        self.idx_not_initial = torch.where(torch.any(states.tensor != 0, 1))[0]
        self._output_shape = states.tensor.shape
        self.quarter_disk = None
        if len(self.idx_is_initial) > 0:
            self.quarter_disk = QuarterDisk(
                delta=delta,
                mixture_logits=mixture_logits[self.idx_is_initial[0], :n_components_s0],
                alpha_r=alpha_r[self.idx_is_initial[0], :n_components_s0],
                beta_r=beta_r[self.idx_is_initial[0], :n_components_s0],
                alpha_theta=alpha_theta[self.idx_is_initial[0], :n_components_s0],
                beta_theta=beta_theta[self.idx_is_initial[0], :n_components_s0],
            )
        self.quarter_circ = None
        if len(self.idx_not_initial) > 0:
            self.quarter_circ = QuarterCircleWithExit(
                delta=delta,
                centers=states[self.idx_not_initial],  # Remove initial states.
                exit_probability=exit_probability[self.idx_not_initial],
                mixture_logits=mixture_logits[self.idx_not_initial, :n_components],
                alpha=alpha_theta[self.idx_not_initial, :n_components],
                beta=beta_theta[self.idx_not_initial, :n_components],
                epsilon=epsilon,
            )

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: the shape of the samples to generate.

        Returns:
            A tensor of shape `(sample_shape + self._output_shape)` containing the
            sampled actions.
        """
        output = torch.zeros(sample_shape + self._output_shape).to(
            self.idx_is_initial.device
        )

        n_disk_samples = len(self.idx_is_initial)
        if n_disk_samples > 0:
            assert self.quarter_disk is not None
            sample_disk = self.quarter_disk.sample(
                sample_shape=torch.Size(sample_shape + (n_disk_samples,))
            )
            output[self.idx_is_initial] = sample_disk
        if len(self.idx_not_initial) > 0:
            assert self.quarter_circ is not None
            sample_circ = self.quarter_circ.sample()
            output[self.idx_not_initial] = sample_circ

        return output

    def log_prob(self, sampled_actions: Tensor) -> Tensor:
        """Computes the log probability of the sampled actions.

        Args:
            sampled_actions: Tensor of shape `(*batch_shape, 2)` with the actions to
                compute the log probability of.

        Returns:
            A tensor of shape `(*batch_shape)` containing the log probabilities.
        """
        log_prob = torch.zeros(sampled_actions.shape[:-1]).to(self.idx_is_initial.device)
        n_disk_samples = len(self.idx_is_initial)
        if n_disk_samples > 0:
            assert self.quarter_disk is not None
            log_prob[self.idx_is_initial] = self.quarter_disk.log_prob(
                sampled_actions[self.idx_is_initial]
            )
        if len(self.idx_not_initial) > 0:
            assert self.quarter_circ is not None
            log_prob[self.idx_not_initial] = self.quarter_circ.log_prob(
                sampled_actions[self.idx_not_initial]
            )
        if torch.any(torch.isinf(log_prob)):
            raise ValueError("log_prob contains inf")
        return log_prob


class BoxPFMLP(MLP):
    """A MLP for the forward policy of the Box environment.

    Attributes:
        n_components_s0 (int): The number of components for s0.
        n_components (int): The number of components for non-s0 states.
        PFs0 (nn.Parameter): The parameters for the s0 distribution.
    """

    n_components_s0: int
    n_components: int
    _n_comp_max: int
    _input_dim: int
    PFs0: nn.Parameter

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components_s0: int,
        n_components: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the MLP.

        Args:
            hidden_dim: The hidden dimension.
            n_hidden_layers: The number of hidden layers.
            n_components_s0: The number of components for s0.
            n_components: The number of components for non-s0 states.
            kwargs: Other arguments for the MLP.
        """
        self._n_comp_max = max(n_components_s0, n_components)
        self.n_components_s0 = n_components_s0
        self.n_components = n_components

        input_dim = 2
        self._input_dim = input_dim

        output_dim = 1 + 3 * self.n_components

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )
        # Does not include the + 1 to handle the exit probability (which is
        # impossible at t=0).
        self.PFs0 = nn.Parameter(torch.zeros(1, 5 * self.n_components_s0))

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 1 + 5 * max_n_components)`.
        """
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]

        if preprocessed_states.ndim != 2:
            raise ValueError(
                f"preprocessed_states should be of shape [B, 2], got {preprocessed_states.shape}"
            )
        B, _ = preprocessed_states.shape
        # The desired output shape is [B, 1 + 5 * n_components_max], let's create the tensor
        desired_out = torch.zeros(B, 1 + 5 * self._n_comp_max).to(
            preprocessed_states.device
        )

        # First calculate network outputs for all states
        out = super().forward(
            preprocessed_states
        )  # This should be of shape [B, 1 + 3 * n_components]

        # Now let's find which of the B indices correspond to s_0
        idx_s0 = torch.all(preprocessed_states == 0.0, 1)

        # Now we can fill the desired output tensor
        # 1st, for the s_0 states, we use the PFs0 parameters
        # Remember, PFs0 is of shape [1, 5 * n_components_s0]
        indices_to_override = (
            torch.arange(5 * self._n_comp_max).fmod(self._n_comp_max)
            < self.n_components_s0
        )
        indices_to_override = torch.cat(
            (torch.zeros(1).bool(), indices_to_override), dim=0
        )
        desired_out_slice = desired_out[idx_s0]
        desired_out_slice[:, indices_to_override] = self.PFs0
        desired_out[idx_s0] = desired_out_slice

        # 2nd, for the states s, t>0, we use the network outputs
        # Remember, out is of shape [B, 1 + 3 * n_components]
        indices_to_override2 = (
            torch.arange(3 * self._n_comp_max).fmod(self._n_comp_max) < self.n_components
        )
        indices_to_override2 = torch.cat(
            (
                torch.ones(1).bool(),
                indices_to_override2,
                torch.zeros(2 * self._n_comp_max).bool(),
            ),
            dim=0,
        )
        desired_out_slice2 = desired_out[~idx_s0]
        desired_out_slice2[:, indices_to_override2] = out[~idx_s0]
        desired_out[~idx_s0] = desired_out_slice2

        # Apply sigmoid to all except the dimensions between 1 and 1 + self._n_comp_max
        # These are the components that represent the concentration parameters of the
        # Betas, before normalizing, and should thus be between 0 and 1 (along with
        # the exit probability).
        desired_out[..., 0] = torch.sigmoid(desired_out[..., 0])
        desired_out[..., 1 + self._n_comp_max :] = torch.sigmoid(
            desired_out[..., 1 + self._n_comp_max :]
        )

        assert desired_out.shape == batch_shape + (1 + 5 * self._n_comp_max,)
        return desired_out


class BoxPBMLP(MLP):
    """A MLP for the backward policy of the Box environment.

    Attributes:
        n_components (int): The number of components for each distribution parameter.
    """

    n_components: int
    _input_dim: int

    def __init__(
        self,
        hidden_dim: int,
        n_hidden_layers: int,
        n_components: int,
        **kwargs: Any,
    ) -> None:
        """Initializes the MLP.

        Args:
            hidden_dim: The hidden dimension.
            n_hidden_layers: The number of hidden layers.
            n_components: The number of components for each distribution parameter.
            kwargs: Other arguments for the MLP.
        """
        input_dim = 2
        self._input_dim = input_dim
        output_dim = 3 * n_components

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            output_dim=output_dim,
            activation_fn="elu",
            **kwargs,
        )

        self.n_components = n_components

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 3 * n_components)`.
        """
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]

        out = super().forward(preprocessed_states)

        # Apply sigmoid to all except the dimensions between 0 and self.n_components.
        out[..., self.n_components :] = torch.sigmoid(out[..., self.n_components :])

        assert out.shape == batch_shape + (3 * self.n_components,)
        return out


class BoxStateFlowModule(MLP):
    """A MLP for the state flow function of the Box environment.

    Attributes:
        logZ_value (nn.Parameter): The log partition function value.
    """

    logZ_value: nn.Parameter

    def __init__(self, logZ_value: Tensor, **kwargs: Any) -> None:
        """Initializes the module.

        Args:
            logZ_value: The log partition function value.
            kwargs: Other arguments for the MLP.
        """
        super().__init__(**kwargs)
        self.logZ_value = nn.Parameter(logZ_value)

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, input_dim)`.

        Returns:
            A tensor of shape `(*batch_shape, output_dim)`.
        """
        out = super().forward(preprocessed_states)
        idx_s0 = torch.all(preprocessed_states == 0.0, 1)
        out[idx_s0] = self.logZ_value

        return out


class BoxPBUniform(nn.Module):
    """A uniform backward policy for the Box environment.

    This module returns `(1, 1, 1)` for all states. Used with `QuarterCircle`,
    it leads to a uniform distribution over parents in the south-western part of
    the circle.
    """

    input_dim: int = 2

    def forward(self, preprocessed_states: Tensor) -> Tensor:
        """Computes the forward pass of the neural network.

        Args:
            preprocessed_states: The tensor states of shape `(*batch_shape, 2)`.

        Returns:
            A tensor of shape `(*batch_shape, 3)` filled with ones.
        """
        # return (1, 1, 1) for all states, thus the "+ (3,)".
        assert preprocessed_states.shape[-1] == 2
        batch_shape = preprocessed_states.shape[:-1]
        return torch.ones(batch_shape + (3,), device=preprocessed_states.device)


def split_PF_module_output(
    output: Tensor, n_comp_max: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Splits the module output into the expected parameter sets.

    Args:
        output: the module_output from the P_F model as a tensor of shape
            `(*batch_shape, output_dim)`.
        n_comp_max: the larger number of the two `n_components` and `n_components_s0`.

    Returns:
        A tuple containing:
            - `exit_probability`: A probability unique to `QuarterCircleWithExit`.
            - `mixture_logits`: Parameters shared by `QuarterDisk` and
                `QuarterCircleWithExit`.
            - `alpha_r`: Parameters shared by `QuarterDisk` and `QuarterCircleWithExit`.
            - `beta_r`: Parameters shared by `QuarterDisk` and `QuarterCircleWithExit`.
            - `alpha_theta`: Parameters unique to `QuarterDisk`.
            - `beta_theta`: Parameters unique to `QuarterDisk`.
    """
    (
        exit_probability,  # Unique to QuarterCircleWithExit.
        mixture_logits,  # Shared by QuarterDisk and QuarterCircleWithExit.
        alpha_theta,  # Shared by QuarterDisk and QuarterCircleWithExit.
        beta_theta,  # Shared by QuarterDisk and QuarterCircleWithExit.
        alpha_r,  # Unique to QuarterDisk.
        beta_r,  # Unique to QuarterDisk.
    ) = torch.split(
        output,
        [
            1,  # Unique to QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Shared by QuarterDisk and QuarterCircleWithExit.
            n_comp_max,  # Unique to QuarterDisk.
            n_comp_max,  # Unique to QuarterDisk.
        ],
        dim=-1,
    )

    return (exit_probability, mixture_logits, alpha_theta, beta_theta, alpha_r, beta_r)


class BoxPFEstimator(Estimator):
    r"""Estimator for `P_F` for the Box environment.

    This estimator uses the `DistributionWrapper` distribution.

    Attributes:
        n_components_s0 (int): The number of components for s0.
        n_components (int): The number of components for non-s0 states.
        min_concentration (float): The minimum concentration for the Beta distributions.
        max_concentration (float): The maximum concentration for the Beta distributions.
        delta (float): The radius of the quarter disk.
        epsilon (float): The epsilon value.
    """

    _n_comp_max: int
    n_components_s0: int
    n_components: int
    min_concentration: float
    max_concentration: float
    delta: float
    epsilon: float

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components_s0: int,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 2.0,
    ) -> None:
        """Initializes the estimator.

        Args:
            env: The environment.
            module: The module to use.
            n_components_s0: The number of components for s0.
            n_components: The number of components for non-s0 states.
            min_concentration: The minimum concentration for the Beta distributions.
            max_concentration: The maximum concentration for the Beta distributions.
        """
        super().__init__(module)
        self._n_comp_max = max(n_components_s0, n_components)
        self.n_components_s0 = n_components_s0
        self.n_components = n_components

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.delta = env.delta
        self.epsilon = env.epsilon

    @property
    def expected_output_dim(self) -> int:
        """Returns the expected output dimension of the module."""
        return 1 + 5 * self._n_comp_max

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Converts the module output to a probability distribution.

        Args:
            states: the states for which to convert the module output to a
                probability distribution.
            module_output: the output of the module for the states as a tensor of
                shape `(*batch_shape, output_dim)`.

        Returns:
            The probability distribution for the states.
        """
        # First, we verify that the batch shape of states is 1
        assert len(states.batch_shape) == 1

        # The module_output is of shape (*batch_shape, 1 + 5 * max_n_components), why:
        # We need:
        #   + one scalar for the exit probability,
        #   + self.n_components for the alpha_theta
        #   + self.n_components for the betas_theta
        #   + self.n_components for the mixture logits
        # but we also need compatibility with the s0 state, which has two additional
        # parameters:
        #   + self.n_s0_components for the alpha_r
        #   + self.n_s0_components for the beta_r
        # and finally, we want to be able to give a different number of parameters to
        # s0 and st. So we need to use self._n_comp_max to split on the larger size, and
        # then index to slice out the smaller size when appropriate.
        assert module_output.shape == states.batch_shape + (1 + 5 * self._n_comp_max,)

        (
            exit_probability,
            mixture_logits,
            alpha_theta,
            beta_theta,
            alpha_r,
            beta_r,
        ) = split_PF_module_output(module_output, self._n_comp_max)
        mixture_logits = mixture_logits  # .contiguous().view(-1)

        def _normalize(x: Tensor) -> Tensor:
            return (
                self.min_concentration
                + (self.max_concentration - self.min_concentration) * x
            )  # .contiguous().view(-1)

        alpha_r = _normalize(alpha_r)
        beta_r = _normalize(beta_r)
        alpha_theta = _normalize(alpha_theta)
        beta_theta = _normalize(beta_theta)

        return DistributionWrapper(
            states,
            self.delta,
            self.epsilon,
            mixture_logits,
            alpha_r,
            beta_r,
            alpha_theta,
            beta_theta,
            exit_probability.squeeze(-1),
            self.n_components,
            self.n_components_s0,
        )


class BoxPBEstimator(Estimator):
    r"""Estimator for `P_B` for the Box environment.

    This estimator uses the `QuarterCircle(northeastern=False)` distribution.

    Attributes:
        n_components (int): The number of components for the mixture.
        min_concentration (float): The minimum concentration for the Beta distributions.
        max_concentration (float): The maximum concentration for the Beta distributions.
        delta (float): The radius of the quarter disk.
    """

    n_components: int
    min_concentration: float
    max_concentration: float
    delta: float

    def __init__(
        self,
        env: Box,
        module: nn.Module,
        n_components: int,
        min_concentration: float = 0.1,
        max_concentration: float = 2.0,
    ) -> None:
        """Initializes the estimator.

        Args:
            env: The environment.
            module: The module to use.
            n_components: The number of components for the mixture.
            min_concentration: The minimum concentration for the Beta distributions.
            max_concentration: The maximum concentration for the Beta distributions.
        """
        super().__init__(module, is_backward=True)
        self.module = module
        self.n_components = n_components

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        self.delta = env.delta

    @property
    def expected_output_dim(self) -> int:
        """Returns the expected output dimension of the module."""
        return 3 * self.n_components

    def to_probability_distribution(
        self, states: States, module_output: Tensor
    ) -> Distribution:
        """Converts the module output to a probability distribution.

        Args:
            states: the states for which to convert the module output to a
                probability distribution.
            module_output: the output of the module for the states as a tensor of
                shape `(*batch_shape, output_dim)`.

        Returns:
            The probability distribution for the states.
        """
        # First, we verify that the batch shape of states is 1
        assert len(states.batch_shape) == 1
        mixture_logits, alpha, beta = torch.split(
            module_output, self.n_components, dim=-1
        )

        def _normalize(x: Tensor) -> Tensor:
            return (
                self.min_concentration
                + (self.max_concentration - self.min_concentration) * x
            )  # .contiguous().view(-1)

        if not isinstance(self.module, BoxPBUniform):
            alpha = _normalize(alpha)
            beta = _normalize(beta)
        return QuarterCircle(
            delta=self.delta,
            northeastern=False,
            centers=states,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
        )
