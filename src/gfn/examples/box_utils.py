"""This file contains utilitary functions for the Box environment."""
from typing import Tuple

import numpy as np
import torch

from torch.distributions import Beta, Categorical, Distribution, MixtureSameFamily
from torchtyping import TensorType as TType

from gfn.estimators import ProbabilityEstimator


class QuarterCircle(Distribution):
    """Represents distributions on quarter circles (or parts thereof), either the northeastern
    ones or the southwestern ones, centered at a point in (0, 1)^2. The distributions
    are Mixture of Beta distributions on the possible angle range.

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment
    """

    def __init__(
        self,
        delta: float,
        northeastern: bool,
        centers: TType["n_states", 2],
        mixture_logits: TType["n_states", "n_components"],
        alpha: TType["n_states", "n_components"],
        beta: TType["n_states", "n_components"],
    ):
        self.delta = delta
        self.northeastern = northeastern
        self.centers = centers
        self.n_states = centers.shape[0]
        self.n_components = mixture_logits.shape[1]

        assert mixture_logits.shape == (self.n_states, self.n_components)
        assert alpha.shape == (self.n_states, self.n_components)
        assert beta.shape == (self.n_states, self.n_components)

        self.base_dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Beta(alpha, beta),
        )

        self.min_angles, self.max_angles = self.get_min_and_max_angles()

    def get_min_and_max_angles(self) -> Tuple[TType["n_states"], TType["n_states"]]:
        if self.northeastern:
            min_angles = torch.where(
                self.centers[:, 0] <= 1 - self.delta,
                0.0,
                2.0 / torch.pi * torch.arccos((1 - self.centers[:, 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers[:, 1] <= 1 - self.delta,
                1.0,
                2.0 / torch.pi * torch.arcsin((1 - self.centers[:, 1]) / self.delta),
            )
        else:
            min_angles = torch.where(
                self.centers[:, 0] >= self.delta,
                0.0,
                2.0 / torch.pi * torch.arccos((self.centers[:, 0]) / self.delta),
            )
            max_angles = torch.where(
                self.centers[:, 1] >= self.delta,
                1.0,
                2.0 / torch.pi * torch.arcsin((self.centers[:, 1]) / self.delta),
            )

        return min_angles, max_angles

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> TType["sample_shape", 2]:
        base_01_samples = self.base_dist.sample(sample_shape=sample_shape)

        sampled_angles = (
            self.min_angles + (self.max_angles - self.min_angles) * base_01_samples
        )
        sampled_angles = torch.pi / 2 * sampled_angles

        sampled_actions = self.delta * torch.stack(
            [torch.cos(sampled_angles), torch.sin(sampled_angles)],
            dim=-1,
        )

        return sampled_actions

    def log_prob(self, sampled_actions: TType["batch_size", 2]) -> TType["batch_size"]:
        sampled_angles = torch.arccos(sampled_actions[..., 0] / self.delta)

        sampled_angles = sampled_angles / (torch.pi / 2)

        base_01_samples = (sampled_angles - self.min_angles) / (
            self.max_angles - self.min_angles
        )

        base_01_logprobs = self.base_dist.log_prob(base_01_samples)

        logprobs = (
            base_01_logprobs
            - np.log(self.delta)
            - np.log(np.pi / 2)
            - torch.log(self.max_angles - self.min_angles)
        )

        return logprobs


class QuarterDisk(Distribution):
    """Represents a distribution on the northeastern quarter disk centered at (0, 0) of maximal radius delta.
    The radius and the angle follow Mixture of Betas distributions.

    Adapted from https://github.com/saleml/continuous-gfn/blob/master/sampling.py

    This is useful for the `Box` environment
    """

    def __init__(
        self,
        delta: float,
        mixture_logits: TType["n_components"],
        alpha_r: TType["n_components"],
        beta_r: TType["n_components"],
        alpha_theta: TType["n_components"],
        beta_theta: TType["n_components"],
    ):
        self.delta = delta
        self.mixture_logits = mixture_logits
        self.n_components = mixture_logits.shape[0]

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

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> TType["sample_shape", 2]:
        base_r_01_samples = self.base_r_dist.sample(sample_shape=sample_shape)
        base_theta_01_samples = self.base_theta_dist.sample(sample_shape=sample_shape)

        sampled_actions = self.delta * (
            torch.stack(
                [
                    base_r_01_samples
                    * torch.cos(torch.pi / 2.0 * base_theta_01_samples),
                    base_r_01_samples
                    * torch.sin(torch.pi / 2.0 * base_theta_01_samples),
                ],
                dim=-1,
            )
        )

        return sampled_actions

    def log_prob(self, sampled_actions: TType["batch_size", 2]) -> TType["batch_size"]:
        base_r_01_samples = (
            torch.sqrt(torch.sum(sampled_actions**2, dim=1)) / self.delta
        )
        base_theta_01_samples = torch.arccos(
            sampled_actions[:, 0] / (base_r_01_samples * self.delta)
        ) / (torch.pi / 2.0)

        logprobs = (
            self.base_r_dist.log_prob(base_r_01_samples)
            + self.base_theta_dist.log_prob(base_theta_01_samples)
            - np.log(self.delta)
            - np.log(np.pi / 2.0)
            - torch.log(base_r_01_samples * self.delta)
        )

        return logprobs


class QuarterCircleWithExit(Distribution):
    """Extends the previous QuarterCircle distribution by considering an extra parameter, called
    `exit_probability` of shape (n_states,). When sampling, then with probability `exit_probability`,
    the `exit_action` [-inf, -inf] is sampled. The `log_prob` function needs to change accordingly
    """

    def __init__(
        self,
        delta: float,
        centers: TType["n_states", 2],
        exit_probability: TType["n_states"],
        mixture_logits: TType["n_states", "n_components"],
        alpha: TType["n_states", "n_components"],
        beta: TType["n_states", "n_components"],
    ):
        self.dist_without_exit = QuarterCircle(
            delta=delta,
            northeastern=True,
            centers=centers,
            mixture_logits=mixture_logits,
            alpha=alpha,
            beta=beta,
        )
        self.exit_probability = exit_probability
        self.exit_action = torch.FloatTensor(
            [-float("inf"), -float("inf")], device=centers.device
        )

        # exit probability should be 1 when torch.norm(1 - states, dim=1) <= env.delta or when torch.any(states >= 1 - env.epsilon, dim=-1)
        # TODO: check that here or elsewhere ?

    def sample(self, sample_shape=()):
        actions = self.dist_without_exit.sample(sample_shape)
        repeated_exit_probability = self.exit_probability.repeat(sample_shape + (1,))
        exit_mask = torch.bernoulli(repeated_exit_probability).bool()
        actions[exit_mask] = self.exit_action

        return actions

    def log_prob(self, sampled_actions):
        logprobs = torch.full_like(self.exit_probability, fill_value=-float("inf"))
        logprobs[~self.exit] = self.dist_without_exit.log_prob(
            sampled_actions[~self.exit]
        )
        logprobs[self.exit] = logprobs[self.exit] + torch.log(1 - self.exit_probability)
        return logprobs


class BoxForwardDist(Distribution):
    """Mixes the QuarterCircleWithExit(northeastern=True) with QuarterDisk. The parameter `centers`
    controls which distribution is called. When the center is [0, 0], we use QuarterDisk, otherwise,
    we use QuarterCircleWithExit(northeaster=True). Not that `centers` represents a batch of states,
    some of which can be [0, 0]."""

    # TODO: do we really need this ? Or should the estimator handle the mapping ?
    pass


class BoxPFEStimator(ProbabilityEstimator):
    r"""Estimator for P_F for the Box environment. Uses the BoxForwardDist distribution"""
    # TODO
    pass


class BoxPBEstimator(ProbabilityEstimator):
    r"""Estimator for P_B for the Box environment. Uses the QuarterCircle(northeastern=False) distribution"""
    # TODO
    pass


if __name__ == "__main__":
    # This code tests the QuarterCircle distribution and makes some plots
    delta = 0.1
    centers = torch.FloatTensor([[0.03, 0.06], [0.2, 0.3], [0.95, 0.7]])
    mixture_logits = torch.FloatTensor([[0.0], [0.0], [0.0]])
    alpha = torch.FloatTensor([[1.0], [1.0], [1.0]])
    beta = torch.FloatTensor([[1.1], [1.0], [1.0]])

    northeastern = True
    dist = QuarterCircle(
        delta=delta,
        northeastern=northeastern,
        centers=centers,
        mixture_logits=mixture_logits,
        alpha=alpha,
        beta=beta,
    )

    n_samples = 10
    samples = dist.sample(sample_shape=(n_samples,))
    print(dist.log_prob(samples))

    # plot the [0, 1] x [0, 1] square, and the centers,
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])

    # plot circles of radius delta around each center and around (0, 0)
    for i in range(centers.shape[0]):
        ax.add_patch(
            plt.Circle(centers[i], delta, fill=False, color="red", linestyle="dashed")
        )
    ax.add_patch(plt.Circle([0, 0], delta, fill=False, color="red", linestyle="dashed"))

    # add each center to its corresponding sampled actions and plot them
    for i in range(centers.shape[0]):
        ax.scatter(
            samples[:, i, 0] + centers[i, 0],
            samples[:, i, 1] + centers[i, 1],
            s=0.2,
            marker="x",
        )
        ax.scatter(centers[i, 0], centers[i, 1], color="red")

    northeastern = False
    dist_backward = QuarterCircle(
        delta=delta,
        northeastern=northeastern,
        centers=centers[1:],
        mixture_logits=mixture_logits[1:],
        alpha=alpha[1:],
        beta=beta[1:],
    )

    samples_backward = dist_backward.sample(sample_shape=(n_samples,))
    print(dist_backward.log_prob(samples_backward))

    # add to the plot a subtraction of the sampled actions from the centers, and plot them
    for i in range(centers[1:].shape[0]):
        ax.scatter(
            centers[1:][i, 0] - samples_backward[:, i, 0],
            centers[1:][i, 1] - samples_backward[:, i, 1],
            s=0.2,
            marker="x",
        )

    quarter_disk_dist = QuarterDisk(
        delta=delta,
        mixture_logits=torch.FloatTensor([0.0]),
        alpha_r=torch.FloatTensor([1.0]),
        beta_r=torch.FloatTensor([1.0]),
        alpha_theta=torch.FloatTensor([1.0]),
        beta_theta=torch.FloatTensor([1.0]),
    )

    samples_disk = quarter_disk_dist.sample(sample_shape=(n_samples * 3,))
    print(quarter_disk_dist.log_prob(samples_disk))

    # add to the plot samples_disk
    ax.scatter(samples_disk[:, 0], samples_disk[:, 1], s=0.1, marker="x")

    # plt.show()

    quarter_circle_with_exit_dist = QuarterCircleWithExit(
        delta=delta,
        centers=centers,
        mixture_logits=mixture_logits,
        alpha=alpha,
        beta=beta,
        exit_probability=torch.FloatTensor([0.5, 0.5, 0.5]),
    )

    samples_exit = quarter_circle_with_exit_dist.sample(sample_shape=(n_samples,))

    print(samples_exit)
