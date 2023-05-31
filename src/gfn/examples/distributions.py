import numpy as np
import torch
from torch.distributions import Beta, Categorical, Distribution, MixtureSameFamily
from torchtyping import TensorType as TT


class UnsqueezedCategorical(Categorical):
    """This class is used to sample actions from a categorical distribution. The samples
    are unsqueezed to be of shape (batch_size, 1) instead of (batch_size,).

    This is used in `DiscretePFEstimator` and `DiscretePBEstimator`, which in turn are used in
    `ActionsSampler`."""

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)


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
        centers: TT["n_states", 2],
        mixture_logits: TT["n_states", "n_components"],
        alpha: TT["n_states", "n_components"],
        beta: TT["n_states", "n_components"],
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

    def get_min_and_max_angles(self):
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

    def rsample(self, sample_shape=()):
        base_01_samples = self.base_dist.rsample(sample_shape=sample_shape)

        sampled_angles = (
            self.min_angles + (self.max_angles - self.min_angles) * base_01_samples
        )
        sampled_angles = torch.pi / 2 * sampled_angles

        sampled_actions = (
            torch.stack(
                [torch.cos(sampled_angles), torch.sin(sampled_angles)],
                dim=1,
            )
            * self.delta
        )

        if self.northeastern:
            sampled_next_states = self.centers + sampled_actions
        else:
            sampled_next_states = self.centers - sampled_actions
        # TODO: do we need to return actions or next states here? -- logprobs are ok, but the previous 4 lines might not be
        return sampled_next_states

    def log_prob(self, sampled_next_states):
        if self.northeastern:
            sampled_actions = sampled_next_states - self.centers
        else:
            sampled_actions = self.centers - sampled_next_states

        sampled_angles = torch.arccos(sampled_actions[:, 0] / self.delta)

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
        mixture_logits: TT["n_components"],
        alpha_r: TT["n_components"],
        beta_r: TT["n_components"],
        alpha_theta: TT["n_components"],
        beta_theta: TT["n_components"],
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

    def rsample(self, sample_shape=()):
        base_r_01_samples = self.base_r_dist.rsample(sample_shape=sample_shape)
        base_theta_01_samples = self.base_theta_dist.rsample(sample_shape=sample_shape)

        sampled_actions = (
            torch.stack(
                [
                    base_r_01_samples
                    * torch.cos(torch.pi / 2.0 * base_theta_01_samples),
                    base_r_01_samples
                    * torch.sin(torch.pi / 2.0 * base_theta_01_samples),
                ],
                dim=1,
            )
            * self.delta
        )

        return sampled_actions

    def log_prob(self, sampled_actions):
        base_r_01_samples = (
            torch.sqrt(torch.sum(sampled_actions**2, dim=1)) / self.delta
        )
        base_theta_01_samples = torch.arccos(
            sampled_actions[:, 0] / (base_r_01_samples * self.delta)
        ) / (torch.pi / 2.0)

        logprobs = (
            self.base_r_dist.log_prob(base_r_01_samples)
            + self.base_theta_dist.log_prob(base_theta_01_samples)
            - torch.log(self.delta)
            - np.log(np.pi / 2.0)
            - torch.log(base_r_01_samples * self.delta)
        )

        return logprobs


class QuarterCircleWithExit(Distribution):
    """Extends the previous QuarterCircle distribution by considering an extra parameter, called
    `exit_probability` of shape (n_states,). When sampling, then with probability `exit_probability`,
    the `exit_action` [-inf, -inf] is sampled. The `log_prob` function needs to change accordingly
    """

    # TODO
    pass


class BoxForwardDist(Distribution):
    """Mixes the QuarterCircleWithExit(northeastern=True) with QuarterDisk. The parameter `centers`
    controls which distribution is called. When the center is [0, 0], we use QuarterDisk, otherwise,
    we use QuarterCircleWithExit(northeaster=True). Not that `centers` represents a batch of states,
    some of which can be [0, 0]."""

    # TODO
    pass
