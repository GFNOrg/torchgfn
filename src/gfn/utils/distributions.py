from typing import Dict
import torch
from torch.distributions import Distribution, Categorical


class UnsqueezedCategorical(Categorical):
    """Samples from a categorical distribution with an unsqueezed final dimension.

    Samples are unsqueezed to be of shape (batch_size, 1) instead of (batch_size,).

    This is used in `DiscretePFEstimator` and `DiscretePBEstimator`, which in turn are
    used in `Sampler`.

    This helper class facilitates representing actions, for discrete environments, which
    when implemented with the `DiscreteActions` class (see
    `gfn/env.py::DiscreteEnv), use an `action_shape = (1,)`. Therefore, according
    to `gfn/actions.py::Actions`, tensors representing actions in discrete environments
    should be of shape (batch_shape, 1).
    """

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Sample actions with an unsqueezed final dimension.

        Args:
            sample_shape: The shape of the sample.

        Returns the sampled actions as a tensor of shape (*sample_shape, *batch_shape, 1).
        """
        out = super().sample(sample_shape).unsqueeze(-1)
        assert out.shape == sample_shape + self._batch_shape + (1,)
        return out

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the log probabilities of an unsqueezed sample.

        Args:
            sample: The sample of for which to compute the log probabilities.

        Returns the log probabilities of the sample as a tensor of shape (*sample_shape, *batch_shape).
        """
        assert sample.shape[-1] == 1
        return super().log_prob(sample.squeeze(-1))


class ComposedDistribution(Distribution):
    """A mixture distribution."""

    def __init__(self, dists: Dict[str, Distribution]):
        """Initializes the mixture distribution.

        Args:
            dists: A dictionary of distributions.
        """
        super().__init__()
        self.dists = dists

    def sample(self, sample_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return {k: v.sample(sample_shape) for k, v in self.dists.items()}