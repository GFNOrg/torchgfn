import torch
from torch import Tensor
from torch.distributions import Categorical


class UnsqueezedCategorical(Categorical):
    """Samples froma categorical distribution with an unsqueezed final dimension.

    Samples are unsqueezed to be of shape (batch_size, 1) instead of (batch_size,).

    This is used in `DiscretePFEstimator` and `DiscretePBEstimator`, which in turn are
    used in `Sampler`.

    This helper class facilitates representing actions, for discrete environments, which
    when implemented with the `DiscreteActions` class (see
    `gfn/env.py::DiscreteEnv), use an `action_shape = (1,)`. Therefore, according
    to `gfn/actions.py::Actions`, tensors representing actions in discrete environments
    should be of shape (batch_shape, 1).
    """

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """Sample actions with an unsqueezed final dimension."""
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, sample: Tensor) -> Tensor:
        """Returns the log probabilities of an unsqueezed sample."""
        return super().log_prob(sample.squeeze(-1))
