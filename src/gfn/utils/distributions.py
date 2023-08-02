import torch
from torch.distributions import Categorical
from torchtyping import TensorType as TT


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

    def sample(self, sample_shape=torch.Size()) -> TT["sample_shape", 1]:
        """Sample actions with an unsqueezed final dimension."""
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, sample: TT["sample_shape", 1]) -> TT["sample_shape"]:
        """Returns the log probabilities of an unsqueezed sample."""
        return super().log_prob(sample.squeeze(-1))
