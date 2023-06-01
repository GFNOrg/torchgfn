import torch
from torch.distributions import Categorical
from torchtyping import TensorType as TT


class UnsqueezedCategorical(Categorical):
    """This class is used to sample actions from a categorical distribution. The samples
    are unsqueezed to be of shape (batch_size, 1) instead of (batch_size,).

    This is used in `DiscretePFEstimator` and `DiscretePBEstimator`, which in turn are used in
    `ActionsSampler`.

    "Why do we need this?" one might wonder. The discrete environment implement a `DiscreteActions` class
    (see `gfn/envs/env.py::DiscreteEnv) with an `action_shape = (1,)`. This means, according to
    `gfn/actions.py::Actions`, that tensors representing actions in discrete environments should be of shape
    (batch_shape, 1)"""

    def sample(self, sample_shape=torch.Size()) -> TT["sample_shape", 1]:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, sample: TT["sample_shape", 1]) -> TT["sample_shape"]:
        return super().log_prob(sample.squeeze(-1))
