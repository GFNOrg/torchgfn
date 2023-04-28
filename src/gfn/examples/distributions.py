import torch
from torch.distributions import Categorical


class UnsqueezedCategorical(Categorical):
    """This class is used to sample actions from a categorical distribution. The samples
    are unsqueezed to be of shape (batch_size, 1) instead of (batch_size,).

    This is used in `DiscretePFEstimator` and `DiscretePBEstimator`, which in turn are used in
    `ActionsSampler`."""

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)
