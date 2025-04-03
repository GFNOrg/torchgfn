import torch
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution


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


class GraphActionDistribution(Distribution):
    """A mixture distribution."""

    def __init__(self, logits: TensorDict):
        """Initializes the mixture distribution.

        Args:
            logits: A TensorDict of logits.
        """
        super().__init__()

        assert "action_type" in logits
        assert "edge_class" in logits
        assert "node_class" in logits
        assert "edge_index" in logits

        self._batch_size = logits["action_type"].shape[:-1]
        self.dists = {
            "action_type": Categorical(logits=logits["action_type"]),
            "edge_class": Categorical(logits=logits["edge_class"]),
            "node_class": Categorical(logits=logits["node_class"]),
            "edge_index": Categorical(logits=logits["edge_index"]),
        }
    
    def sample(self, sample_shape=torch.Size()) -> TensorDict:
        """Samples from the distribution.

        Args:
            sample_shape: The shape of the sample.

        Returns the sampled actions as a tensor of shape (*sample_shape, *batch_shape, 1).
        """
        return TensorDict(
            {
                key: dist.sample(sample_shape) for key, dist in self.dists.items()
            },
            batch_size=sample_shape + self._batch_size,
        )

    def log_prob(self, sample: TensorDict) -> torch.Tensor:
        """Returns the log probabilities of a sample.

        Args:
            sample: The sample of for which to compute the log probabilities.
        """
        log_prob = torch.zeros(sample.batch_size)
        for key, dist in self.dists.items():
            log_prob += dist.log_prob(sample[key])
        return log_prob
