from typing import OrderedDict

import torch
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution

from gfn.actions import GraphActionType


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

        # Check if no possible edge can be added, 
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_index = torch.isneginf(logits["edge_index"]).all(-1)
        assert torch.isneginf(
            logits["action_type"][no_possible_edge_index, GraphActionType.ADD_EDGE]
        ).all()
        logits["edge_index"][no_possible_edge_index] = 0

        # Check if no possible edge class can be added, 
        # and assert that action type cannot be ADD_EDGE
        no_possible_edge_class = torch.isneginf(logits["edge_class"]).all(-1)
        assert torch.isneginf(
            logits["action_type"][no_possible_edge_class, GraphActionType.ADD_EDGE]
        ).all()
        logits["edge_class"][no_possible_edge_class] = 0

        # Check if no possible node can be added, 
        # and assert that action type cannot be ADD_NODE
        no_possible_node = torch.isneginf(logits["node_class"]).all(-1)
        assert torch.isneginf(
            logits["action_type"][no_possible_node, GraphActionType.ADD_NODE]
        ).all()
        logits["node_class"][no_possible_node] = 0

        self._batch_size = logits["action_type"].shape[:-1]
        self.dists = OrderedDict(
            [
                ("action_type", Categorical(logits=logits["action_type"])),
                ("node_class", Categorical(logits=logits["node_class"])),
                ("edge_class", Categorical(logits=logits["edge_class"])),
                ("edge_index", Categorical(logits=logits["edge_index"])),
            ]
        )

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: The shape of the sample.

        Returns the sampled actions as a tensor of shape (*sample_shape, *batch_shape, 1).
        """
        return torch.cat(
            [dist.sample(sample_shape).unsqueeze(-1) for dist in self.dists.values()],
            dim=-1,
        )

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the log probabilities of a sample.

        Args:
            sample: The sample of for which to compute the log probabilities.
        """
        log_prob = torch.zeros(sample.shape[:-1])
        for i, dist in enumerate(self.dists.values()):
            log_prob += dist.log_prob(sample[..., i])
        return log_prob
