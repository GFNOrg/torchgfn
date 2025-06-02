import torch
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution

from gfn.actions import GraphActions, GraphActionType


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

    def __init__(self, probs: TensorDict):
        """Initializes the mixture distribution.

        Args:
            probs: A TensorDict of probs.
        """
        super().__init__()

        self.dists = {
            key: Categorical(probs=probs[key])
            for key in GraphActions.ACTION_INDICES.keys()
        }

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Samples from the distribution.

        Args:
            sample_shape: The shape of the sample.

        Returns the sampled actions as a tensor of shape (*sample_shape, *batch_shape, 4).
        """
        action_types = self.dists[GraphActions.ACTION_TYPE_KEY].sample(sample_shape)
        node_classes = torch.zeros_like(action_types)
        edge_classes = torch.zeros_like(action_types)
        edge_indices = torch.zeros_like(action_types)

        add_node_idx = action_types == GraphActionType.ADD_NODE
        if add_node_idx.any():
            node_classes_all = self.dists[GraphActions.NODE_CLASS_KEY].sample(
                sample_shape
            )
            node_classes[add_node_idx] = node_classes_all[add_node_idx]

        add_edge_idx = action_types == GraphActionType.ADD_EDGE
        if add_edge_idx.any():
            edge_classes_all = self.dists[GraphActions.EDGE_CLASS_KEY].sample(
                sample_shape
            )
            edge_classes[add_edge_idx] = edge_classes_all[add_edge_idx]
            edge_indices_all = self.dists[GraphActions.EDGE_INDEX_KEY].sample(
                sample_shape
            )
            edge_indices[add_edge_idx] = edge_indices_all[add_edge_idx]

        components = {
            GraphActions.ACTION_TYPE_KEY: action_types,
            GraphActions.NODE_CLASS_KEY: node_classes,
            GraphActions.EDGE_CLASS_KEY: edge_classes,
            GraphActions.EDGE_INDEX_KEY: edge_indices,
        }

        # Stack components in the order specified by ACTION_INDICES
        return torch.stack(
            [components[key] for key in GraphActions.ACTION_INDICES.keys()], dim=-1
        )

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the log probabilities for a batch of action samples.

        Args:
            sample: A tensor of shape (*sample_shape, *batch_shape, 4) containing action samples, where the last
                dimension is the action type, node class, edge class, and edge index.

        Returns:
            A tensor of shape (*sample_shape, *batch_shape) containing the log probabilities for each sample.
        """
        log_prob = torch.zeros(sample.shape[:-1], device=sample.device)

        # Add log_prob for ACTION_TYPE_KEY
        action_types = sample[
            ..., GraphActions.ACTION_INDICES[GraphActions.ACTION_TYPE_KEY]
        ]
        log_prob += self.dists[GraphActions.ACTION_TYPE_KEY].log_prob(action_types)

        # If action_type is ADD_NODE, add log_prob for NODE_CLASS_KEY
        add_node_idx = action_types == GraphActionType.ADD_NODE
        if add_node_idx.any():
            log_prob_node_class_all = self.dists[GraphActions.NODE_CLASS_KEY].log_prob(
                sample[..., GraphActions.ACTION_INDICES[GraphActions.NODE_CLASS_KEY]]
            )
            log_prob[add_node_idx] += log_prob_node_class_all[add_node_idx]

        # If action_type is ADD_EDGE, add log_prob for EDGE_CLASS_KEY and EDGE_INDEX_KEY
        add_edge_idx = action_types == GraphActionType.ADD_EDGE
        if add_edge_idx.any():
            log_prob_edge_class_all = self.dists[GraphActions.EDGE_CLASS_KEY].log_prob(
                sample[..., GraphActions.ACTION_INDICES[GraphActions.EDGE_CLASS_KEY]]
            )
            log_prob[add_edge_idx] += log_prob_edge_class_all[add_edge_idx]

            log_prob_edge_index_all = self.dists[GraphActions.EDGE_INDEX_KEY].log_prob(
                sample[..., GraphActions.ACTION_INDICES[GraphActions.EDGE_INDEX_KEY]]
            )
            log_prob[add_edge_idx] += log_prob_edge_index_all[add_edge_idx]

        return log_prob
