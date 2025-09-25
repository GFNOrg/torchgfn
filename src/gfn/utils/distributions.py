import torch
from tensordict import TensorDict
from torch.distributions import Categorical, Distribution

from gfn.actions import GraphActions, GraphActionType


class UnsqueezedCategorical(Categorical):
    """A `torch.distributions.Categorical` that unsqueezes the last dimension.

    This is useful for discrete environments that have an action shape of (1,), as
    the samples will have a shape of (batch_size, 1) instead of (batch_size,).
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
    """A mixture of categorical distributions for graph actions.

    This class is used to sample graph actions and compute their log probabilities.
    A graph action is a tuple of (action_type, node_class, edge_class, edge_index).
    The distribution of each component of the tuple is a categorical distribution.
    The components are conditionally dependent on the action_type.

    - If the action_type is ADD_NODE, then the node_class is sampled from a
        categorical distribution.
    - If the action_type is ADD_EDGE, then the edge_class and edge_index are
        sampled from categorical distributions.
    - If the action_type is EXIT, then no other components are sampled.
    """

    def __init__(
        self,
        logits: TensorDict | None = None,
        probs: TensorDict | None = None,
        is_backward: bool = False,
    ):
        """Initializes the mixture distribution.

        Args:
            logits: A TensorDict of logits (preferred).
            probs: A TensorDict of probs.
            is_backward: A boolean indicating whether the distribution is for backward policy.
        """
        super().__init__()
        self.is_backward = is_backward
        assert (probs is None) ^ (logits is None), "Pass exactly one of logits or probs."

        # In practice, we never sample from the undefined distributions. However, if we
        # don't disable validation, the inputs are checked at initialization, which can
        # fail when all actions of a particular type are impossible (e.g., unable to
        # add an edge.) TODO: validation should be disabled only when all actions of a
        # particular type are impossible.
        validate_args = False  # edge_index.numel() == 0 when no nodes are present
        self.dists = {
            key: Categorical(
                logits=logits[key] if logits is not None else None,  # type: ignore
                probs=probs[key] if probs is not None else None,  # type: ignore
                validate_args=validate_args,
            )
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
        node_indices = torch.zeros_like(action_types)
        edge_classes = torch.zeros_like(action_types)
        edge_indices = torch.zeros_like(action_types)

        add_node_idx = action_types == GraphActionType.ADD_NODE
        if add_node_idx.any():
            # In backward mode, node_class is irrelevant for ADD_NODE (we remove by index)
            # so we do not sample it and leave zeros. We only sample node_index.
            if self.is_backward:
                node_indices_all = self.dists[GraphActions.NODE_INDEX_KEY].sample(
                    sample_shape
                )
                node_indices[add_node_idx] = node_indices_all[add_node_idx]
            # In forwards mode, node_index is irrelevant for ADD_NODE (we add by class)
            # so we do not sample it and leave zeros. We only sample node_class.
            else:
                node_classes_all = self.dists[GraphActions.NODE_CLASS_KEY].sample(
                    sample_shape
                )
                node_classes[add_node_idx] = node_classes_all[add_node_idx]

        add_edge_idx = action_types == GraphActionType.ADD_EDGE

        # Only sample edge classes and indices if there are any possible edges.
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
            GraphActions.NODE_INDEX_KEY: node_indices,
            GraphActions.EDGE_CLASS_KEY: edge_classes,
            GraphActions.EDGE_INDEX_KEY: edge_indices,
        }

        samples = torch.zeros(
            (*action_types.shape, len(GraphActions.ACTION_INDICES)),
            device=action_types.device,
            dtype=torch.long,
        )
        for key, idx in GraphActions.ACTION_INDICES.items():
            samples[..., idx] = components[key]
        return samples

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Returns the log probabilities for a batch of action samples.

        Note that as we are using hierarchical sampling, the log_prob is the sum of the
        log_probs of the individual components. It is one of:
            - log_prob = p(action_type=add_node) + p(node_class)
            - log_prob = p(action_type=add_edge) + p(edge_class) + p(edge_index)
            - log_prob = p(action_type=remove_node) + p(node_index)
            - log_prob = p(action_type=remove_edge) + p(edge_index)
            - log_prob = p(action_type=exit)

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

        # If action_type is ADD_NODE, add log_prob for NODE_CLASS_KEY or NODE_INDEX_KEY,
        # depending on the mode.
        add_node_idx = action_types == GraphActionType.ADD_NODE
        if add_node_idx.any():
            # For backward mode, ignore node_class contribution; only node_index matters.
            if self.is_backward:
                log_prob_node_index_all = self.dists[
                    GraphActions.NODE_INDEX_KEY
                ].log_prob(
                    sample[..., GraphActions.ACTION_INDICES[GraphActions.NODE_INDEX_KEY]]
                )
                assert torch.isfinite(
                    log_prob_node_index_all[add_node_idx]
                ).all(), (
                    "add_node_idx is indexing masked values in log_prob_node_index_all"
                )
                log_prob[add_node_idx] += log_prob_node_index_all[add_node_idx]

            # In forwards mode, ignore node_index contribution; only node_class matters.
            else:
                log_prob_node_class_all = self.dists[
                    GraphActions.NODE_CLASS_KEY
                ].log_prob(
                    sample[..., GraphActions.ACTION_INDICES[GraphActions.NODE_CLASS_KEY]]
                )
                assert torch.isfinite(
                    log_prob_node_class_all[add_node_idx]
                ).all(), (
                    "add_node_idx is indexing masked values in log_prob_node_class_all"
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
