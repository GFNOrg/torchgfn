from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch_geometric.data import Batch as GeometricBatch

from gfn.states import DiscreteStates, GraphStates, States


class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    @abstractmethod
    def preprocess(self, states: States) -> torch.Tensor:
        """Transform the states to the input of the neural network.

        Args:
            states: The states to preprocess.

        Returns the preprocessed states as a tensor of shape (*batch_shape, output_dim).
        """

    def __call__(self, states: States) -> torch.Tensor:
        """Transform the states to the input of the neural network, calling the preprocess method."""
        out = self.preprocess(states)
        assert out.shape[-1] == self.output_dim
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"


class IdentityPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    def preprocess(self, states: States) -> torch.Tensor:
        """Identity preprocessor. Returns the states as they are."""
        return (
            states.tensor.float()
        )  # TODO: should we typecast here? not a true identity...


class EnumPreprocessor(Preprocessor):
    "Preprocessor applicable to environments with discrete states."

    def __init__(
        self,
        get_states_indices: Callable[[DiscreteStates], torch.Tensor],
    ) -> None:
        """Preprocessor for environments with enumerable states (finite number of states).
        Each state is represented by a unique integer (>= 0) index.

        Args:
            get_states_indices: function that returns the unique indices of the states.
                torch.Tensor is a tensor of shape (*batch_shape, 1).
        """
        super().__init__(output_dim=1)
        self.get_states_indices = get_states_indices

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        """Preprocess the states by returning their unique indices.

        Args:
            states: The states to preprocess.

        Returns the unique indices of the states as a tensor of shape `batch_shape`.
        """
        return self.get_states_indices(states).long().unsqueeze(-1)


class GraphPreprocessor(Preprocessor):
    """Preprocessor for graph states to extract the tensor representation.

    This simple preprocessor extracts the GeometricBatch from GraphStates to make
    it compatible with the policy networks. It doesn't perform any complex
    transformations, just ensuring the tensors are accessible in the right format.

    Args:
        feature_dim: The dimension of features in the graph (default: 1)
    """

    def __init__(self, feature_dim: int = 1):
        super().__init__(output_dim=feature_dim)

    def preprocess(self, states: GraphStates) -> GeometricBatch:
        return states.tensor

    def __call__(self, states: GraphStates) -> GeometricBatch:
        return self.preprocess(states)
