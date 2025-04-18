from abc import ABC, abstractmethod
from typing import Callable

import torch
from einops import rearrange
from torch.nn.functional import one_hot
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

    def __call__(self, states: States | GraphStates) -> torch.Tensor | GeometricBatch:
        """Transform the states to the input of the neural network, calling the preprocess method."""
        out = self.preprocess(states)
        if isinstance(out, torch.Tensor):
            assert out.shape[-1] == self.output_dim

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"


class IdentityPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used, and handles graph and tensor-based states.
    """

    def preprocess(self, states: States | GraphStates) -> torch.Tensor | GeometricBatch:
        """Identity preprocessor. Returns the states as they are."""
        return states.tensor


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


class OneHotPreprocessor(Preprocessor):
    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[DiscreteStates], torch.Tensor],
    ) -> None:
        """One Hot Preprocessor for environments with enumerable states (finite number of states).

        Args:
            n_states (int): The total number of states in the environment (not including s_f).
            get_states_indices (Callable[[States], BatchOutputTensor]): function that returns
                the unique indices of the states.
            BatchOutputTensor is a tensor of shape (*batch_shape, input_dim).
        """
        super().__init__(output_dim=n_states)
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        state_indices = self.get_states_indices(states)

        return one_hot(state_indices, self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    def __init__(
        self,
        height: int,
        ndim: int,
    ) -> None:
        """K Hot Preprocessor for environments with enumerable states (finite number of states) with a grid structure.

        Args:
            height (int): number of unique values per dimension.
            ndim (int): number of dimensions.
        """
        super().__init__(output_dim=height * ndim)
        self.height = height
        self.ndim = ndim

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        states_tensor = states.tensor
        assert (
            states_tensor.dtype == torch.long
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, self.height).float()
        hot = rearrange(hot, "... a b -> ... (a b)")

        return hot
