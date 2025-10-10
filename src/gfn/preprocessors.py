from abc import ABC, abstractmethod
from typing import Callable

import torch
from einops import rearrange
from torch.nn.functional import one_hot

from gfn.states import DiscreteStates, GraphStates, States
from gfn.utils.common import is_int_dtype
from gfn.utils.graphs import GeometricBatch


class Preprocessor(ABC):
    """Base class for state preprocessors.

    Preprocessors transform raw state tensors into formats suitable for neural network
    inputs. They handle the conversion from environment-specific state representations
    to standardized tensor formats that can be processed by neural networks.

    Attributes:
        output_dim: The dimensionality of the preprocessed output tensor, which is
            compatible with the neural network that will be used. If None, the output
            dimension will not be checked.
    """

    def __init__(
        self, output_dim: int | None, target_dtype: torch.dtype | None = None
    ) -> None:
        """Initializes a Preprocessor with the specified output dimension.

        Args:
            output_dim: The dimensionality of the preprocessed output tensor, which is
                compatible with the neural network that will be used.
                If None, the output dimension will not be checked.
            target_dtype: Optional dtype to cast tensor outputs to. When set, any
                tensor returned by `preprocess` will be cast to this dtype in
                `__call__` before returning.
        """
        self.output_dim = output_dim
        self.target_dtype = target_dtype

    @abstractmethod
    def preprocess(self, states: States) -> torch.Tensor:
        """Transforms the states to the input format for neural networks.

        Args:
            states: The states to preprocess.

        Returns:
            A tensor of shape (*batch_shape, output_dim) containing the preprocessed
            states.
        """

    def __call__(self, states: States | GraphStates) -> torch.Tensor | GeometricBatch:
        """Calls the preprocess method and validates the output shape.

        Args:
            states: The states to preprocess.

        Returns:
            The preprocessed states as a tensor or GeometricBatch.
        """
        out = self.preprocess(states)
        if isinstance(out, torch.Tensor):
            if self.output_dim is not None:
                assert out.shape[-1] == self.output_dim
            if self.target_dtype is not None and out.dtype != self.target_dtype:
                out = out.to(self.target_dtype)

        return out

    def __repr__(self):
        """Returns a string representation of the Preprocessor.

        Returns:
            A string summary of the Preprocessor.
        """
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"


class IdentityPreprocessor(Preprocessor):
    """Simple preprocessor that returns states without modification.

    This preprocessor serves as the default preprocessor. It can handle both graph and
    tensor-based states by returning them as-is.

    Attributes:
        output_dim: The dimensionality of the input states.
    """

    def preprocess(self, states: States) -> torch.Tensor | GeometricBatch:
        """Returns the states without any preprocessing.

        Args:
            states: The states to preprocess.

        Returns:
            Tensor or GeometricBatch representing the states.
        """
        return states.tensor


class EnumPreprocessor(Preprocessor):
    """Preprocessor for environments with discrete, enumerable states.

    This preprocessor converts discrete states to their unique integer indices,
    making them suitable for neural network processing. It is designed for
    environments with a finite number of states where each state can be uniquely
    identified by an index.

    Attributes:
        output_dim: Always 1, as states are represented by single indices.
        get_states_indices: Function that returns unique indices for states.
    """

    def __init__(
        self,
        get_states_indices: Callable[[DiscreteStates], torch.Tensor],
    ) -> None:
        """Initializes an EnumPreprocessor.

        Args:
            get_states_indices: Function that returns the unique indices of the states.
                Should return a tensor of shape (*batch_shape, 1).
        """
        super().__init__(output_dim=1)
        self.get_states_indices = get_states_indices

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        """Preprocesses the states by returning their unique indices.

        Args:
            states: The discrete states to preprocess.

        Returns:
            A tensor of shape (*batch_shape, 1) containing the unique indices of the
            states.
        """
        return self.get_states_indices(states).long().unsqueeze(-1)


class OneHotPreprocessor(Preprocessor):
    """Preprocessor that converts discrete states to one-hot encoded vectors.

    This preprocessor is designed for environments with enumerable states where each
    state is represented as a one-hot vector. The output dimension equals the total
    number of possible states.

    Attributes:
        output_dim: The total number of states in the environment.
        get_states_indices: Function that returns unique indices for states.
    """

    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[DiscreteStates], torch.Tensor],
    ) -> None:
        """Initializes a OneHotPreprocessor.

        Args:
            n_states: The total number of states in the environment (not including s_f).
            get_states_indices: Function that returns the unique indices of the states.
                Should return a tensor of shape (*batch_shape, 1).
        """
        super().__init__(output_dim=n_states)
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        """Preprocesses the states by converting them to one-hot encoded vectors.

        Args:
            states: The discrete states to preprocess.

        Returns:
            A tensor of shape (*batch_shape, n_states) containing one-hot encoded states.
        """
        state_indices = self.get_states_indices(states)

        return one_hot(state_indices, self.output_dim).to(torch.get_default_dtype())


class KHotPreprocessor(Preprocessor):
    """Preprocessor for grid-structured discrete states with multi-dimensional encoding.

    This preprocessor is designed for environments with grid-like state spaces where
    each dimension can take on a finite number of values. It creates a k-hot encoding
    where each dimension is one-hot encoded and then concatenated.

    Attributes:
        output_dim: The total output dimension (height * ndim).
        height: Number of unique values per dimension.
        ndim: Number of dimensions in the state space.
    """

    def __init__(
        self,
        height: int,
        ndim: int,
    ) -> None:
        """Initializes a KHotPreprocessor.

        Args:
            height: Number of unique values per dimension.
            ndim: Number of dimensions in the state space.
        """
        super().__init__(output_dim=height * ndim)
        self.height = height
        self.ndim = ndim
        self.output_dim = height * ndim

    def preprocess(self, states: DiscreteStates) -> torch.Tensor:
        """Preprocesses the states by creating k-hot encoded vectors.

        Each dimension of the state is one-hot encoded and then concatenated into
        a single vector.

        Args:
            states: The discrete states to preprocess.

        Returns:
            A tensor of shape (*batch_shape, height * ndim) containing k-hot encoded states.

        Note:
            This preprocessor only works for integer state tensors.
        """
        states_tensor = states.tensor
        assert is_int_dtype(
            states_tensor
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, self.height)
        hot = rearrange(hot, "... a b -> ... (a b)")

        return hot
