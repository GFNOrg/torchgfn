from typing import Callable

import torch
from einops import rearrange
from torch.nn.functional import one_hot
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.preprocessors.base import Preprocessor

# Typing
OutputTensor = TensorType["batch_shape", "dim_in"]


class OneHotPreprocessor(Preprocessor):
    name = "one_hot"

    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[States], OutputTensor],
    ) -> None:
        """One Hot Preprocessor for environments with enumerable states (finite number of states).

        Args:
            n_states (int): The total number of states in the environment (not including s_f).
            get_states_indices (Callable[[States], OutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_shape=(n_states,))
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states):
        state_indices = self.get_states_indices(states)
        return one_hot(state_indices, self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    name = "k_hot"

    def __init__(
        self,
        height: int,
        ndim: int,
        get_states_indices: Callable[[States], OutputTensor],
    ) -> None:
        """K Hot Preprocessor for environments with enumerable states (finite number of states) with a grid structure.

        Args:
            height (int): number of unique values per dimension.
            ndim (int): number of dimensions.
            get_states_indices (Callable[[States], OutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_shape=(height * ndim,))
        self.height = height
        self.ndim = ndim
        self.get_states_indices = get_states_indices

    def preprocess(self, states):
        states_tensor = states.states_tensor
        assert (
            states_tensor.dtype == torch.long
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, self.height).float()
        hot = rearrange(hot, "... a b -> ... (a b)")
        return hot
