from typing import Callable

import torch
from einops import rearrange
from torch.nn.functional import one_hot
from torchtyping import TensorType as TT

from gfn.preprocessors import Preprocessor
from gfn.states import States


class OneHotPreprocessor(Preprocessor):
    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[States], TT["batch_shape", "input_dim"]],
    ) -> None:
        """One Hot Preprocessor for environments with enumerable states (finite number of states).

        Args:
            n_states (int): The total number of states in the environment (not including s_f).
            get_states_indices (Callable[[States], BatchOutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_dim=n_states)
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states):
        state_indices = self.get_states_indices(states)
        return one_hot(state_indices, self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    def __init__(
        self,
        height: int,
        ndim: int,
        get_states_indices: Callable[[States], TT["batch_shape", "input_dim"]],
    ) -> None:
        """K Hot Preprocessor for environments with enumerable states (finite number of states) with a grid structure.

        Args:
            height (int): number of unique values per dimension.
            ndim (int): number of dimensions.
            get_states_indices (Callable[[States], BatchOutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_dim=height * ndim)
        self.height = height
        self.ndim = ndim
        self.get_states_indices = get_states_indices

    def preprocess(self, states):
        states_tensor = states.tensor
        assert (
            states_tensor.dtype == torch.long
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, self.height).float()
        hot = rearrange(hot, "... a b -> ... (a b)")
        return hot
