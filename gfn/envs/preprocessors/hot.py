from typing import Callable

from einops import rearrange
from torch.nn.functional import one_hot
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.preprocessors.base import Preprocessor

# Typing
OutputTensor = TensorType["batch_shape", "dim_in"]


class OneHotPreprocessor(Preprocessor):
    "Use One Hot Preprocessing for environment with enumerable states"
    name = "one_hot"

    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[States], OutputTensor],
        **kwargs
    ) -> None:
        del kwargs
        super().__init__(output_shape=(n_states,))
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states):
        state_indices = self.get_states_indices(states)
        return one_hot(state_indices, self.output_dim).float()


class KHotPreprocessor(Preprocessor):
    "Use K Hot Preprocessing for environment with enumerable states with a grid structure"
    name = "k_hot"

    def __init__(
        self,
        height: int,
        ndim: int,
        get_states_indices: Callable[[States], OutputTensor],
        **kwargs
    ) -> None:
        del kwargs
        super().__init__(output_shape=(height * ndim,))
        self.height = height
        self.ndim = ndim
        self.get_states_indices = get_states_indices

    def preprocess(self, states):
        states_tensor = states.states
        assert states_tensor.equal(
            states_tensor.floor()
        ), "K Hot preprocessing only works for integer states"
        states_tensor = states_tensor.long()
        hot = one_hot(states_tensor, self.height).float()
        hot = rearrange(hot, "... a b -> ... (a b)")
        return hot
