from abc import ABC, abstractmethod
from typing import Callable, Tuple

from torchtyping import TensorType

from gfn.containers import States

# Typing
OutputTensor = TensorType["batch_shape", "dim_in"]


class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """

    name: str = "Preprocessor"

    def __init__(self, output_shape: Tuple[int]) -> None:
        self.output_shape = output_shape

    @abstractmethod
    def preprocess(self, states: States) -> OutputTensor:
        pass

    def __call__(self, states: States) -> OutputTensor:
        return self.preprocess(states)

    def __repr__(self):
        return f"{self.name}, output_shape={self.output_shape}"


class IdentityPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    name = "IdentityPreprocessor"

    def preprocess(self, states: States) -> OutputTensor:
        return states.states_tensor.float()


class EnumPreprocessor(Preprocessor):
    "Preprocessor applicable to environments with discrete states."
    name = "EnumPreprocessor"

    def __init__(self, get_states_indices: Callable[[States], OutputTensor]) -> None:
        """Preprocessor for environments with enumerable states (finite number of states).
        Each state is represented by a unique integer (>= 0) index.

        Args:
            get_states_indices (Callable[[States], OutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_shape=(1,))
        self.get_states_indices = get_states_indices

    def preprocess(self, states):
        return self.get_states_indices(states).long().unsqueeze(-1)
