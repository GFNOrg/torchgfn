from abc import ABC, abstractmethod
import torch
from torchtyping import TensorType
from torch import Tensor
from typing import Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

from gym.spaces import Discrete


@dataclass
class AbstractStatesBatch(ABC):
    batch_shape: Tuple[int, ...]  # The shape of the batch, usually (n_envs,)
    # The shape of the states, should be (*batch_shape, *state_dim)
    shape: Tuple[int, ...] = field(init=False)
    states: Tensor
    masks: Tensor  # Boolean tensor representing possible actions at each state of the batch
    # Boolean tensor representing possible actions that could have led to each state of the batch
    backward_masks: Tensor
    # Boolean tensor representing if the state is the last of its trajectory
    already_dones: Tensor
    device: torch.device = field(init=False)
    random: bool = False  # If True, then initial state is chosen randomly.

    def __post_init__(self):
        self.device = self.states.device
        self.zero_the_dones()
        self.update_masks()

    @abstractmethod
    def make_masks(self) -> Tensor:
        pass

    @abstractmethod
    def make_backward_masks(self) -> Tensor:
        pass

    def update_masks(self) -> None:
        self.masks = self.make_masks()
        self.backward_masks = self.make_backward_masks()

    def zero_the_dones(self) -> None:
        self.already_dones = torch.zeros(
            self.batch_shape, dtype=torch.bool, device=self.device)

    @abstractmethod
    def update_the_dones(self) -> None:
        # When doing backward sampling, we need to know if the state is the last of its trajectory, i.e.
        # is it s_0.
        pass


@dataclass
class Env(ABC):
    """
    Base class for environments, showing which methods should be implemented.
    A common assumption for all environments is that all actions are discrete,
    represented by a number in {0, ..., n_actions - 1}.
    """
    n_envs: int = 1  # number of environments to run in a vectorized wat
    n_actions: int = field(init=False)  # number of actions
    n_states: int = field(init=False)
    action_space: Discrete = field(init=False)
    state_shape: Tuple = field(init=False)  # shape of the states
    ndim: int = field(init=False)
    state_dim: Tuple = field(init=False)
    StatesBatch: type = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.StatesBatch = self.make_state_class()
        self.state_dim = (self.n_envs, *self.state_shape)
        self.action_space = Discrete(self.n_actions)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def make_state_class(self) -> type:
        """
        :return: a class that represents a state.
        """
        pass

    def reset(self, *kwargs) -> AbstractStatesBatch:
        """
        :return: a batch of states, instance of StatesBatch
        """
        self._state = self.StatesBatch(batch_shape=(self.n_envs, ), *kwargs)
        return deepcopy(self._state)

    @abstractmethod
    def step(self, actions: TensorType[n_envs, torch.long]
             ) -> Tuple[AbstractStatesBatch, TensorType[n_envs, bool]]:
        pass

    @abstractmethod
    def backward_step(self, states: AbstractStatesBatch,
                      actions: TensorType['batch_shape', torch.long]
                      ) -> Tuple[AbstractStatesBatch,
                                 TensorType['batch_shape', bool]]:
        pass

    @abstractmethod
    def reward(self, final_states: Union[AbstractStatesBatch, Tensor]
               ) -> TensorType['batch_shape', float]:
        pass

    @abstractmethod
    def get_states_indices(self, states: Union[AbstractStatesBatch, Tensor]
                           ) -> TensorType['batch_shape', torch.long]:
        pass
