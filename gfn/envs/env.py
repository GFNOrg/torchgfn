from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
from gym.spaces import Discrete
from torch import Tensor
from torchtyping import TensorType

from gfn.containers.states import States, StatesMetaClass, make_States_class

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


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
            self.batch_shape, dtype=torch.bool, device=self.device
        )

    @abstractmethod
    def update_the_dones(self) -> None:
        # When doing backward sampling, we need to know if the state is the last of its trajectory, i.e.
        # is it s_0.
        pass


class Env(ABC):
    """
    Base class for environments, showing which methods should be implemented.
    A common assumption for all environments is that all actions are discrete,
    represented by a number in {0, ..., n_actions - 1}.
    """

    def __init__(
        self, n_actions: int, s_0: OneStateTensor, s_f: Optional[OneStateTensor] = None
    ):
        if isinstance(s_f, torch.Tensor) and (
            s_f.shape != s_0.shape or s_f.device != s_0.device
        ):
            raise ValueError(
                "If s_f is specified, it should be a tensor of shape {} and device {}".format(
                    s_0.shape, s_0.device
                )
            )
        self.n_actions = n_actions
        self.s_0 = s_0
        self.state_shape = tuple(s_0.shape)

        self.action_space = Discrete(self.n_actions)
        self.device = s_0.device
        self.States: StatesMetaClass = make_States_class(
            class_name=self.__class__.__name__ + "States",
            n_actions=n_actions,
            s_0=s_0,
            s_f=s_f,
            make_random_states=lambda _, batch_shape: self.make_random_states(
                batch_shape
            ),
            update_masks=self.update_masks,
        )

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        "Returns True if the action is an exit action."
        return actions == self.n_actions - 1

    def reset(self, batch_shape: Union[int, Tuple[int]]) -> StatesTensor:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States(batch_shape=batch_shape)

    @abstractmethod
    def make_random_states(self, batch_shape: Tuple[int]) -> StatesTensor:
        pass

    @abstractmethod
    def update_masks(self, states: States) -> None:
        pass

    @abstractmethod
    def step(
        self,
        states: States,
        actions: TensorLong,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch."""
        pass

    @abstractmethod
    def backward_step(
        self, states: States, actions: TensorLong
    ) -> Tuple[States, TensorBool]:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        pass

    @abstractmethod
    def reward(self, final_states: States) -> TensorFloat:
        pass

    @staticmethod
    @abstractmethod
    def get_states_indices(states: States) -> TensorLong:
        pass
