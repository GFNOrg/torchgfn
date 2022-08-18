from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from torchtyping import TensorType

from gfn.containers.states import States, make_States_class

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]

NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


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
            s_f.shape != s_0.shape or s_f.device != s_0.device  # type: ignore
        ):
            raise ValueError(
                "If s_f is specified, it should be a tensor of shape {} and device {}".format(
                    s_0.shape, s_0.device
                )
            )
        self.n_actions = n_actions
        self.s_0 = s_0
        self.state_shape = tuple(s_0.shape)

        self.device = s_0.device
        self.States: type[States] = make_States_class(
            class_name=self.__class__.__name__ + "States",
            n_actions=n_actions,
            s_0=s_0,
            s_f=s_f,
            make_random_states_tensor=lambda _, batch_shape: self.make_random_states_tensor(
                batch_shape
            ),
            update_masks=lambda states: self.update_masks(states),
        )
        self.s_f = self.States.s_f

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        "Returns True if the action is an exit action."
        return actions == self.n_actions - 1

    def reset(
        self, batch_shape: Union[int, Tuple[int]], random_init: bool = False, **kwargs
    ) -> States:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States(batch_shape=batch_shape, random_init=random_init, **kwargs)

    def step(
        self,
        states: States,
        actions: TensorLong,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch."""
        new_states = deepcopy(states)
        sink_states: TensorBool = new_states.is_sink_state

        non_sink_states_masks = new_states.forward_masks[~sink_states]
        non_sink_actions = actions[~sink_states]
        actions_valid = all(
            torch.gather(non_sink_states_masks, 1, non_sink_actions.unsqueeze(1))
        )
        if not actions_valid:
            raise NonValidActionsError("Actions are not valid")

        new_sink_states = self.is_exit_actions(actions)
        new_states.states[new_sink_states] = self.s_f
        new_sink_states = sink_states | new_sink_states

        not_done_states = new_states.states[~new_sink_states]
        not_done_actions = actions[~new_sink_states]

        self.step_no_worry(not_done_states, not_done_actions)

        new_states.states[~new_sink_states] = not_done_states

        self.update_masks(new_states)
        return new_states

    def backward_step(self, states: States, actions: TensorLong) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = deepcopy(states)
        initial_states: TensorBool = new_states.is_initial_state

        non_initial_states_masks = new_states.backward_masks[~initial_states]
        non_initial_actions = actions[~initial_states]
        actions_valid = all(
            torch.gather(non_initial_states_masks, 1, non_initial_actions.unsqueeze(1))
        )
        if not actions_valid:
            raise NonValidActionsError("Actions are not valid")

        not_done_states = new_states.states[~initial_states]
        self.backward_step_no_worry(not_done_states, non_initial_actions)

        new_states.states[~initial_states] = not_done_states

        self.update_masks(new_states)
        return new_states

    @abstractmethod
    def make_random_states_tensor(self, batch_shape: Tuple[int]) -> StatesTensor:
        pass

    @abstractmethod
    def update_masks(self, states: States) -> None:
        pass

    @abstractmethod
    def step_no_worry(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    @abstractmethod
    def backward_step_no_worry(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the backward_step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    @abstractmethod
    def reward(self, final_states: States) -> TensorFloat:
        pass

    @abstractmethod
    def get_states_indices(self, states: States) -> TensorLong:
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        pass
