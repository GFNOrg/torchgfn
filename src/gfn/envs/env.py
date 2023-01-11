from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from gymnasium.spaces import Discrete, Space
from torchtyping import TensorType

from gfn.containers.states import States, correct_cast
from gfn.envs.preprocessors import IdentityPreprocessor, Preprocessor

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
PmfTensor = TensorType["n_states", torch.float]

NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


class Env(ABC):
    """
    Base class for environments, showing which methods should be implemented.
    A common assumption for all environments is that all actions are discrete,
    represented by a number in {0, ..., n_actions - 1}, the last one being the
    exit action.
    """

    def __init__(
        self,
        action_space: Space,
        s0: OneStateTensor,
        sf: Optional[OneStateTensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.s0 = s0
        if sf is None:
            sf = torch.full(s0.shape, -float("inf"))
        self.sf = sf
        self.action_space = action_space
        self.device = torch.device(device_str) if device_str is not None else s0.device
        self.States = self.make_States_class()

        if preprocessor is None:
            preprocessor = IdentityPreprocessor(output_shape=tuple(s0.shape))

        self.preprocessor = preprocessor

    @abstractmethod
    def make_States_class(self) -> type[States]:
        "Returns a class that inherits from States and implements the environment-specific methods."
        pass

    @abstractmethod
    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        "Returns True if the action is an exit action."
        pass

    @abstractmethod
    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    @abstractmethod
    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the backward_step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    def reward(self, final_states: States) -> TensorFloat:
        """Either this or log_reward needs to be implemented."""
        return torch.exp(self.log_reward(final_states))

    def log_reward(self, final_states: States) -> TensorFloat:
        """Either this or reward needs to be implemented."""
        raise NotImplementedError("log_reward function not implemented")

    def get_states_indices(self, states: States) -> TensorLong:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def get_terminating_states_indices(self, states: States) -> TensorLong:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_actions(self) -> int:
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        else:
            raise NotImplementedError("Only discrete action spaces are supported")

    @property
    def n_states(self) -> int:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_terminating_states(self) -> int:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def true_dist_pmf(self) -> PmfTensor:
        "Returns a one-dimensional tensor representing the true distribution."
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def log_partition(self) -> float:
        "Returns the logarithm of the partition function."
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def all_states(self) -> States:
        """Returns a batch of all states for environments with enumerable states.
        The batch_shape should be (n_states,).
        This should satisfy:
        self.get_states_indices(self.all_states) == torch.arange(self.n_states)
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def terminating_states(self) -> States:
        """Returns a batch of all terminating states for environments with enumerable states.
        The batch_shape should be (n_terminating_states,).
        This should satisfy:
        self.get_terminating_states_indices(self.terminating_states) == torch.arange(self.n_terminating_states)
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def reset(
        self, batch_shape: Union[int, Tuple[int]], random: bool = False
    ) -> States:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States.from_batch_shape(batch_shape=batch_shape, random=random)

    def step(
        self,
        states: States,
        actions: TensorLong,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch."""
        new_states = deepcopy(states)
        valid_states: TensorBool = ~states.is_sink_state
        valid_actions = actions[valid_states]

        if new_states.forward_masks is not None:
            new_forward_masks, _ = correct_cast(
                new_states.forward_masks, new_states.backward_masks
            )
            valid_states_masks = new_forward_masks[valid_states]
            valid_actions_bool = all(
                torch.gather(valid_states_masks, 1, valid_actions.unsqueeze(1))
            )
            if not valid_actions_bool:
                raise NonValidActionsError("Actions are not valid")

        new_sink_states = self.is_exit_actions(actions)
        new_states.states_tensor[new_sink_states] = self.sf
        new_sink_states = ~valid_states | new_sink_states

        not_done_states = new_states.states_tensor[~new_sink_states]
        not_done_actions = actions[~new_sink_states]

        self.maskless_step(not_done_states, not_done_actions)

        new_states.states_tensor[~new_sink_states] = not_done_states

        new_states.update_masks()
        return new_states

    def backward_step(self, states: States, actions: TensorLong) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = deepcopy(states)
        valid_states: TensorBool = ~new_states.is_initial_state
        valid_actions = actions[valid_states]

        if new_states.backward_masks is not None:
            _, new_backward_masks = correct_cast(
                new_states.forward_masks, new_states.backward_masks
            )
            valid_states_masks = new_backward_masks[valid_states]
            valid_actions_bool = all(
                torch.gather(valid_states_masks, 1, valid_actions.unsqueeze(1))
            )
            if not valid_actions_bool:
                raise NonValidActionsError("Actions are not valid")

        not_done_states = new_states.states_tensor[valid_states]
        self.maskless_backward_step(not_done_states, valid_actions)

        new_states.states_tensor[valid_states] = not_done_states

        new_states.update_masks()
        return new_states
