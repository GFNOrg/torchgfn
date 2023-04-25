from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from gymnasium.spaces import Discrete, Space
from torchtyping import TensorType

from gfn.actions import Actions
from gfn.casting import correct_cast
from gfn.envs.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.states import DiscreteStates, States

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
OneActionTensor = TensorType["action_shape"]
ActionsTensor = TensorType["batch_shape", "action_shape"]
PmfTensor = TensorType["n_states", torch.float]

# Errors
NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


class Env(ABC):
    """Base class for all environments. Environments require that individual states be represented as a unique tensor of
    arbitrary shape."""

    def __init__(
        self,
        action_space: Space,
        s0: OneStateTensor,
        sf: Optional[OneStateTensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """**Important**: remember that while the action space can be as complex as a Dict or a Tuple, in order to be easily
        understandable by humans, you should still be able to convert them to a unique tensor in order to create
        an `Actions` object.

        Args:
            action_space (Space): Representation of the action space. It could be a Discrete space, a Box space, a composite space
                such as a Tuple or a Dict, or any other space defined in the gymnasium library. Objects of the action_space
                could be thought of as raw actions, human readable.
                In order to be processed by the library, unique actions need to be
                converted to a unique tensor of arbitrary shape. Such a tensor is used to create an `Actions` object.
                The `flatten` method in `gymnasium.spaces.utils` can be used for this purpose.
            s0 (OneStateTensor): Representation of the initial state. All individual states would be of the same shape.
            sf (Optional[OneStateTensor], optional): Representation of the final state. Only used for a human readable representation of
                the states or trajectories.
            device_str (Optional[str], optional): 'cpu' or 'cuda'. Defaults to None, in which case the device is inferred from s0.
            preprocessor (Optional[Preprocessor], optional): a Preprocessor object that converts raw states to a tensor that can be fed
                into a neural network. Defaults to None, in which case the IdentityPreprocessor is used.
        """
        self.s0 = s0
        if sf is None:
            sf = torch.full(s0.shape, -float("inf"))
        self.sf = sf
        self.action_space = action_space
        self.device = torch.device(device_str) if device_str is not None else s0.device
        self.States = self.make_States_class()
        self.Actions = self.make_Actions_class()

        if preprocessor is None:
            preprocessor = IdentityPreprocessor(output_shape=tuple(s0.shape))

        self.preprocessor = preprocessor

    @abstractmethod
    def make_States_class(self) -> type[States]:
        """Returns a class that inherits from States and implements the environment-specific methods."""
        pass

    @abstractmethod
    def make_Actions_class(self) -> type[Actions]:
        """Returns a class that inherits from Actions and implements the environment-specific methods."""
        pass

    def reset(
        self,
        batch_shape: Optional[Union[int, Tuple[int]]] = None,
        random: bool = False,
        sink: bool = False,
        seed: int = None,
    ) -> States:
        """
        Instantiates a batch of initial states. random and sink cannot be both True.
        When random is true and seed is not None, environment randomization is fixed by
        the submitted seed for reproducibility.
        """
        assert not (random and sink)

        if random and seed is not None:
            torch.manual_seed(seed)

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.States.from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )

    @abstractmethod
    def maskless_step(self, states: States, actions: Actions) -> StatesTensor:
        """Function that takes a batch of states and actions and returns a batch of next
        states. Does not need to check whether the actions are valid or the states are sink states.
        """
        pass

    @abstractmethod
    def maskless_backward_step(self, states: States, actions: Actions) -> StatesTensor:
        """Function that takes a batch of states and actions and returns a batch of previous
        states. Does not need to check whether the actions are valid or the states are sink states.
        """
        pass

    @abstractmethod
    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Returns True if the actions are valid in the given states."""
        pass

    def validate_actions(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """First, asserts that states and actions have the same batch_shape.
        Then, uses `is_action_valid`.
        Returns a boolean indicating whether states/actions pairs are valid."""
        assert states.batch_shape == actions.batch_shape
        return self.is_action_valid(states, actions, backward)

    def step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch."""
        new_states = deepcopy(states)
        valid_states_idx: TensorBool = ~states.is_sink_state
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.validate_actions(valid_states, valid_actions):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        new_sink_states_idx = actions.is_exit
        new_states.tensor[new_sink_states_idx] = self.sf
        new_sink_states_idx = ~valid_states_idx | new_sink_states_idx

        not_done_states = new_states[~new_sink_states_idx]
        not_done_actions = actions[~new_sink_states_idx]

        new_not_done_states_tensor = self.maskless_step(
            not_done_states, not_done_actions
        )
        # if isinstance(new_states, DiscreteStates):
        #     new_not_done_states.masks = self.update_masks(not_done_states, not_done_actions)

        new_states.tensor[~new_sink_states_idx] = new_not_done_states_tensor

        if isinstance(new_states, DiscreteStates):  # TODO: move this to DiscreteEnv ?
            new_states.update_masks()  # TODO: probably use `not_done_actions` (and `not_done_states` `~new_sink_states` ?) as input to update_masks
            # TODO: or `locals()`
        return new_states

    def backward_step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = deepcopy(states)
        valid_states_idx: TensorBool = ~new_states.is_initial_state
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.validate_actions(valid_states, valid_actions, backward=True):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        # Calculate the backward step, and update only the states which are not Done.
        new_not_done_states_tensor = self.maskless_backward_step(
            valid_states, valid_actions
        )
        new_states.tensor[valid_states_idx] = new_not_done_states_tensor

        if isinstance(new_states, DiscreteStates):
            new_states.update_masks()

        return new_states

    def reward(self, final_states: States) -> TensorFloat:
        """Either this or log_reward needs to be implemented."""
        return torch.exp(self.log_reward(final_states))

    def log_reward(self, final_states: States) -> TensorFloat:
        """Either this or reward needs to be implemented."""
        raise NotImplementedError("log_reward function not implemented")

    # TODO: some, or all of the following methods should probably move to DiscreteEnv - Basically all DiscreteEnvs should have a Discrete action_space. Do we actually need `action_space` attribute ?

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
            raise NotImplementedError(
                "Only discrete action spaces have a fixed number of actions."
            )

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


class DiscreteEnv(Env, ABC):
    """
    Base class for discrete environments, where actions are represented by a number in
    {0, ..., n_actions - 1}, the last one being the exit action.
    `DiscreteEnv` allow specifying the validity of actions (forward and backward), via mask tensors, that
    are directly attached to `States` objects.
    """

    def make_Actions_class(self) -> type[Actions]:
        env = self
        action_space = self.action_space
        assert isinstance(action_space, Discrete)
        n_actions = self.n_actions

        class DiscreteEnvActions(Actions):
            action_shape = (1,)
            dummy_action = torch.tensor([1.0], device=env.device)  # Double check
            exit_action = torch.tensor([n_actions - 1], device=env.device)

        return DiscreteEnvActions

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        assert states.forward_masks is not None and states.backward_masks is not None
        masks_tensor = states.backward_masks if backward else states.forward_masks
        return torch.gather(masks_tensor, 1, actions.tensor).all()
