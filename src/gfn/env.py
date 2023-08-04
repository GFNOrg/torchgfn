from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.states import DiscreteStates, States

# Errors
NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


class Env(ABC):
    """Base class for all environments. Environments require that individual states be represented as a unique tensor of
    arbitrary shape."""

    def __init__(
        self,
        s0: TT["state_shape", torch.float],
        sf: Optional[TT["state_shape", torch.float]] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """Initializes an environment.

        Args:
            s0: Representation of the initial state. All individual states would be of the same shape.
            sf (optional): Representation of the final state. Only used for a human readable representation of
                the states or trajectories.
            device_str (Optional[str], optional): 'cpu' or 'cuda'. Defaults to None, in which case the device is inferred from s0.
            preprocessor (Optional[Preprocessor], optional): a Preprocessor object that converts raw states to a tensor that can be fed
                into a neural network. Defaults to None, in which case the IdentityPreprocessor is used.
        """
        self.device = torch.device(device_str) if device_str is not None else s0.device

        self.s0 = s0.to(self.device)
        if sf is None:
            sf = torch.full(s0.shape, -float("inf")).to(self.device)
        self.sf = sf

        self.States = self.make_States_class()
        self.Actions = self.make_Actions_class()

        if preprocessor is None:
            assert (
                s0.ndim == 1
            ), "The default preprocessor can only be used for uni-dimensional states."
            output_dim = s0.shape[0]
            preprocessor = IdentityPreprocessor(output_dim=output_dim)

        self.preprocessor = preprocessor
        self.is_discrete = False

    @abstractmethod
    def make_States_class(self) -> type[States]:
        """Returns a class that inherits from States and implements the environment-specific methods."""

    @abstractmethod
    def make_Actions_class(self) -> type[Actions]:
        """Returns a class that inherits from Actions and implements the environment-specific methods."""

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
    def maskless_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        """Function that takes a batch of states and actions and returns a batch of next
        states. Does not need to check whether the actions are valid or the states are sink states.
        """

    @abstractmethod
    def maskless_backward_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", "state_shape", torch.float]:
        """Function that takes a batch of states and actions and returns a batch of previous
        states. Does not need to check whether the actions are valid or the states are sink states.
        """

    @abstractmethod
    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Returns True if the actions are valid in the given states."""

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
        valid_states_idx: TT["batch_shape", torch.bool] = ~states.is_sink_state
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

        return new_states

    def backward_step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = deepcopy(states)
        valid_states_idx: TT["batch_shape", torch.bool] = ~new_states.is_initial_state
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

    def reward(self, final_states: States) -> TT["batch_shape", torch.float]:
        """Either this or log_reward needs to be implemented."""
        return torch.exp(self.log_reward(final_states))

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.float]:
        """Either this or reward needs to be implemented."""
        raise NotImplementedError("log_reward function not implemented")

    @property
    def log_partition(self) -> float:
        "Returns the logarithm of the partition function."
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

    def __init__(
        self,
        n_actions: int,
        s0: TT["state_shape", torch.float],
        sf: Optional[TT["state_shape", torch.float]] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """Initializes a discrete environment.

        Args:
            n_actions: The number of actions in the environment.

        """
        self.n_actions = n_actions
        super().__init__(s0, sf, device_str, preprocessor)
        self.is_discrete = True

    def make_Actions_class(self) -> type[Actions]:
        env = self
        n_actions = self.n_actions

        class DiscreteEnvActions(Actions):
            action_shape = (1,)
            dummy_action = torch.tensor([-1], device=env.device)  # Double check
            exit_action = torch.tensor([n_actions - 1], device=env.device)

        return DiscreteEnvActions

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        assert states.forward_masks is not None and states.backward_masks is not None
        masks_tensor = states.backward_masks if backward else states.forward_masks
        return torch.gather(masks_tensor, 1, actions.tensor).all()

    def step(
        self,
        states: DiscreteStates,
        actions: Actions,
    ) -> States:
        new_states = super().step(states, actions)
        new_states.update_masks()
        return new_states

    def get_states_indices(
        self, states: DiscreteStates
    ) -> TT["batch_shape", torch.long]:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def get_terminating_states_indices(
        self, states: DiscreteStates
    ) -> TT["batch_shape", torch.long]:
        return NotImplementedError(
            "The environment does not support enumeration of states"
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
    def true_dist_pmf(self) -> TT["n_states", torch.float]:
        "Returns a one-dimensional tensor representing the true distribution."
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def all_states(self) -> DiscreteStates:
        """Returns a batch of all states.
        The batch_shape should be (n_states,).
        This should satisfy:
        self.get_states_indices(self.all_states) == torch.arange(self.n_states)
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def terminating_states(self) -> DiscreteStates:
        """Returns a batch of all terminating states for environments with enumerable states.
        The batch_shape should be (n_terminating_states,).
        This should satisfy:
        self.get_terminating_states_indices(self.terminating_states) == torch.arange(self.n_terminating_states)
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )
