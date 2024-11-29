from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

from gfn.actions import Actions
from gfn.preprocessors import IdentityPreprocessor, Preprocessor
from gfn.states import DiscreteStates, States
from gfn.utils.common import set_seed

# Errors
NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


def get_device(device_str, default_device):
    return torch.device(device_str) if device_str is not None else default_device


class Env(ABC):
    """Base class for all environments. Environments require that individual states be represented as a unique tensor of
    arbitrary shape."""

    def __init__(
        self,
        s0: torch.Tensor,
        state_shape: Tuple,
        action_shape: Tuple,
        dummy_action: torch.Tensor,
        exit_action: torch.Tensor,
        sf: Optional[torch.Tensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """Initializes an environment.

        Args:
            s0: Tensor of shape "state_shape" representing the initial state.
                All individual states would be of the same shape.
            state_shape: Tuple representing the shape of the states.
            action_shape: Tuple representing the shape of the actions.
            dummy_action: Tensor of shape "action_shape" representing a dummy action.
            exit_action: Tensor of shape "action_shape" representing the exit action.
            sf: Tensor of shape "state_shape" representing the final state.
                Only used for a human readable representation of the states or trajectories.
            device_str: 'cpu' or 'cuda'. Defaults to None, in which case the device is
                inferred from s0.
            preprocessor: a Preprocessor object that converts raw states to a tensor
                that can be fed into a neural network. Defaults to None, in which case
                the IdentityPreprocessor is used.
        """
        self.device = get_device(device_str, default_device=s0.device)

        self.s0 = s0.to(self.device)
        assert s0.shape == state_shape
        if sf is None:
            sf = torch.full(s0.shape, -float("inf")).to(self.device)
        self.sf: torch.Tensor = sf
        assert self.sf.shape == state_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dummy_action = dummy_action
        self.exit_action = exit_action

        # Warning: don't use self.States or self.Actions to initialize an instance of the class.
        # Use self.states_from_tensor or self.actions_from_tensor instead.
        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()

        if preprocessor is None:
            assert (
                s0.ndim == 1
            ), "The default preprocessor can only be used for uni-dimensional states."
            output_dim = s0.shape[0]
            preprocessor = IdentityPreprocessor(output_dim=output_dim)

        self.preprocessor = preprocessor
        self.is_discrete = False

    def states_from_tensor(self, tensor: torch.Tensor):
        """Wraps the supplied Tensor in a States instance.

        Args:
            tensor: The tensor of shape "state_shape" representing the states.

        Returns:
            States: An instance of States.
        """
        return self.States(tensor)

    def states_from_batch_shape(
        self, batch_shape: Tuple, random: bool = False, sink: bool = False
    ):
        """Returns a batch of s0 states with a given batch_shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.

        Returns:
            States: A batch of initial states.
        """
        return self.States.from_batch_shape(batch_shape, random=random, sink=sink)

    def actions_from_tensor(self, tensor: torch.Tensor):
        """Wraps the supplied Tensor an an Actions instance.

        Args:
            tensor: The tensor of shape "action_shape" representing the actions.

        Returns:
            Actions: An instance of Actions.
        """
        return self.Actions(tensor)

    def actions_from_batch_shape(self, batch_shape: Tuple):
        """Returns a batch of dummy actions with the supplied batch_shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of actions.

        Returns:
            Actions: A batch of dummy actions.
        """
        return self.Actions.make_dummy_actions(batch_shape)

    # To be implemented by the User.
    @abstractmethod
    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Function that takes a batch of states and actions and returns a batch of next
        states. Does not need to check whether the actions are valid or the states are sink states.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            torch.Tensor: A batch of next states.
        """

    @abstractmethod
    def backward_step(  # TODO: rename to backward_step, other method becomes _backward_step.
        self, states: States, actions: Actions
    ) -> torch.Tensor:
        """Function that takes a batch of states and actions and returns a batch of previous
        states. Does not need to check whether the actions are valid or the states are sink states.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            torch.Tensor: A batch of previous states.
        """

    @abstractmethod
    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Returns True if the actions are valid in the given states."""

    def make_random_states_tensor(self, batch_shape: Tuple) -> torch.Tensor:
        """Optional method inherited by all States instances to emit a random tensor."""
        raise NotImplementedError

    # Optionally implemented by the user when advanced functionality is required.
    def make_states_class(self) -> type[States]:
        """The default States class factory for all Environments.

        Returns a class that inherits from States and implements assumed methods.
        The make_states_class method should be overwritten to achieve more
        environment-specific States functionality.
        """
        env = self

        class DefaultEnvState(States):
            """Defines a States class for this environment."""

            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

        return DefaultEnvState

    def make_actions_class(self) -> type[Actions]:
        """The default Actions class factory for all Environments.

        Returns a class that inherits from Actions and implements assumed methods.
        The make_actions_class method should be overwritten to achieve more
        environment-specific Actions functionality.
        """
        env = self

        class DefaultEnvAction(Actions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DefaultEnvAction

    # In some cases overwritten by the user to support specific use-cases.
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
            set_seed(seed, performance_mode=True)

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        return self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )

    def validate_actions(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """First, asserts that states and actions have the same batch_shape.
        Then, uses `is_action_valid`.
        Returns a boolean indicating whether states/actions pairs are valid."""
        assert states.batch_shape == actions.batch_shape
        return self.is_action_valid(states, actions, backward)

    def _step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Core step function. Calls the user-defined self.step() function.

        Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating sink states in the new batch.
        """
        assert states.batch_shape == actions.batch_shape
        new_states = states.clone()  # TODO: Ensure this is efficient!
        valid_states_idx: torch.Tensor = ~states.is_sink_state
        assert valid_states_idx.shape == states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.validate_actions(valid_states, valid_actions):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        new_sink_states_idx = actions.is_exit
        new_states.tensor[new_sink_states_idx] = self.sf
        new_sink_states_idx = ~valid_states_idx | new_sink_states_idx
        assert new_sink_states_idx.shape == states.batch_shape

        not_done_states = new_states[~new_sink_states_idx]
        not_done_actions = actions[~new_sink_states_idx]

        new_not_done_states_tensor = self.step(not_done_states, not_done_actions)
        if not isinstance(new_not_done_states_tensor, torch.Tensor):
            raise Exception(
                "User implemented env.step function *must* return a torch.Tensor!"
            )

        new_states.tensor[~new_sink_states_idx] = new_not_done_states_tensor

        return new_states

    def _backward_step(
        self,
        states: States,
        actions: Actions,
    ) -> States:
        """Core backward_step function. Calls the user-defined self.backward_step fn.

        This function takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch.
        """
        assert states.batch_shape == actions.batch_shape
        new_states = states.clone()  # TODO: Ensure this is efficient!
        valid_states_idx: torch.Tensor = ~new_states.is_initial_state
        assert valid_states_idx.shape == states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.validate_actions(valid_states, valid_actions, backward=True):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        # Calculate the backward step, and update only the states which are not Done.
        new_not_done_states_tensor = self.backward_step(valid_states, valid_actions)
        new_states.tensor[valid_states_idx] = new_not_done_states_tensor

        if isinstance(new_states, DiscreteStates):
            self.update_masks(new_states)

        return new_states

    def reward(self, final_states: States) -> torch.Tensor:
        """The environment's reward given a state.
        This or log_reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the rewards.
        """
        raise NotImplementedError("Reward function is not implemented.")

    def log_reward(self, final_states: States) -> torch.Tensor:
        """Calculates the log reward.
        This or reward must be implemented.

        Args:
            final_states: A batch of final states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the log rewards.
        """
        return torch.log(self.reward(final_states))

    @property
    def log_partition(self) -> float:
        "Returns the logarithm of the partition function."
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        "Returns a one-dimensional tensor representing the true distribution."
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )


class DiscreteEnv(Env, ABC):
    """
    Base class for discrete environments, where actions are represented by a number in
    {0, ..., n_actions - 1}, the last one being the exit action.

    `DiscreteEnv` allows for  specifying the validity of actions (forward and backward),
    via mask tensors, that are directly attached to `States` objects.
    """

    def __init__(
        self,
        n_actions: int,
        s0: torch.Tensor,
        state_shape: Tuple,
        action_shape: Tuple = (1,),
        dummy_action: Optional[torch.Tensor] = None,
        exit_action: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """Initializes a discrete environment.

        Args:
            n_actions: The number of actions in the environment.
            s0: Tensor of shape "state_shape" representing the initial state (shared among all trajectories).
            state_shape: Tuple representing the shape of the states.
            action_shape: Tuple representing the shape of the actions.
            dummy_action: Optional tensor of shape "action_shape" representing the dummy (padding) action.
            exit_action: Optional tensor of shape "action_shape" representing the exit action.
            sf: Tensor of shape "state_shape" representing the final state tensor (shared among all trajectories).
            device_str: String representation of a torch.device.
            preprocessor: An optional preprocessor for intermediate states.
        """
        device = get_device(device_str, default_device=s0.device)

        # The default dummy action is -1.
        if dummy_action is None:
            dummy_action: torch.Tensor = torch.tensor([-1], device=device)

        # The default exit action index is the final element of the action space.
        if exit_action is None:
            exit_action: torch.Tensor = torch.tensor([n_actions - 1], device=device)

        assert s0.shape == state_shape
        assert dummy_action.shape == action_shape
        assert exit_action.shape == action_shape

        self.n_actions = n_actions  # Before init, for compatibility with States.
        super().__init__(
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
            device_str,
            preprocessor,
        )

        self.is_discrete = True  # After init, else it will be overwritten.

    def states_from_tensor(self, tensor: torch.Tensor):
        """Wraps the supplied Tensor in a States instance & updates masks.

        Args:
            tensor: The tensor of shape "state_shape" representing the states.

        Returns:
            States: An instance of States.
        """
        states_instance = self.make_states_class()(tensor)
        self.update_masks(states_instance)
        return states_instance

    # In some cases overwritten by the user to support specific use-cases.
    def reset(
        self,
        batch_shape: Optional[Union[int, Tuple[int]]] = None,
        random: bool = False,
        sink: bool = False,
        seed: int = None,
    ) -> States:
        """Instantiates a batch of initial states.

        `random` and `sink` cannot be both True. When `random` is `True` and `seed` is
            not `None`, environment randomization is fixed by the submitted seed for
            reproducibility.
        """
        assert not (random and sink)

        if random and seed is not None:
            torch.manual_seed(seed)  # TODO: Improve seeding here?

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        states = self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )
        self.update_masks(states)

        return states

    @abstractmethod
    def update_masks(self, states: States) -> None:
        """Updates the masks in States.

        Called automatically after each step for discrete environments.
        """

    def make_states_class(self) -> type[DiscreteStates]:
        env = self

        class DiscreteEnvStates(DiscreteStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor
            n_actions = env.n_actions
            device = env.device

        return DiscreteEnvStates

    def make_actions_class(self) -> type[Actions]:
        env = self

        class DiscreteEnvActions(Actions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action.to(device=env.device)
            exit_action = env.exit_action.to(device=env.device)

        return DiscreteEnvActions

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        assert states.forward_masks is not None and states.backward_masks is not None
        masks_tensor = states.backward_masks if backward else states.forward_masks
        return torch.gather(masks_tensor, 1, actions.tensor).all()

    def _step(self, states: DiscreteStates, actions: Actions) -> States:
        """Calls the core self._step method of the parent class, and updates masks."""
        new_states = super()._step(states, actions)
        self.update_masks(
            new_states
        )  # TODO: update_masks is owned by the env, not the states!!
        return new_states

    def get_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Returns the indices of the states in the environment.

        Args:
            states: The batch of states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the indices of the states.
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def get_terminating_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Returns the indices of the terminating states in the environment.

        Args:
            states: The batch of states.

        Returns:
            torch.Tensor: Tensor of shape "batch_shape" containing the indices of the terminating states.
        """
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_states(self) -> int:
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_terminating_states(self) -> int:
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        "Returns a tensor of shape (n_states,) representing the true distribution."
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def all_states(self) -> DiscreteStates:
        """Returns a batch of all states.
        The batch_shape should be (n_states,).
        This should satisfy:
        self.get_states_indices(self.all_states) == torch.arange(self.n_states)
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def terminating_states(self) -> DiscreteStates:
        """Returns a batch of all terminating states for environments with enumerable states.
        The batch_shape should be (n_terminating_states,).
        This should satisfy:
        self.get_terminating_states_indices(self.terminating_states) == torch.arange(self.n_terminating_states)
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )
