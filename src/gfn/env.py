from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast

import torch
from torch_geometric.data import Data as GeometricData

from gfn.actions import Actions, GraphActions
from gfn.states import DiscreteStates, GraphStates, States
from gfn.utils.common import ensure_same_device, set_seed

# Errors
NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


class Env(ABC):
    """Base class for all environments.

    Environments define the state and action spaces, as well as the forward & backward
    transition and reward functions.

    Attributes:
        s0: The initial state (tensor or GeometricData).
        sf: The sink (final) state (tensor or GeometricData).
        state_shape: Tuple representing the shape of the states.
        action_shape: Tuple representing the shape of the actions.
        dummy_action: Tensor representing the dummy action for padding.
        exit_action: Tensor representing the exit action.
        States: The States class associated with this environment.
        Actions: The Actions class associated with this environment.
        is_discrete: Class variable, whether the environment is discrete.
    """

    is_discrete: bool = False

    def __init__(
        self,
        s0: torch.Tensor | GeometricData,
        state_shape: Tuple,
        action_shape: Tuple,
        dummy_action: torch.Tensor,
        exit_action: torch.Tensor,
        sf: Optional[torch.Tensor | GeometricData] = None,
    ):
        """Initializes an environment.

        Args:
            s0: Tensor of shape (*state_shape) or GeometricData representing the initial
                state.
            state_shape: Tuple representing the shape of the states.
            action_shape: Tuple representing the shape of the actions.
            dummy_action: Tensor of shape (*action_shape) representing the dummy action.
            exit_action: Tensor of shape (*action_shape) representing the exit action.
            sf: (Optional) Tensor of shape (*state_shape) or GeometricData representing
                the sink (final) state.
        """
        if isinstance(s0.device, str):  # This can happen when s0 is a GeometricData.
            s0.device = torch.device(s0.device)
        assert isinstance(s0.device, torch.device)

        self.s0 = s0

        if sf is None:
            sf = torch.full(s0.shape, -float("inf"))
        self.sf = sf.to(
            s0.device  # pyright: ignore / torch_geometric has weird type hints.
        )

        assert self.s0.shape == self.sf.shape == state_shape

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dummy_action = dummy_action.to(s0.device)
        self.exit_action = exit_action.to(s0.device)

        # Warning: don't use self.States or self.Actions to initialize an instance of
        # the class. Use self.states_from_tensor or self.actions_from_tensor instead.
        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()

    @property
    def device(self) -> torch.device:
        """The device on which the environment's elements are stored.

        Returns:
            The device of the initial state.
        """
        return self.s0.device

    def states_from_tensor(self, tensor: torch.Tensor) -> States:
        """Wraps the supplied tensor in a States instance.

        Args:
            tensor: Tensor of shape (*state_shape) representing the states.

        Returns:
            A States instance.
        """
        return self.States(tensor)

    def states_from_batch_shape(
        self, batch_shape: Tuple, random: bool = False, sink: bool = False
    ) -> States:
        """Returns a batch of random, initial, or sink states with a given batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            random: If True, initialize states randomly (requires implementation).
            sink: If True, initialize states as sink states ($s_f$).

        Returns:
            A batch of random, initial, or sink states.
        """
        return self.States.from_batch_shape(
            batch_shape, random=random, sink=sink, device=self.device
        )

    def actions_from_tensor(self, tensor: torch.Tensor) -> Actions:
        """Wraps the supplied tensor in an Actions instance.

        Args:
            tensor: Tensor of shape (*action_shape) representing the actions.

        Returns:
            An Actions instance.
        """
        return self.Actions(tensor)

    def actions_from_batch_shape(self, batch_shape: Tuple) -> Actions:
        """Returns a batch of dummy actions with the supplied batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of actions.

        Returns:
            A batch of dummy actions.
        """
        return self.Actions.make_dummy_actions(batch_shape, device=self.device)

    @abstractmethod
    def step(self, states: States, actions: Actions) -> States:
        """Forward transition function of the environment.

        This method takes a batch of states and actions and returns a batch of next
        states. It does not need to check whether the actions are valid or the states are
        sink states, because the `_step` method wraps it and checks for validity.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            A batch of next states.
        """

    @abstractmethod
    def backward_step(self, states: States, actions: Actions) -> States:
        """Backward transition function of the environment.

        This method takes a batch of states and actions and returns a batch of previous
        states. It does not need to check whether the actions are valid or the states are
        sink states, because the `_backward_step` method wraps it and checks for validity.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            A batch of previous states.
        """

    @abstractmethod
    def is_action_valid(
        self,
        states: States,
        actions: Actions,
        backward: bool = False,
    ) -> bool:
        """Checks whether the actions are valid in the given states.

        Args:
            states: A batch of states.
            actions: A batch of actions.
            backward: If True, checks validity for backward actions.

        Returns:
            True if all actions are valid in the given states, False otherwise.
        """

    def make_random_states(
        self, batch_shape: Tuple, device: torch.device | None = None
    ) -> States:
        """Optional method to return a batch of random states.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            device: The device to create the states on.

        Returns:
            A batch of random states.
        """
        raise NotImplementedError

    def make_states_class(self) -> type[States]:
        """Returns the States class for this environment.

        Defines a custom States class that inherits from States and implements assumed
        methods. The make_states_class method should be overwritten to achieve more
        environment-specific States functionalities.

        Returns:
            A type of a subclass of States with environment-specific functionalities.
        """
        env = self

        class DefaultEnvState(States):
            """Defines a States class for this environment."""

            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states

        return DefaultEnvState

    def make_actions_class(self) -> type[Actions]:
        """Returns the Actions class for this environment.

        Defines a custom Actions class that inherits from Actions and implements assumed
        methods. The make_actions_class method should be overwritten to achieve more
        environment-specific Actions functionalities.

        Returns:
            A type of a subclass of Actions with environment-specific functionalities.
        """
        env = self

        class DefaultEnvAction(Actions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DefaultEnvAction

    def reset(
        self,
        batch_shape: int | Tuple[int, ...] | list[int],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
    ) -> States:
        """Instantiates a batch of random, initial, or sink states.

        Args:
            batch_shape: Shape of the batch (int, tuple, or list).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.

        Returns:
            A batch of initial or sink states.
        """
        assert not (random and sink)

        if random and seed is not None:
            set_seed(seed, performance_mode=True)

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        elif isinstance(batch_shape, list):
            batch_shape = tuple(batch_shape)
        return self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )

    def _step(self, states: States, actions: Actions) -> States:
        """Wrapper for the user-defined `step` function.

        This wrapper ensures that the `step` is called only on valid states and actions,
        and sets the states to the sink state when the action is exit. It also ensures
        that the new states are a distinct object from the old states.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            A batch of next states.
        """
        assert states.batch_shape == actions.batch_shape

        # IMPORTANT: states.clone() is used to ensure that the new states are a
        # distinct object from the old states. This is important for the sampler to
        # work correctly when building the trajectories. If you want to override this
        # method in your custom environment, you must ensure that the `new_states`
        # returned is a distinct object from the submitted states.
        new_states = states.clone()

        valid_states_idx: torch.Tensor = ~states.is_sink_state
        assert valid_states_idx.shape == states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = states[valid_states_idx]

        if not self.is_action_valid(valid_states, valid_actions):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        # Set to the sink state when the action is exit.
        new_sink_states_idx = actions.is_exit
        sf_states = self.States.make_sink_states(
            (int(new_sink_states_idx.sum().item()),), device=states.device
        )
        new_states[new_sink_states_idx] = sf_states
        new_sink_states_idx = ~valid_states_idx | new_sink_states_idx
        assert new_sink_states_idx.shape == states.batch_shape

        not_done_states = new_states[~new_sink_states_idx]
        not_done_actions = actions[~new_sink_states_idx]

        not_done_states = self.step(not_done_states, not_done_actions)
        if not isinstance(not_done_states, States):
            raise ValueError(
                f"The step function must return a States instance, but got {type(not_done_states)} instead."
            )
        new_states[~new_sink_states_idx] = not_done_states
        return new_states

    def _backward_step(self, states: States, actions: Actions) -> States:
        """Wrapper for the user-defined `backward_step` function.

        This wrapper ensures that the `backward_step` is called only on valid states and
        actions, and sets the states to the initial state when the action is not valid.
        It also ensures that the new states are a distinct object from the old states.

        Args:
            states: A batch of states.
            actions: A batch of actions.

        Returns:
            A batch of previous states.
        """
        assert states.batch_shape == actions.batch_shape

        # IMPORTANT: states.clone() is used to ensure that the new states are a
        # distinct object from the old states. This is important for the sampler to
        # work correctly when building the trajectories. If you want to override this
        # method in your custom environment, you must ensure that the `new_states`
        # returned is a distinct object from the submitted states.
        new_states = states.clone()

        valid_states_idx: torch.Tensor = ~new_states.is_initial_state
        assert valid_states_idx.shape == new_states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = new_states[valid_states_idx]

        if not self.is_action_valid(valid_states, valid_actions, backward=True):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        # Calculate the backward step, and update only the states which are not Done.
        new_states[valid_states_idx] = self.backward_step(valid_states, valid_actions)

        return new_states

    def reward(self, states: States) -> torch.Tensor:
        """Returns the environment's rewards for a batch of states.

        This or `log_reward` must be implemented by the environment.

        Args:
            states: A batch of states with a batch_shape.

        Returns:
            Tensor of shape (*batch_shape) containing the rewards.
        """
        raise NotImplementedError("Reward function is not implemented.")

    def log_reward(self, states: States) -> torch.Tensor:
        """Returns the environment's log of rewards for a batch of states.

        This or `reward` must be implemented by the environment.

        Args:
            states: A batch of states with a batch_shape.

        Returns:
            Tensor of shape (*batch_shape) containing the log rewards.
        """
        return torch.log(self.reward(states))

    @property
    def log_partition(self) -> float:
        """Optional method to return the logarithm of the partition function.

        Returns:
            The log partition function.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        """Optional method to return the true distribution.

        Returns:
            The true distribution as a 1-dimensional tensor.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )


class DiscreteEnv(Env, ABC):
    """Base class for discrete environments, where states are defined in a discrete
    space, and actions are represented by an integer in {0, ..., n_actions - 1}, the
    last one being the exit action.

    For a guide on creating your own environments, see the documentation at:
    :doc:`guides/creating_environments`.

    For a complete example, see the HyperGrid environment in
    `src/gfn/gym/hypergrid.py`.

    Attributes:
        s0: Tensor of shape (*state_shape) representing the initial state.
        sf: Tensor of shape (*state_shape) representing the sink (final) state.
        n_actions: The number of actions in the environment.
        state_shape: Tuple representing the shape of the states.
        action_shape: Tuple representing the shape of the actions.
        dummy_action: Tensor of shape (*action_shape) representing the dummy action.
        exit_action: Tensor of shape (*action_shape) representing the exit action.
        States: The States class associated with this environment.
        Actions: The Actions class associated with this environment.
        is_discrete: Class variable, whether the environment is discrete.
    """

    s0: torch.Tensor  # this tells the type checker that s0 is a torch.Tensor
    sf: torch.Tensor  # this tells the type checker that sf is a torch.Tensor
    is_discrete: bool = True

    def __init__(
        self,
        n_actions: int,
        s0: torch.Tensor,
        state_shape: Tuple | int,
        # Advanced parameters (optional):
        action_shape: Tuple | int = (1,),
        dummy_action: Optional[torch.Tensor] = None,
        exit_action: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
    ):
        """Initializes a discrete environment.

        Args:
            n_actions: The number of actions in the environment.
            s0: Tensor of shape (*state_shape) representing the initial state.
            state_shape: Tuple representing the shape of the states.
            action_shape: Tuple representing the shape of the actions.
            dummy_action: (Optional) Tensor of shape (*action_shape) representing the
                dummy (padding) action.
            exit_action: (Optional) Tensor of shape (*action_shape) representing the
                exit action.
            sf: (Optional) Tensor of shape (*state_shape) representing the final state.
        """
        # Add validation/warnings for advanced usage
        if dummy_action is not None or exit_action is not None or sf is not None:
            import warnings

            expert_parameters_used = []
            if dummy_action is not None:
                expert_parameters_used.append("dummy_action")
            if exit_action is not None:
                expert_parameters_used.append("exit_action")
            if sf is not None:
                expert_parameters_used.append("sf")

            warnings.warn(
                "You're using advanced parameters: ({}). "
                "These are only needed for custom action handling. "
                "For basic environments, you can omit these.".format(
                    ", ".join(expert_parameters_used)
                ),
                UserWarning,
            )

        # The default dummy action is -1.
        if dummy_action is None:
            dummy_action = torch.tensor([-1], device=s0.device)

        # The default exit action index is the final element of the action space.
        if exit_action is None:
            exit_action = torch.tensor([n_actions - 1], device=s0.device)

        # If these shapes are integers, convert them to tuples.
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        assert dummy_action is not None
        assert exit_action is not None
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
        )

    def states_from_tensor(self, tensor: torch.Tensor) -> DiscreteStates:
        """Wraps the supplied tensor in a DiscreteStates instance and updates masks.

        Args:
            tensor: Tensor of shape (*state_shape) representing the states.

        Returns:
            An instance of DiscreteStates.
        """
        states_instance = self.make_states_class()(tensor)
        self.update_masks(states_instance)
        return states_instance

    def states_from_batch_shape(
        self, batch_shape: Tuple, random: bool = False, sink: bool = False
    ) -> DiscreteStates:
        r"""Returns a batch of random, initial, or sink states with a given batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).

        Returns:
            DiscreteStates: A batch of random, initial, or sink states.
        """
        out = super().states_from_batch_shape(batch_shape, random, sink)
        assert isinstance(out, DiscreteStates)
        return out

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
    ) -> DiscreteStates:
        """Instantiates a batch of random, initial, or sink states and updates masks.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.

        Returns:
            A batch of initial or sink states.
        """
        states = super().reset(batch_shape, random, sink, seed)
        states = cast(DiscreteStates, states)
        self.update_masks(states)
        return states

    @abstractmethod
    def update_masks(self, states: DiscreteStates) -> None:
        """Updates the masks in DiscreteStates.

        Called automatically after each step for discrete environments.

        Args:
            states: The DiscreteStates object whose masks will be updated.
        """

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns the DiscreteStates class for this environment.

        Returns:
            A type of a subclass of DiscreteStates with environment-specific
            functionalities.
        """
        env = self

        class DiscreteEnvStates(DiscreteStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions
            device = env.device

        return DiscreteEnvStates

    def make_actions_class(self) -> type[Actions]:
        """Returns the Actions class for this environment.

        Returns:
            A type of a subclass of Actions with environment-specific functionalities.
        """
        env = self

        class DiscreteEnvActions(Actions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DiscreteEnvActions

    def is_action_valid(
        self, states: DiscreteStates, actions: Actions, backward: bool = False
    ) -> bool:
        """Checks whether the actions are valid in the given discrete states.

        Args:
            states: The batch of discrete states.
            actions: The batch of actions.
            backward: If True, checks validity for backward actions.

        Returns:
            True if all actions are valid in the given states, False otherwise.
        """
        assert states.forward_masks is not None and states.backward_masks is not None
        masks_tensor = states.backward_masks if backward else states.forward_masks
        return bool(torch.gather(masks_tensor, 1, actions.tensor).all().item())

    def _step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Wrapper for the user-defined `step` function.

        This calls the `_step` method of the parent class and updates masks.

        Args:
            states: The batch of discrete states.
            actions: The batch of actions.

        Returns:
            The batch of next discrete states.
        """
        new_states = super()._step(states, actions)
        new_states = cast(DiscreteStates, new_states)
        self.update_masks(new_states)
        return new_states

    def _backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Wrapper for the user-defined `backward_step` function.

        This calls the `_backward_step` method of the parent class and updates masks.

        Args:
            states: The batch of discrete states.
            actions: The batch of actions.

        Returns:
            The batch of previous discrete states.
        """
        new_states = super()._backward_step(states, actions)
        new_states = cast(DiscreteStates, new_states)
        self.update_masks(new_states)
        return new_states

    def get_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Optional method to return the indices of the states in the environment.

        Args:
            states: The batch of states.

        Returns:
            Tensor of shape (*batch_shape) containing the indices of the states.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def get_terminating_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Optional method to return the indices of the terminating states in the
            environment.

        Args:
            states: The batch of states.

        Returns:
            Tensor of shape (*batch_shape) containing the indices of the terminating
            states.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_states(self) -> int:
        """Optional method to return the number of states in the environment.

        Returns:
            The number of states.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def n_terminating_states(self) -> int:
        """Optional method to return the number of terminating states in the environment.

        Returns:
            The number of terminating states.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def all_states(self) -> DiscreteStates:
        """Optional method to return a batch of all discrete states in the environment.

        Returns:
            A batch of all discrete states (batch_shape = (n_states,)).

        Note:
            self.get_states_indices(self.all_states) and torch.arange(self.n_states)
            should be equivalent.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    @property
    def terminating_states(self) -> DiscreteStates:
        """Optional method to return a batch of all terminating states in the environment.

        Returns:
            A batch of all terminating states (batch_shape = (n_terminating_states,)).

        Note:
            self.get_terminating_states_indices(self.terminating_states) and
            torch.arange(self.n_terminating_states) should be equivalent.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )


class GraphEnv(Env):
    """Base class for graph-based environments.

    Graph environments represent states as graphs (torch_geometric Data objects) and
    actions as graph modifications.

    Attributes:
        s0: GeometricData representing the initial graph state.
        sf: GeometricData representing the sink (final) graph state.
        num_node_classes: Number of node classes.
        num_edge_classes: Number of edge classes.
        is_directed: Whether the graph is directed.
        States: The States class associated with this environment.
        Actions: The Actions class associated with this environment.
    """

    s0: GeometricData  # this tells the type checker that s0 is a GeometricData
    sf: GeometricData  # this tells the type checker that sf is a GeometricData

    def __init__(
        self,
        s0: GeometricData,
        sf: GeometricData,
        num_node_classes: int,
        num_edge_classes: int,
        is_directed: bool,
    ):
        """Initializes a graph-based environment.

        Args:
            s0: GeometricData representing the initial graph state.
            sf: GeometricData representing the sink (final) graph state.
            num_node_classes: Number of node classes.
            num_edge_classes: Number of edge classes.
            is_directed: Whether the graph is directed.
        """
        assert s0.x is not None and sf.x is not None
        assert s0.edge_attr is not None and sf.edge_attr is not None
        assert s0.edge_index is not None and sf.edge_index is not None
        ensure_same_device(s0.x.device, sf.x.device)
        ensure_same_device(s0.edge_attr.device, sf.edge_attr.device)
        ensure_same_device(s0.edge_index.device, sf.edge_index.device)

        self.s0 = s0
        self.sf = sf
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.is_directed = is_directed
        assert s0.x is not None
        assert sf.x is not None
        assert s0.x.shape[-1] == sf.x.shape[-1]

        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()

    @property
    def device(self) -> torch.device:
        """The device on which the graph states are stored.

        Returns:
            The device of the initial graph state's node features.
        """
        assert self.s0.x is not None
        return self.s0.x.device

    def make_states_class(self) -> type[GraphStates]:
        """Returns the GraphStates class for this environment.

        Returns:
            A type of a subclass of GraphStates with environment-specific
            functionalities.
        """
        env = self

        class GraphEnvStates(GraphStates):
            """Graph states for the environment."""

            num_node_classes = env.num_node_classes
            num_edge_classes = env.num_edge_classes
            is_directed = env.is_directed

            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states

        return GraphEnvStates

    def make_actions_class(self) -> type[GraphActions]:
        """Returns the GraphActions class for this environment.

        Returns:
            A type of a subclass of GraphActions with environment-specific
            functionalities.
        """
        return GraphActions

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
    ) -> GraphStates:
        """Instantiates a batch of random, initial, or sink graph states.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.

        Returns:
            A batch of random, initial, or sink graph states.
        """
        states = super().reset(batch_shape, random, sink, seed)
        states = cast(GraphStates, states)
        return states

    @abstractmethod
    def step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        """Forward transition function of the graph environment.

        This method takes a batch of graph states and actions and returns a batch of next
        graph states.

        Args:
            states: A batch of graph states.
            actions: A batch of graph actions.

        Returns:
            A batch of next graph states.
        """

    @abstractmethod
    def backward_step(self, states: GraphStates, actions: GraphActions) -> GraphStates:
        """Backward transition function of the graph environment.

        This method takes a batch of graph states and actions and returns a batch of
        previous graph states.

        Args:
            states: A batch of graph states.
            actions: A batch of graph actions.

        Returns:
            A batch of previous graph states.
        """

    def make_random_states(
        self, batch_shape: int | Tuple, device: torch.device | None = None
    ) -> GraphStates:
        """Optional method to return a batch of random graph states.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            device: The device to create the graph states on.

        Returns:
            A batch of random graph states.
        """
        raise NotImplementedError
