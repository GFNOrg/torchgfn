import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from gfn.gflownet import GFlowNet

import numpy as np
import torch
from torch_geometric.data import Data as GeometricData

from gfn.actions import Actions, GraphActions
from gfn.states import DiscreteStates, GraphStates, States
from gfn.utils.common import default_fill_value_for_dtype, ensure_same_device, set_seed

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
        is_conditional: Class variable, whether the environment is conditional.
    """

    is_discrete: bool = False
    is_conditional: bool = False

    def __init__(
        self,
        s0: torch.Tensor | GeometricData,
        state_shape: Tuple,
        action_shape: Tuple,
        dummy_action: torch.Tensor,
        exit_action: torch.Tensor,
        sf: Optional[torch.Tensor | GeometricData] = None,
        debug: bool = False,
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
            debug: If True, States/Actions created by this env will run runtime guards
                (not torch.compile friendly). Keep False in compiled runs.
        """
        if isinstance(s0.device, str):  # This can happen when s0 is a GeometricData.
            s0.device = torch.device(s0.device)
        assert isinstance(s0.device, torch.device)

        self.s0 = s0

        if sf is None:
            assert isinstance(s0, torch.Tensor), "When sf is None, s0 must be a Tensor"
            sf = torch.full(
                s0.shape, default_fill_value_for_dtype(s0.dtype), dtype=s0.dtype
            )
        self.sf = sf.to(s0.device)  # pyright: ignore - torch_geometric type hint fix

        assert self.s0.shape == self.sf.shape == state_shape

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dummy_action = dummy_action.to(s0.device)
        self.exit_action = exit_action.to(s0.device)
        self.debug = debug

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

    def states_from_tensor(
        self, tensor: torch.Tensor, conditions: torch.Tensor | None = None
    ) -> States:
        """Wraps the supplied tensor in a States instance.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing the states.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.

        Returns:
            A States instance.
        """
        return self.States(tensor=tensor, conditions=conditions, debug=self.debug)

    def states_from_batch_shape(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        conditions: torch.Tensor | None = None,
    ) -> States:
        """Returns a batch of random, initial, or sink states with a given batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            random: If True, initialize states randomly (requires implementation).
            sink: If True, initialize states as sink states ($s_f$).
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.

        Returns:
            A batch of random, initial, or sink states.
        """
        return self.States.from_batch_shape(
            batch_shape,
            random=random,
            sink=sink,
            conditions=conditions,
            device=self.device,
            debug=self.debug,
        )

    def actions_from_tensor(self, tensor: torch.Tensor) -> Actions:
        """Wraps the supplied tensor in an Actions instance.

        Args:
            tensor: Tensor of shape (*action_shape) representing the actions.

        Returns:
            An Actions instance.
        """
        return self.Actions(tensor, debug=self.debug)

    def actions_from_batch_shape(self, batch_shape: Tuple) -> Actions:
        """Returns a batch of dummy actions with the supplied batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of actions.

        Returns:
            A batch of dummy actions.
        """
        return self.Actions.make_dummy_actions(
            batch_shape, device=self.device, debug=self.debug
        )

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
        self,
        batch_shape: Tuple,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> States:
        """Optional method to return a batch of random states.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
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
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
    ) -> States:
        """Instantiates a batch of random, initial, or sink states.

        Args:
            batch_shape: Shape of the batch (int, tuple, or list).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.
            conditions: (Optional) Tensor of shape (*batch_shape, condition_dim)
                containing the conditions.

        Returns:
            A batch of initial or sink states.
        """
        assert not (random and sink)

        if random and seed is not None:
            set_seed(seed, deterministic_mode=False)  # TODO: configurable?

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        # If the environment is conditional, we need to sample conditions if not provided.
        if self.is_conditional:
            if conditions is None:
                try:
                    conditions = self.sample_conditions(batch_shape)
                except NotImplementedError as e:
                    raise NotImplementedError(
                        f"Environment {self.__class__.__name__} is conditional, "
                        "but `sample_conditions` method is not implemented."
                    ) from e
            assert conditions.shape[:-1] == batch_shape, (
                f"Conditions batch shape {conditions.shape[:-1]} doesn't match "
                f"expected batch shape {batch_shape}"
            )
            ensure_same_device(conditions.device, self.device)

        return self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink, conditions=conditions
        )

    def sample_conditions(self, batch_shape: int | Tuple[int, ...]) -> torch.Tensor:
        """Sample conditions for the environment. Required for conditional environments.

        Args:
            batch_shape: The shape of the batch of conditions to sample.

        Returns:
            A tensor of shape (*batch_shape, condition_dim) containing the conditions.
        """
        raise NotImplementedError(
            "`sample_conditions` method is not implemented for this environment."
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
        if self.debug:
            # Debug-only guards to avoid graph breaks in compiled runs.
            assert states.batch_shape == actions.batch_shape
            assert (
                len(states.batch_shape) == 1
            ), "Batch shape must be 1 for the step method."

        valid_states_idx: torch.Tensor = ~states.is_sink_state
        if self.debug:
            assert valid_states_idx.shape == states.batch_shape
            assert valid_states_idx.dtype == torch.bool

            # Action validity checks only when debug is enabled to keep compiled hot paths lean.
            valid_actions = actions[valid_states_idx]
            valid_states = states[valid_states_idx]

            if not self.is_action_valid(valid_states, valid_actions):
                raise NonValidActionsError(
                    "Some actions are not valid in the given states. See `is_action_valid`."
                )

        # We only step on states that are not sink states.
        # Note that exit actions directly set the states to the sink state, so they
        # are not included in the valid_states_idx.
        new_valid_states_idx = valid_states_idx & ~actions.is_exit  # boolean mask.

        # IMPORTANT: .clone() ensures new states are a distinct object from the
        # old states. The sampler requires this when building trajectories. If you
        # override this method, you must ensure the returned states are independent.
        not_done_states = states[new_valid_states_idx].clone()
        not_done_actions = actions[new_valid_states_idx]

        not_done_states = self.step(not_done_states, not_done_actions)
        if self.debug:
            assert isinstance(
                not_done_states, States
            ), f"The step function must return a States instance, but got {type(not_done_states)} instead."

        # Create a batch of sink states with the same batch shape as the input states.
        # For the indices where the new states are not sink states (i.e., where the
        # state is not already a sink and the action is not exit), update those
        # positions with the result of the environment's step function.
        new_states = self.States.make_sink_states(
            states.batch_shape, device=states.device
        )
        new_states[new_valid_states_idx] = not_done_states
        # Propagate conditions without re-validation (unchanged from source).
        if states.conditions is not None:
            new_states.conditions = states.conditions
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
        if self.debug:
            assert states.batch_shape == actions.batch_shape

        # IMPORTANT: .clone() ensures new states are a distinct object from the
        # old states. The sampler requires this when building trajectories. If you
        # override this method, you must ensure the returned states are independent.
        new_states = states.clone()

        valid_states_idx: torch.Tensor = ~new_states.is_initial_state
        if self.debug:
            assert valid_states_idx.shape == new_states.batch_shape
            assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = new_states[valid_states_idx]

        if self.debug and not self.is_action_valid(
            valid_states, valid_actions, backward=True
        ):
            raise NonValidActionsError(
                "Some actions are not valid in the given states. See `is_action_valid`."
            )

        # Calculate the backward step, and update only the states which are not Done.
        new_states[valid_states_idx] = self.backward_step(valid_states, valid_actions)
        # Propagate conditions without re-validation (unchanged from source).
        if states.conditions is not None:
            new_states.conditions = states.conditions
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

    def log_partition(self, condition: torch.Tensor | None = None) -> float:
        """Optional method to return the logarithm of the partition function.

        Args:
            condition: Optional tensor of shape (condition_dim,) containing the condition.

        Returns:
            The log partition function.
        """
        raise NotImplementedError(
            "The environment does not support calculating the log partition"
        )

    def true_dist(self, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Optional method to return the true distribution.

        Args:
            condition: Optional tensor of shape (condition_dim,) containing the condition.

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
        dummy_action: Tensor of shape (1,) representing the dummy action.
        exit_action: Tensor of shape (1,) representing the exit action.
        States: The States class associated with this environment.
        Actions: The Actions class associated with this environment.
        is_discrete: Class variable, whether the environment is discrete.
    """

    s0: torch.Tensor
    sf: torch.Tensor
    is_discrete: bool = True

    def __init__(
        self,
        n_actions: int,
        s0: torch.Tensor,
        state_shape: Tuple | int,
        # Advanced parameters (optional):
        dummy_action: Optional[torch.Tensor] = None,
        exit_action: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
        debug: bool = False,
    ):
        """Initializes a discrete environment.

        Args:
            n_actions: The number of actions in the environment.
            s0: Tensor of shape (*state_shape) representing the initial state.
            state_shape: Tuple representing the shape of the states.
            dummy_action: (Optional) Tensor of shape (1,) representing the
                dummy (padding) action.
            exit_action: (Optional) Tensor of shape (1,) representing the
                exit action.
            sf: (Optional) Tensor of shape (*state_shape) representing the final state.
            debug: If True, States created by this env will run runtime guards
                (not torch.compile friendly). Keep False in compiled runs.
        """
        if debug and (
            dummy_action is not None or exit_action is not None or sf is not None
        ):
            import warnings

            params = [
                name
                for name, val in [
                    ("dummy_action", dummy_action),
                    ("exit_action", exit_action),
                    ("sf", sf),
                ]
                if val is not None
            ]
            warnings.warn(
                f"Overriding DiscreteEnv defaults: {', '.join(params)}.",
                UserWarning,
            )

        if dummy_action is None:
            dummy_action = torch.tensor([-1], device=s0.device)

        if exit_action is None:
            exit_action = torch.tensor([n_actions - 1], device=s0.device)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        assert s0.shape == state_shape
        assert dummy_action.shape == (1,)
        assert exit_action.shape == (1,)

        self.n_actions = n_actions  # Before init, for compatibility with States.
        super().__init__(
            s0,
            state_shape,
            (1,),  # action shape is always (1,) for discrete environments
            dummy_action,
            exit_action,
            sf,
            debug=debug,
        )

    def states_from_tensor(
        self, tensor: torch.Tensor, conditions: torch.Tensor | None = None
    ) -> DiscreteStates:
        """Wraps the supplied tensor in a DiscreteStates instance.

        Args:
            tensor: Tensor of shape (*batch_shape, *state_shape) representing the states.
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.

        Returns:
            An instance of DiscreteStates.
        """
        states_instance = cast(
            DiscreteStates,
            self.States(tensor=tensor, conditions=conditions, debug=self.debug),
        )
        return states_instance

    def states_from_batch_shape(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        conditions: torch.Tensor | None = None,
    ) -> DiscreteStates:
        r"""Returns a batch of random, initial, or sink states with a given batch shape.

        Args:
            batch_shape: Tuple representing the shape of the batch of states.
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).

        Returns:
            DiscreteStates: A batch of random, initial, or sink states.
        """
        out = super().states_from_batch_shape(batch_shape, random, sink, conditions)
        assert isinstance(out, DiscreteStates)
        return out

    def reset(
        self,
        batch_shape: int | Tuple[int, ...],
        random: bool = False,
        sink: bool = False,
        seed: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
    ) -> DiscreteStates:
        """Instantiates a batch of random, initial, or sink states.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.
            conditions: (Optional) Tensor of shape (*batch_shape, condition_dim)
                containing the conditions.

        Returns:
            A batch of initial or sink states.
        """
        states = super().reset(batch_shape, random, sink, seed, conditions=conditions)
        states = cast(DiscreteStates, states)
        return states

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
            True if all actions are valid in the given states, False otherwise. When
            `debug` is False, returns True without checking to keep hot paths
            compile-friendly.
        """
        if not self.debug:
            # Skip costly validity checks in production/compiled runs.
            return True

        assert states.forward_masks is not None and states.backward_masks is not None
        masks_tensor = states.backward_masks if backward else states.forward_masks
        return bool(torch.gather(masks_tensor, 1, actions.tensor).all().item())

    def _step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Wrapper for the user-defined `step` function.

        Args:
            states: The batch of discrete states.
            actions: The batch of actions.

        Returns:
            The batch of next discrete states.
        """
        new_states = super()._step(states, actions)
        new_states = cast(DiscreteStates, new_states)
        return new_states

    def _backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Wrapper for the user-defined `backward_step` function.

        Args:
            states: The batch of discrete states.
            actions: The batch of actions.

        Returns:
            The batch of previous discrete states.
        """
        new_states = super()._backward_step(states, actions)
        new_states = cast(DiscreteStates, new_states)
        return new_states

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
        """Jensen-Shannon divergence between two discrete distributions.

        Uses the convention 0 * log(0 / x) = 0 via masking so that zero-
        probability bins contribute nothing (no clamping, no mass distortion).

        Args:
            p: First distribution (1-D, sums to ~1).
            q: Second distribution (1-D, sums to ~1).

        Returns:
            JSD in nats (base-e). Bounded in [0, ln(2)].
        """
        m = 0.5 * (p + q)

        # KL(p || m): only sum over bins where p > 0 (m > 0 is guaranteed there).
        p_pos = p > 0
        kl_pm = (p[p_pos] * (p[p_pos] / m[p_pos]).log()).sum()

        # KL(q || m): only sum over bins where q > 0.
        q_pos = q > 0
        kl_qm = (q[q_pos] * (q[q_pos] / m[q_pos]).log()).sum()

        return (0.5 * (kl_pm + kl_qm)).item()

    def get_terminating_state_dist(self, states: DiscreteStates) -> torch.Tensor:
        """Computes the empirical distribution over terminating states.

        Uses vectorized ``scatter_add_`` for efficient histogram computation.

        Args:
            states: A batch of terminating ``DiscreteStates``.

        Returns:
            A 1D CPU tensor of shape ``(n_terminating_states,)`` with empirical
            frequencies summing to 1.

        Raises:
            NotImplementedError: If the environment lacks
                ``get_terminating_states_indices`` or ``n_terminating_states``.
            ValueError: If *states* is empty, or if the environment's state
                space is too large to histogram (``get_terminating_states_indices``
                returned something other than a ``torch.Tensor``).
        """
        indices = self.get_terminating_states_indices(states)
        n_bins = self.n_terminating_states

        # Histogram on CPU to avoid allocating a potentially large (n_bins)
        # tensor on GPU just to immediately .cpu() the result.
        if not isinstance(indices, torch.Tensor):
            # The environment signalled that its canonical index space is too
            # large to represent as a dense int64 tensor (e.g. HyperGrid's
            # numpy-object bigint fallback).  A length-``n_bins`` histogram is
            # fundamentally infeasible in that regime — ``n_bins`` itself does
            # not fit in machine integers — so we cannot build an empirical
            # distribution.  Raise ``ValueError`` (not ``NotImplementedError``)
            # so callers like ``Env.validate()`` do not silently swallow this
            # and replace it with a generic "environment doesn't implement
            # get_terminating_states_indices" message.
            raise ValueError(
                "Cannot compute an empirical terminating-state distribution: "
                "this environment's state space is too large to histogram. "
                "Use sample-based estimators instead."
            )
        flat_indices = indices.reshape(-1).long().cpu()
        n_samples = flat_indices.shape[0]

        if n_samples == 0:
            raise ValueError(
                "No terminating states provided to compute empirical distribution."
            )

        counts = torch.zeros(n_bins, dtype=torch.get_default_dtype())
        ones = torch.ones_like(flat_indices, dtype=torch.get_default_dtype())
        counts.scatter_add_(0, flat_indices, ones)

        return counts / n_samples

    def _warn_if_insufficient_samples(self, n_validation_samples: int) -> None:
        """Emit a warning if validation sample count is too low for the state space."""
        try:
            n_ts = self.n_terminating_states
        except NotImplementedError:
            return  # n_terminating_states not available; skip check.

        samples_per_state = n_validation_samples / n_ts
        mode_info = ""
        n_ms = getattr(self, "n_mode_states", None)
        if callable(n_ms):
            n_ms = n_ms  # property already resolved
        if isinstance(n_ms, (int, float)) and n_ms > 0:
            sparsity = n_ms / n_ts
            mode_info = f" Mode sparsity: {n_ms}/{n_ts} = {sparsity:.4f}."
        if samples_per_state < 1:
            warnings.warn(
                f"n_validation_samples={n_validation_samples} is less "
                f"than n_terminating_states={n_ts} "
                f"({samples_per_state:.2f} samples/state). Most states "
                f"will have zero counts — L1 and JSD will be unreliable."
                f"{mode_info} Recommend at least "
                f"{10 * n_ts} samples.",
                UserWarning,
                stacklevel=3,
            )
        elif samples_per_state < 10:
            warnings.warn(
                f"n_validation_samples={n_validation_samples} for "
                f"n_terminating_states={n_ts} "
                f"({samples_per_state:.1f} samples/state) may produce "
                f"noisy validation metrics.{mode_info} Recommend at "
                f"least {10 * n_ts} samples.",
                UserWarning,
                stacklevel=3,
            )

    def validate(
        self,
        gflownet: "GFlowNet",
        n_validation_samples: int = 1000,
        visited_terminating_states: Optional[DiscreteStates] = None,
        validate_condition: torch.Tensor | None = None,
        sampling_chunk_size: int = 5000,
        check_sample_sufficiency: bool = True,
    ) -> Tuple[Dict[str, float], DiscreteStates | None]:
        """Evaluate a GFlowNet against this environment's true distribution.

        Always samples fresh from the current policy to produce an unbiased
        estimate. Computes L1 distance and Jensen-Shannon divergence between
        the empirical and true distributions. If the GFlowNet has a learned
        ``logZ`` and the environment implements ``log_partition``, also reports
        the absolute difference.

        Args:
            gflownet: The GFlowNet to evaluate.
            n_validation_samples: Number of fresh trajectories to sample.
            visited_terminating_states: **Deprecated.** Ignored if passed; a
                ``DeprecationWarning`` is emitted.
            validate_condition: Optional condition tensor for conditional envs.
            sampling_chunk_size: Max trajectories to sample at once (avoids OOM).
            check_sample_sufficiency: If True, emits a one-time warning when
                ``n_validation_samples`` is too small relative to the state
                space. Set False to suppress.

        Returns:
            ``(metrics_dict, sampled_terminating_states)`` where
            *metrics_dict* contains ``"l1_dist"``, ``"jsd"``, and optionally
            ``"logZ_diff"``.

        Raises:
            ValueError: If ``true_dist`` is unavailable, ``n_validation_samples``
                is non-positive, or enumeration APIs are missing.
        """
        # --- Deprecation warning for visited_terminating_states ---
        if visited_terminating_states is not None:
            warnings.warn(
                "The `visited_terminating_states` parameter is deprecated and "
                "ignored. validate() now always samples fresh from the policy.",
                DeprecationWarning,
                stacklevel=2,
            )

        # --- Validate preconditions ---
        if n_validation_samples <= 0:
            raise ValueError(
                f"n_validation_samples must be > 0, got {n_validation_samples}"
            )

        # --- Sample sufficiency check (once per env instance) ---
        if check_sample_sufficiency and not getattr(
            self, "_sample_sufficiency_checked", False
        ):
            self._sample_sufficiency_checked = True  # type: ignore[attr-defined]
            self._warn_if_insufficient_samples(n_validation_samples)

        # --- Get true distribution ---
        try:
            true_dist = self.true_dist(validate_condition)
        except (NotImplementedError, AssertionError):
            raise ValueError(
                "Environment does not implement true_dist(); cannot validate. "
                "Ensure store_all_states=True or implement true_dist()."
            ) from None

        if not isinstance(true_dist, torch.Tensor):
            raise ValueError(
                f"true_dist() returned {type(true_dist)}, expected torch.Tensor."
            )
        true_dist = true_dist.cpu()

        # --- Fresh sampling with chunking ---
        all_chunks: list[DiscreteStates] = []
        remaining = n_validation_samples
        while remaining > 0:
            chunk_n = min(remaining, sampling_chunk_size)
            chunk_states = gflownet.sample_terminating_states(self, chunk_n)
            if not isinstance(chunk_states, DiscreteStates):
                raise ValueError(
                    f"sample_terminating_states returned {type(chunk_states)}, "
                    f"expected DiscreteStates."
                )
            all_chunks.append(chunk_states)
            remaining -= chunk_n

        sampled_terminating_states = all_chunks[0]
        for chunk in all_chunks[1:]:
            sampled_terminating_states.extend(chunk)

        # --- Compute empirical distribution ---
        try:
            empirical_dist = self.get_terminating_state_dist(sampled_terminating_states)
        except NotImplementedError:
            raise ValueError(
                "Environment does not implement get_terminating_states_indices() "
                "or n_terminating_states; cannot compute empirical distribution."
            ) from None

        # --- Shape validation ---
        if empirical_dist.shape != true_dist.shape:
            raise ValueError(
                f"Shape mismatch: empirical_dist {empirical_dist.shape} vs "
                f"true_dist {true_dist.shape}."
            )

        # --- Compute metrics ---
        l1_dist = (empirical_dist - true_dist).abs().sum().item()
        jsd = self._jsd(empirical_dist, true_dist)
        validation_info: Dict[str, float] = {"l1_dist": l1_dist, "jsd": jsd}

        # --- logZ comparison (best-effort) ---
        self._add_logz_diff(validation_info, gflownet, validate_condition)

        return validation_info, sampled_terminating_states

    def _add_logz_diff(
        self,
        validation_info: Dict[str, float],
        gflownet: "GFlowNet",
        validate_condition: torch.Tensor | None,
    ) -> None:
        """Compute |learned_logZ - true_logZ| and add to validation_info."""
        true_logZ: float | None = None
        try:
            true_logZ = self.log_partition(validate_condition)
        except NotImplementedError:
            return

        learned_logZ: float | None = None
        if hasattr(gflownet, "logZ"):
            if isinstance(gflownet.logZ, torch.Tensor):
                learned_logZ = float(gflownet.logZ.item())
            else:
                from gfn.estimators import ScalarEstimator

                if isinstance(gflownet.logZ, ScalarEstimator):
                    if validate_condition is not None:
                        learned_logZ = gflownet.logZ(validate_condition).item()
                    else:
                        warnings.warn(
                            "gflownet.logZ is a ScalarEstimator but no "
                            "validate_condition was provided; skipping logZ "
                            "comparison.",
                            UserWarning,
                            stacklevel=3,
                        )

        if learned_logZ is not None and true_logZ is not None:
            validation_info["logZ_diff"] = abs(learned_logZ - true_logZ)

    def get_states_indices(
        self, states: DiscreteStates
    ) -> Union[torch.Tensor, np.ndarray]:
        """Optional method to return the indices of the states in the environment.

        Most implementations return a ``torch.Tensor`` of shape ``(*batch_shape,)``
        with dtype ``torch.int64``.  Implementations whose canonical index space
        exceeds int64 (e.g. :class:`gfn.gym.HyperGrid` with ``height ** ndim > 2 ** 63``)
        may instead return a ``numpy.ndarray`` of dtype ``object`` containing
        arbitrary-precision Python ints — in that regime an int64 tensor would
        silently overflow and produce hash collisions between distinct states.

        Args:
            states: The batch of states.

        Returns:
            Tensor or numpy object array of shape ``(*batch_shape,)`` containing
            the canonical indices of the states.
        """
        raise NotImplementedError(
            "The environment does not support enumeration of states"
        )

    def get_terminating_states_indices(
        self, states: DiscreteStates
    ) -> Union[torch.Tensor, np.ndarray]:
        """Optional method to return the indices of the terminating states in the
            environment.

        See :meth:`get_states_indices` for the return-type contract.

        Args:
            states: The batch of states.

        Returns:
            Tensor or numpy object array of shape ``(*batch_shape,)`` containing
            the canonical indices of the terminating states.
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
        debug: bool = False,
    ):
        """Initializes a graph-based environment.

        Args:
            s0: GeometricData representing the initial graph state.
            sf: GeometricData representing the sink (final) graph state.
            num_node_classes: Number of node classes.
            num_edge_classes: Number of edge classes.
            is_directed: Whether the graph is directed.
            debug: Kept for consistency with the other environments. Currently does not
                optimize runtime.
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
        self.debug = debug

        assert s0.x.shape[-1] == sf.x.shape[-1]

        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()
        self.dummy_action = self.Actions.make_dummy_actions(
            (1,), device=self.device
        ).tensor
        self.exit_action = self.Actions.make_exit_actions(
            (1,), device=self.device
        ).tensor

    @property
    def device(self) -> torch.device:
        """The device on which the graph states are stored.

        Returns:
            The device of the initial graph state's node features.
        """
        if self.debug:
            assert self.s0.x is not None
        return self.s0.x.device  # type: ignore[union-attr]

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
        conditions: Optional[torch.Tensor] = None,
    ) -> GraphStates:
        """Instantiates a batch of random, initial, or sink graph states.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            random: If True, initialize states randomly.
            sink: If True, initialize states as sink states ($s_f$).
            seed: (Optional) Random seed for reproducibility.
            conditions: (Optional) Tensor of shape (*batch_shape, condition_dim)
                containing the conditions.

        Returns:
            A batch of random, initial, or sink graph states.
        """
        states = super().reset(batch_shape, random, sink, seed, conditions=conditions)
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
        self,
        batch_shape: int | Tuple,
        conditions: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> GraphStates:
        """Optional method to return a batch of random graph states.

        Args:
            batch_shape: Shape of the batch (int or tuple).
            conditions: Optional tensor of shape (*batch_shape, condition_dim) containing
                condition vectors for conditional GFlowNets.
            device: The device to create the graph states on.

        Returns:
            A batch of random graph states.
        """
        raise NotImplementedError
