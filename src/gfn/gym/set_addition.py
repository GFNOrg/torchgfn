import math
from typing import Callable, Literal

import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates


class SetAddition(DiscreteEnv):
    """Append only MDP, similarly to what is described in Remark 8 of Shen et al. 2023
    [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html)

    The state is a binary vector of length `n_items`, where 1 indicates the presence of an item.
    Actions are integers from 0 to `n_items - 1` to add the corresponding item, or `n_items` to exit.
    Adding an existing item is invalid. The trajectory must end when `max_items` are present.

    Recommended preprocessor: `IdentityPreprocessor`.

    Attributes:
        n_items (int): The number of items in the set.
        max_items (int): The maximum number of items that can be added to the set.
        reward_fn (Callable): The reward function.
        fixed_length (bool): Whether the trajectories have a fixed length.
    """

    supports_enumeration = True

    def __init__(
        self,
        n_items: int,
        max_items: int,
        reward_fn: Callable,
        fixed_length: bool = False,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
        debug: bool = False,
    ):
        """Initializes the SetAddition environment.

        Args:
            n_items: The number of items in the set.
            max_items: The maximum number of items that can be added to the set.
            reward_fn: The reward function.
            fixed_length: Whether the trajectories have a fixed length.
            debug: If True, emit States with debug guards (not compile-friendly).
        """
        if device is None:
            device = torch.get_default_device()

        device = torch.device(device)
        self.n_items = n_items
        self.reward_fn = reward_fn
        self._terminating_positions: torch.Tensor | None = None
        self.max_traj_len = max_items
        self.fixed_length = fixed_length
        n_actions = n_items + 1
        s0 = torch.zeros(n_items, device=device)
        state_shape = (n_items,)

        super().__init__(
            n_actions,
            s0,
            state_shape,
            debug=debug,
        )
        self.States: type[DiscreteStates] = self.States

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns the DiscreteStates class for the SetAddition environment."""
        env = self

        class SetAdditionStates(DiscreteStates):
            state_shape = (env.n_items,)
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self) -> torch.Tensor:
                """Computes forward masks for SetAddition states."""
                n_items_per_state = self.tensor.sum(dim=-1)
                states_that_must_end = n_items_per_state >= env.max_traj_len
                states_that_may_continue = (n_items_per_state < env.max_traj_len) & (
                    n_items_per_state >= 0
                )

                forward_masks = torch.zeros(
                    (*self.batch_shape, self.n_actions),
                    dtype=torch.bool,
                    device=self.device,
                )

                # For states that may continue: can add items not yet in set
                forward_masks[states_that_may_continue, : env.n_items] = (
                    self.tensor[states_that_may_continue] == 0
                )

                # For states that must end: only exit action allowed
                forward_masks[states_that_must_end, -1] = True

                # Allow exit action for all states if not fixed_length
                if not env.fixed_length:
                    forward_masks[..., -1] = True

                return forward_masks

            def _compute_backward_masks(self) -> torch.Tensor:
                """Computes backward masks for SetAddition states."""
                backward_masks = torch.zeros(
                    (*self.batch_shape, self.n_actions - 1),
                    dtype=torch.bool,
                    device=self.device,
                )
                # Can remove items that are in the set
                backward_masks[..., : env.n_items] = self.tensor != 0
                return backward_masks

        return SetAdditionStates

    def get_states_indices(self, states: DiscreteStates):
        """Returns the indices of the states.

        Args:
            states: The states to get the indices of.

        Returns:
            The indices of the states.
        """
        states_raw = states.tensor

        canonical_base = 2 ** torch.arange(
            self.n_items - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        return self.States(new_states_tensor)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a backward step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The previous states.
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")
        return self.States(new_states_tensor)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """Computes the reward for a batch of final states.

        Args:
            final_states: The final states.

        Returns:
            The reward of the final states.
        """
        return self.reward_fn(final_states.tensor)

    @property
    def all_states(self) -> DiscreteStates:
        """Returns all the states of the environment."""
        digits = torch.arange(0, 2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.n_items)
        return self.states_from_tensor(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        """Returns the terminating states of the environment.

        A state terminates a trajectory if it is *reachable* from $s_0$ and its forward
        mask permits the exit action. Reachability matters here: ``all_states`` is the
        full binary cube, but the forward masks forbid adding an item once
        ``max_items`` are present, so any state with more than ``max_items`` items is
        unreachable and carries no probability mass.

        With ``fixed_length``, only sets of exactly ``max_items`` items terminate.
        Otherwise every reachable state does — including the empty set $s_0$, which the
        forward masks do allow the policy to exit from immediately.
        """
        return self.all_states[self._terminating_mask]

    @property
    def _terminating_mask(self) -> torch.Tensor:
        """Boolean mask over ``all_states`` selecting the reachable terminating states."""
        sizes = self.all_states.tensor.sum(dim=1)
        if self.fixed_length:
            return sizes == self.max_traj_len
        return sizes <= self.max_traj_len

    @property
    def n_states(self) -> int:
        """Returns the number of states, i.e. the size of the binary cube."""
        return 2**self.n_items

    @property
    def n_terminating_states(self) -> int:
        """Returns the number of reachable states that can terminate a trajectory."""
        if self.fixed_length:
            return math.comb(self.n_items, self.max_traj_len)
        return sum(math.comb(self.n_items, k) for k in range(self.max_traj_len + 1))

    def get_terminating_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Returns positions in the canonical terminating-state ordering.

        The ordering is the one induced by ``terminating_states``, i.e. the order the
        terminating states appear in ``all_states``.

        Args:
            states: The terminating states to index.

        Returns:
            A tensor of shape (*batch_shape,) of indices into ``terminating_states``.
        """
        if self._terminating_positions is None:
            positions = torch.full(
                (self.n_states,), -1, dtype=torch.long, device=self.device
            )
            mask = self._terminating_mask
            positions[mask] = torch.arange(int(mask.sum()), device=self.device)
            self._terminating_positions = positions

        indices = self._terminating_positions[self.get_states_indices(states)]
        if self.debug:
            assert bool((indices >= 0).all()), (
                "get_terminating_states_indices was given states that cannot terminate "
                "a trajectory (they are unreachable, or their forward mask forbids exit)."
            )
        return indices
