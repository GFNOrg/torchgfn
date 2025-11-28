from typing import Callable, Literal

import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates, EnvFastPathMixin


class SetAddition(EnvFastPathMixin, DiscreteEnv):
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

    def __init__(
        self,
        n_items: int,
        max_items: int,
        reward_fn: Callable,
        fixed_length: bool = False,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        check_action_validity: bool = True,
    ):
        """Initializes the SetAddition environment.

        Args:
            n_items: The number of items in the set.
            max_items: The maximum number of items that can be added to the set.
            reward_fn: The reward function.
            fixed_length: Whether the trajectories have a fixed length.
            check_action_validity: Whether to check the action validity.
        """
        device = torch.device(device)
        self.n_items = n_items
        self.reward_fn = reward_fn
        self.max_traj_len = max_items
        self.fixed_length = fixed_length
        n_actions = n_items + 1
        s0 = torch.zeros(n_items, device=device)
        state_shape = (n_items,)

        super().__init__(
            n_actions,
            s0,
            state_shape,
            check_action_validity=check_action_validity,
        )
        self.States: type[DiscreteStates] = self.States

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

    def update_masks(self, states: DiscreteStates) -> None:
        """Updates the masks of the states.

        Args:
            states: The states to update the masks of.
        """
        n_items_per_state = states.tensor.sum(dim=-1)
        states_that_must_end = n_items_per_state >= self.max_traj_len
        states_that_may_continue = (n_items_per_state < self.max_traj_len) & (
            n_items_per_state >= 0
        )

        cont_f_mask = torch.cat(
            (
                (states.tensor[states_that_may_continue] == 0),
                torch.zeros(
                    states.tensor[states_that_may_continue].shape[0],
                    1,
                    dtype=torch.bool,
                    device=states.tensor.device,
                ),
            ),
            1,
        )

        states.forward_masks[states_that_may_continue] = cont_f_mask
        # Disallow everything for trajs that must end
        end_f_mask = torch.zeros(
            states.tensor[states_that_must_end].shape[0],
            states.forward_masks.shape[-1],
            dtype=torch.bool,
            device=states.tensor.device,
        )
        end_f_mask[..., -1] = True

        states.forward_masks[states_that_must_end] = end_f_mask

        # Disallow everything for trajs that must end
        # states.forward_masks[states_that_must_end, : self.n_items] = 0
        if not self.fixed_length:
            states.forward_masks[..., -1] = 1  # Allow exit action

        states.backward_masks[..., : self.n_items] = states.tensor != 0

    def forward_action_masks_tensor(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """Tensor equivalent of `update_masks` for forward masks."""

        batch = states_tensor.shape[0]
        device = states_tensor.device
        masks = torch.zeros((batch, self.n_actions), dtype=torch.bool, device=device)

        n_items_per_state = states_tensor.sum(dim=-1)
        states_that_must_end = n_items_per_state >= self.max_traj_len
        states_that_may_continue = ~states_that_must_end

        if states_that_may_continue.any():
            cont_states = states_tensor[states_that_may_continue] == 0
            cont_masks = torch.zeros(
                (cont_states.shape[0], self.n_actions),
                dtype=torch.bool,
                device=device,
            )
            cont_masks[:, : self.n_items] = cont_states
            masks[states_that_may_continue] = cont_masks

        if states_that_must_end.any():
            end_masks = torch.zeros(
                (int(states_that_must_end.sum().item()), self.n_actions),
                dtype=torch.bool,
                device=device,
            )
            end_masks[:, -1] = True
            masks[states_that_must_end] = end_masks

        if not self.fixed_length:
            masks[..., -1] = True

        return masks

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

    def step_tensor(
        self, states_tensor: torch.Tensor, actions_tensor: torch.Tensor
    ) -> DiscreteEnv.TensorStepResult:
        """Tensor-only transition mirroring the legacy `_step` path."""

        if actions_tensor.ndim == 1:
            actions_idx = actions_tensor.view(-1, 1)
        else:
            assert actions_tensor.shape[-1] == 1
            actions_idx = actions_tensor

        exit_idx = self.n_actions - 1
        is_exit = actions_idx.squeeze(-1) == exit_idx
        next_states = states_tensor.clone()

        non_exit_mask = ~is_exit
        if torch.any(non_exit_mask):
            sel_states = next_states[non_exit_mask]
            sel_actions = actions_idx[non_exit_mask]
            sel_states = sel_states.scatter(-1, sel_actions, 1, reduce="add")
            next_states[non_exit_mask] = sel_states

        if torch.any(is_exit):
            next_states[is_exit] = self.sf.to(device=states_tensor.device)

        forward_masks = self.forward_action_masks_tensor(next_states)
        backward_masks = torch.zeros_like(forward_masks)
        backward_masks[..., : self.n_items] = next_states != 0
        is_sink_state = (next_states == self.sf.to(device=states_tensor.device)).all(
            dim=-1
        )

        return self.TensorStepResult(
            next_states=next_states,
            is_sink_state=is_sink_state,
            forward_masks=forward_masks,
            backward_masks=backward_masks,
        )

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
        """Returns the terminating states of the environment."""
        if self.fixed_length:
            return self.all_states[
                self.all_states.tensor.sum(dim=1) == self.max_traj_len
            ]

        else:
            return self.all_states[1:]  # Remove initial state s_0
