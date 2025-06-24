from typing import Callable

import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates


class SetAddition(DiscreteEnv):
    """Append only MDP, similarly to what is described in Remark 8 of Shen et al. 2023
    [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html)

    The state is a binary vector of length `n_items`, where 1 indicates the presence of an item.
    Actions are integers from 0 to `n_items - 1` to add the corresponding item, or `n_items` to exit.
    Adding an existing item is invalid. The trajectory must end when `max_items` are present.

    Recommended preprocessor: `IdentityPreprocessor`.
    """

    def __init__(
        self,
        n_items: int,
        max_items: int,
        reward_fn: Callable,
        fixed_length: bool = False,
    ):
        """
        Args:
            n_items: number of items in the append MDP
            max_items: max number of items that can be added to the 'set'. Defines maximum trajectory length
            reward_fn: reward function to use
            fixed_length: indicates whether every trajectory ends after the same number of steps (i.e. if terminating state space contains only sets of the same size).
        """
        self.n_items = n_items
        self.reward_fn = reward_fn
        self.max_traj_len = max_items
        self.fixed_length = fixed_length
        n_actions = n_items + 1
        s0 = torch.zeros(n_items)
        state_shape = (n_items,)

        super().__init__(n_actions, s0, state_shape)
        self.States: type[DiscreteStates] = self.States

    def get_states_indices(self, states: DiscreteStates):
        states_raw = states.tensor

        canonical_base = 2 ** torch.arange(
            self.n_items - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    def update_masks(self, states: DiscreteStates) -> None:
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

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        return self.States(new_states_tensor)

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")
        return self.States(new_states_tensor)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return self.reward_fn(final_states.tensor)

    @property
    def all_states(self) -> DiscreteStates:
        digits = torch.arange(0, 2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.n_items)
        return self.states_from_tensor(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        if self.fixed_length:
            return self.all_states[
                self.all_states.tensor.sum(dim=1) == self.max_traj_len
            ]

        else:
            return self.all_states[1:]  # Remove initial state s_0
