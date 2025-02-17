from typing import Callable
from gfn.env import Actions, DiscreteEnv, DiscreteStates
import torch


class SetAddition(DiscreteEnv):
    """Append only MDP, similarly to what is described in Remark 8 of Shen et al. 2023
    [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html)
    """

    def __init__(self, n_items: int, max_items: int, reward_fn: Callable):
        self.n_items = n_items
        self.reward_fn = reward_fn
        self.max_traj_len = max_items
        n_actions = n_items + 1
        s0 = torch.zeros(n_items)
        state_shape = (n_items,)

        super().__init__(n_actions, s0, state_shape)

    def get_states_indices(self, states: DiscreteStates):
        states_raw = states.tensor

        canonical_base = 2 ** torch.arange(
            self.n_items - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        return indices

    def update_masks(self, states: DiscreteStates) -> None:
        trajs_that_must_end = states.tensor.sum(dim=1) >= self.max_traj_len
        trajs_that_may_continue = states.tensor.sum(dim=1) < self.max_traj_len

        states.forward_masks[trajs_that_may_continue, : self.n_items] = (
            states.tensor[trajs_that_may_continue] == 0
        )

        # Disallow everything for trajs that must end
        states.forward_masks[trajs_that_must_end, : self.n_items] = 0
        states.forward_masks[..., -1] = 1  # Allow exit action

        states.backward_masks[..., : self.n_items] = states.tensor != 0

        # Disallow exit action if at s_0
        at_initial_state = torch.all(states.tensor == 0, dim=1)
        states.forward_masks[at_initial_state, -1] = 0

    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        return new_states_tensor

    def backward_step(self, states: DiscreteStates, actions: Actions):
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, -1, reduce="add")
        return new_states_tensor

    def get_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return self.reward_fn(final_states.tensor)

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        return torch.log(self.get_reward(final_states))

    @property
    def all_states(self) -> DiscreteStates:
        digits = torch.arange(0, 2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.n_items)
        return DiscreteStates(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        return self.all_states[1:]  # Remove initial state s_0
