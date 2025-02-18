from typing import Callable

from gfn.states import States
import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates
from gfn.gym.helpers.preprocessors import OneHotPreprocessor


class PerfectBinaryTree(DiscreteEnv):
    r"""
    Perfect Tree Environment where there is a bijection between trajectories and terminating states.
    e.g.:

               1
          /         \
         2           3
       /   \       /   \
      4     5     6     7
     / \   / \   / \   / \
    8   9 10 11 12 13 14 15

    """

    def __init__(self, reward_fn: Callable, depth: int = 4):
        self.reward_fn = reward_fn
        self.depth = depth
        self.branching_factor = 2
        self.n_actions = self.branching_factor + 1
        self.n_nodes = 2 ** (self.depth + 1) - 1

        self.s0 = torch.zeros((1,), dtype=torch.long)
        self.sf = torch.full((1,), fill_value=-1, dtype=torch.long)
        preprocessor = OneHotPreprocessor(self.n_nodes, self.get_states_indices)
        super().__init__(
            self.n_actions, self.s0, (1,), preprocessor=preprocessor, sf=self.sf
        )

        (
            self.transition_table,
            self.inverse_transition_table,
            self.term_states,
        ) = self._build_tree()

    def _build_tree(self) -> tuple[dict, dict, DiscreteStates]:
        """Create a transition table ensuring a bijection between trajectories and last states."""
        transition_table = {}
        inverse_transition_table = {}
        node_index = 0
        queue = [(node_index, 0)]  # (current_node, depth)

        terminating_states_id = set()
        while queue:
            node, d = queue.pop(0)
            if d < self.depth:
                for a in range(self.branching_factor):
                    node_index += 1
                    transition_table[(node, a)] = node_index
                    inverse_transition_table[(node_index, a)] = node
                    queue.append((node_index, d + 1))
            else:
                terminating_states_id.add(node)
        terminating_states_id = torch.tensor(list(terminating_states_id)).reshape(-1, 1)
        terminating_states = self.states_from_tensor(terminating_states_id)

        return transition_table, inverse_transition_table, terminating_states

    def backward_step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        tuples = torch.hstack((states.tensor, actions.tensor)).tolist()
        tuples = tuple((tuple_) for tuple_ in tuples)
        next_states_tns = [
            self.inverse_transition_table.get(tuple_) for tuple_ in tuples
        ]
        next_states_tns = torch.tensor(next_states_tns).reshape(-1, 1)
        return next_states_tns

    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        tuples = torch.hstack((states.tensor, actions.tensor)).tolist()
        tuples = tuple(tuple(tuple_) for tuple_ in tuples)
        next_states_tns = [self.transition_table.get(tuple_) for tuple_ in tuples]
        next_states_tns = torch.tensor(next_states_tns).reshape(-1, 1).long()
        return next_states_tns

    def update_masks(self, states: DiscreteStates) -> None:
        terminating_states_mask = torch.isin(
            states.tensor, self.terminating_states.tensor
        ).flatten()
        initial_state_mask = (states.tensor == self.s0).flatten()
        even_states = (states.tensor % 2 == 0).flatten()

        # Going from any node, we can choose action 0 or 1
        # Except terminating states where we must end the trajectory
        states.forward_masks[~terminating_states_mask, -1] = False
        states.forward_masks[terminating_states_mask, :] = False
        states.forward_masks[terminating_states_mask, -1] = True

        # Even states are to the right, so tied to action 1
        # Uneven states are to the left, tied to action 0
        states.backward_masks[even_states, 1] = True
        states.backward_masks[even_states, 0] = False
        states.backward_masks[~even_states, 1] = False
        states.backward_masks[~even_states, 0] = True

        # Initial state has no available backward action
        states.backward_masks[initial_state_mask, :] = False

    def get_states_indices(self, states: States):
        return torch.flatten(states.tensor)

    @property
    def all_states(self) -> DiscreteStates:
        return self.states_from_tensor(torch.arange(self.n_nodes).reshape(-1, 1))

    @property
    def terminating_states(self) -> DiscreteStates:
        lb = 2**self.depth - 1
        ub = 2 ** (self.depth + 1) - 1
        return self.make_states_class()(torch.arange(lb, ub).reshape(-1, 1))

    def reward(self, final_states):
        return self.reward_fn(final_states.tensor)
