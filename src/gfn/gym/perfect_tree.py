from typing import Callable

import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates
from gfn.states import States


class PerfectBinaryTree(DiscreteEnv):
    r"""Perfect Tree Environment.

    This environment is a perfect binary tree, where there is a bijection between
    trajectories and terminating states. Nodes are represented by integers, starting
    from 0 for the root. States are represented by a single integer tensor
    corresponding to the node index. Actions are integers: 0 (left child), 1 (right
    child), 2 (exit).

    e.g.:

               0 (root)
          /         \
         1           2
       /   \       /   \
      3     4     5     6
     / \   / \   / \   / \
    7   8  9  10 11 12 13 14 (terminating states if depth=3)

    Recommended preprocessor: `OneHotPreprocessor`.

    Attributes:
        reward_fn (Callable): A function that computes the reward for a given state.
        depth (int): The depth of the tree.
        branching_factor (int): The branching factor of the tree.
        n_actions (int): The number of actions.
        n_nodes (int): The number of nodes in the tree.
        transition_table (dict): A dictionary that maps (state, action) to the next state.
        inverse_transition_table (dict): A dictionary that maps (state, action) to the previous state.
        term_states (DiscreteStates): The terminating states.
    """

    def __init__(self, reward_fn: Callable, depth: int = 4):
        """Initializes the PerfectBinaryTree environment.

        Args:
            reward_fn: A function that computes the reward for a given state.
            depth: The depth of the tree.
        """
        self.reward_fn = reward_fn
        self.depth = depth
        self.branching_factor = 2
        self.n_actions = self.branching_factor + 1
        self.n_nodes = 2 ** (self.depth + 1) - 1

        self.s0 = torch.zeros((1,), dtype=torch.long)
        self.sf = torch.full((1,), fill_value=-1)
        super().__init__(self.n_actions, self.s0, (1,), sf=self.sf)
        self.States: type[DiscreteStates] = self.States

        (
            self.transition_table,
            self.inverse_transition_table,
            self.term_states,
        ) = self._build_tree()

    def _build_tree(self) -> tuple[dict, dict, DiscreteStates]:
        """Builds the tree and the transition tables.

        Returns:
            A tuple containing the transition table, the inverse transition table,
            and the terminating states.
        """
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

    def backward_step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a backward step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The previous states.
        """
        tuples = torch.hstack((states.tensor, actions.tensor)).tolist()
        tuples = tuple((tuple_) for tuple_ in tuples)
        next_states_tns = [
            self.inverse_transition_table.get(tuple(tuple_)) for tuple_ in tuples
        ]
        next_states_tns = torch.tensor(next_states_tns).reshape(-1, 1).long()
        return self.States(next_states_tns)

    def step(self, states: DiscreteStates, actions: Actions) -> DiscreteStates:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        tuples = torch.hstack((states.tensor, actions.tensor)).tolist()
        tuples = tuple(tuple(tuple_) for tuple_ in tuples)
        next_states_tns = [self.transition_table.get(tuple_) for tuple_ in tuples]
        next_states_tns = torch.tensor(next_states_tns).reshape(-1, 1).long()
        return self.States(next_states_tns)

    def update_masks(self, states: DiscreteStates) -> None:
        """Updates the masks of the states.

        Args:
            states: The states to update the masks of.
        """
        terminating_states_mask = torch.isin(
            states.tensor, self.terminating_states.tensor
        ).squeeze()
        initial_state_mask = (states.tensor == self.s0).squeeze()
        even_states = (states.tensor % 2 == 0).squeeze()

        # Going from any node, we can choose action 0 or 1
        # Except terminating states where we must end the trajectory
        not_term_mask = states.forward_masks[~terminating_states_mask]
        not_term_mask[:, -1] = False

        term_mask = states.forward_masks[terminating_states_mask]
        term_mask[:, :] = False
        term_mask[:, -1] = True

        states.forward_masks[~terminating_states_mask] = not_term_mask
        states.forward_masks[terminating_states_mask] = term_mask

        # Even states are to the right, so tied to action 1
        # Uneven states are to the left, tied to action 0
        even_mask = states.backward_masks[even_states]
        odd_mask = states.backward_masks[~even_states]

        even_mask[:, 0] = False
        even_mask[:, 1] = True
        odd_mask[:, 0] = True
        odd_mask[:, 1] = False
        states.backward_masks[even_states] = even_mask
        states.backward_masks[~even_states] = odd_mask

        # Initial state has no available backward action
        states.backward_masks[initial_state_mask] = False
        states.backward_masks[initial_state_mask] = False

    def get_states_indices(self, states: States):
        """Returns the indices of the states.

        Args:
            states: The states to get the indices of.

        Returns:
            The indices of the states.
        """
        return torch.flatten(states.tensor)

    @property
    def all_states(self) -> DiscreteStates:
        """Returns all the states of the environment."""
        return self.states_from_tensor(torch.arange(self.n_nodes).reshape(-1, 1))

    @property
    def terminating_states(self) -> DiscreteStates:
        """Returns the terminating states of the environment."""
        lb = 2**self.depth - 1
        ub = 2 ** (self.depth + 1) - 1
        return self.make_states_class()(torch.arange(lb, ub).reshape(-1, 1))

    def reward(self, final_states):
        """Computes the reward for a batch of final states.

        Args:
            final_states: The final states.

        Returns:
            The reward of the final states.
        """
        return self.reward_fn(final_states.tensor)
