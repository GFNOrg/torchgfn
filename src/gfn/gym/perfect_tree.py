from typing import Callable, Literal

import torch

from gfn.env import Actions, DiscreteEnv, DiscreteStates, EnvFastPathMixin
from gfn.states import States


class PerfectBinaryTree(EnvFastPathMixin, DiscreteEnv):
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

    def __init__(
        self,
        reward_fn: Callable,
        depth: int = 4,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        check_action_validity: bool = True,
    ):
        """Initializes the PerfectBinaryTree environment.

        Args:
            reward_fn: A function that computes the reward for a given state.
            depth: The depth of the tree.
            check_action_validity: Whether to check the action validity.
        """
        device = torch.device(device)
        self.reward_fn = reward_fn
        self.depth = depth
        self.branching_factor = 2
        self.n_actions = self.branching_factor + 1
        self.n_nodes = 2 ** (self.depth + 1) - 1

        self.s0 = torch.zeros((1,), dtype=torch.long, device=device)
        self.sf = torch.full((1,), fill_value=-1, device=device)
        super().__init__(
            self.n_actions,
            self.s0,
            (1,),
            sf=self.sf,
            check_action_validity=check_action_validity,
        )
        self.States: type[DiscreteStates] = self.States

        (
            self.transition_table,
            self.inverse_transition_table,
            self.term_states,
        ) = self._build_tree()
        self._leaf_lower = 2**self.depth - 1
        self._leaf_upper = 2 ** (self.depth + 1) - 1

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
        terminating_states_id = torch.tensor(
            list(terminating_states_id), device=self.device
        ).reshape(-1, 1)

        # Create the terminating states object. Don't use `self.states_from_tensor`
        # because it will internally trigger update_masks, which would fail because
        # `term_states` isn't yet defined.
        terminating_states = self.States(terminating_states_id)

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
        next_states_tns = (
            torch.tensor(next_states_tns, device=states.tensor.device)
            .reshape(-1, 1)
            .long()
        )
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
        next_states_tns = (
            torch.tensor(next_states_tns, device=states.tensor.device)
            .reshape(-1, 1)
            .long()
        )
        return self.States(next_states_tns)

    def update_masks(self, states: DiscreteStates) -> None:
        """Updates the masks of the states.

        Args:
            states: The states to update the masks of.
        """
        # Flatten the states and terminating states tensors for efficient comparison.
        states_flat = states.tensor.view(-1, 1)
        term_tensor = self.term_states.tensor.view(1, -1)
        terminating_states_mask = (states_flat == term_tensor).any(dim=1)
        initial_state_mask = (states.tensor == self.s0).view(-1)
        even_states = (states.tensor % 2 == 0).view(-1)

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

    def _is_leaf_tensor(self, states_tensor: torch.Tensor) -> torch.Tensor:
        values = states_tensor.view(-1)
        return (values >= self._leaf_lower) & (values < self._leaf_upper)

    def forward_action_masks_tensor(self, states_tensor: torch.Tensor) -> torch.Tensor:
        batch = states_tensor.shape[0]
        device = states_tensor.device
        masks = torch.zeros((batch, self.n_actions), dtype=torch.bool, device=device)
        leaf_mask = self._is_leaf_tensor(states_tensor)
        sink_mask = (states_tensor == self.sf.to(device)).all(dim=-1)
        non_leaf = ~(leaf_mask | sink_mask)
        masks[non_leaf, : self.branching_factor] = True
        masks[leaf_mask | sink_mask, -1] = True
        return masks

    def step_tensor(
        self, states_tensor: torch.Tensor, actions_tensor: torch.Tensor
    ) -> DiscreteEnv.TensorStepResult:
        if actions_tensor.ndim == 1:
            actions_idx = actions_tensor.view(-1, 1)
        else:
            assert actions_tensor.shape[-1] == 1
            actions_idx = actions_tensor

        exit_idx = self.n_actions - 1
        device = states_tensor.device
        next_states = states_tensor.clone()
        actions_flat = actions_idx.squeeze(-1)
        state_vals = next_states.squeeze(-1)

        is_exit = actions_flat == exit_idx
        non_exit = ~is_exit
        if non_exit.any():
            parents = state_vals[non_exit]
            child_idx = parents.clone()
            left_mask = actions_flat[non_exit] == 0
            right_mask = actions_flat[non_exit] == 1
            if left_mask.any():
                child_idx[left_mask] = 2 * parents[left_mask] + 1
            if right_mask.any():
                child_idx[right_mask] = 2 * parents[right_mask] + 2
            next_states[non_exit, 0] = child_idx

        if is_exit.any():
            next_states[is_exit] = self.sf.to(device=device)

        forward_masks = self.forward_action_masks_tensor(next_states)
        backward_masks = torch.zeros(
            (next_states.shape[0], self.branching_factor),
            dtype=torch.bool,
            device=device,
        )
        next_vals = next_states.squeeze(-1)
        sink_mask = (next_states == self.sf.to(device)).all(dim=-1)
        initial_mask = next_vals == self.s0.item()
        even_mask = (next_vals % 2 == 0) & ~sink_mask
        odd_mask = (next_vals % 2 == 1) & ~sink_mask
        backward_masks[even_mask, 1] = True
        backward_masks[odd_mask, 0] = True
        backward_masks[initial_mask] = False

        is_sink_state = sink_mask

        return self.TensorStepResult(
            next_states=next_states,
            is_sink_state=is_sink_state,
            forward_masks=forward_masks,
            backward_masks=backward_masks,
        )

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
