from typing import Callable, Literal

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

    def __init__(
        self,
        reward_fn: Callable,
        depth: int = 4,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
        debug: bool = False,
    ):
        """Initializes the PerfectBinaryTree environment.

        Args:
            reward_fn: A function that computes the reward for a given state.
            depth: The depth of the tree.
            debug: If True, emit States with debug guards (not compile-friendly).
        """
        if device is None:
            device = torch.get_default_device()

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
            debug=debug,
        )
        self.States: type[DiscreteStates] = self.States

        (
            self.transition_table,
            self.inverse_transition_table,
            self.term_states,
        ) = self._build_tree()

    def make_states_class(self) -> type[DiscreteStates]:
        """Returns the DiscreteStates class for the PerfectBinaryTree environment."""
        env = self

        class PerfectBinaryTreeStates(DiscreteStates):
            state_shape = (1,)
            s0 = env.s0
            sf = env.sf
            make_random_states = env.make_random_states
            n_actions = env.n_actions

            def _compute_forward_masks(self) -> torch.Tensor:
                """Computes forward masks for PerfectBinaryTree states."""
                forward_masks = torch.ones(
                    (*self.batch_shape, self.n_actions),
                    dtype=torch.bool,
                    device=self.device,
                )

                # Flatten the states and terminating states tensors for efficient comparison.
                states_flat = self.tensor.view(-1, 1)
                term_tensor = env.term_states.tensor.view(1, -1)
                terminating_states_mask = (states_flat == term_tensor).any(dim=1)

                # Going from any node, we can choose action 0 or 1
                # Except terminating states where we must end the trajectory
                # Reshape mask to match batch shape
                terminating_states_mask = terminating_states_mask.view(self.batch_shape)

                # Non-terminating states: can take actions 0 or 1, but not exit
                forward_masks[~terminating_states_mask, -1] = False

                # Terminating states: only exit action allowed
                forward_masks[terminating_states_mask, :] = False
                forward_masks[terminating_states_mask, -1] = True

                return forward_masks

            def _compute_backward_masks(self) -> torch.Tensor:
                """Computes backward masks for PerfectBinaryTree states."""
                backward_masks = torch.zeros(
                    (*self.batch_shape, self.n_actions - 1),
                    dtype=torch.bool,
                    device=self.device,
                )

                initial_state_mask = (self.tensor == env.s0).view(self.batch_shape)
                even_states = (self.tensor % 2 == 0).view(self.batch_shape)

                # Even states are to the right, so tied to action 1
                # Uneven states are to the left, tied to action 0
                backward_masks[even_states & ~initial_state_mask, 1] = True
                backward_masks[~even_states & ~initial_state_mask, 0] = True

                return backward_masks

        return PerfectBinaryTreeStates

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
        next_states_tns = torch.tensor(
            next_states_tns, device=states.tensor.device, dtype=torch.int64
        ).reshape(-1, 1)
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
        next_states_tns = torch.tensor(
            next_states_tns, device=states.tensor.device, dtype=torch.int64
        ).reshape(-1, 1)
        return self.States(next_states_tns)

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
