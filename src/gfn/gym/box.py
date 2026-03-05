from math import log
from typing import Literal, Tuple

import torch

from gfn.actions import Actions
from gfn.env import Env
from gfn.states import States


class Box(Env):
    """Box environment, corresponding to the one in Section 4.1 of https://arxiv.org/abs/2301.12594

    Attributes:
        delta: The step size.
        R0: The base reward.
        R1: The reward for being outside the first box.
        R2: The reward for being inside the second box.
        epsilon: A small value to avoid numerical issues.
        device (Literal["cpu", "cuda"] | torch.device): The device to use.
    """

    def __init__(
        self,
        delta: float = 0.1,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        epsilon: float = 1e-4,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        debug: bool = False,
    ):
        """Initializes the Box environment.

        Args:
            delta: The step size.
            R0: The base reward.
            R1: The reward for being outside the first box.
            R2: The reward for being inside the second box.
            epsilon: A small value to avoid numerical issues.
            device: The device to use.
            debug: If True, emit States with debug guards (not compile-friendly).
        """
        assert 0 < delta <= 1, "delta must be in (0, 1]"
        self.delta = delta
        self.epsilon = epsilon

        s0 = torch.tensor([0.0, 0.0], device=device)
        exit_action = torch.tensor([-float("inf"), -float("inf")], device=device)
        dummy_action = torch.tensor([float("inf"), float("inf")], device=device)

        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

        super().__init__(
            s0=s0,
            state_shape=(2,),
            action_shape=(2,),
            dummy_action=dummy_action,
            exit_action=exit_action,
            debug=debug,
        )

    def make_random_states(
        self,
        batch_shape: Tuple[int, ...],
        device: torch.device | None = None,
        debug: bool = False,
    ) -> States:
        """Generates random states tensor of shape (*batch_shape, 2).

        Args:
            batch_shape: The shape of the batch.
            device: The device to use.

        Returns:
            A States object with random states.
        """
        device = self.device if device is None else device
        return self.States(torch.rand(batch_shape + (2,), device=device), debug=debug)

    def step(self, states: States, actions: Actions) -> States:
        """Step function for the Box environment.

        Args:
            states: States object representing the current states.
            actions: Actions object representing the actions to be taken.

        Returns:
            The next states as a States object.
        """
        return self.States(states.tensor + actions.tensor)

    def backward_step(self, states: States, actions: Actions) -> States:
        """Backward step function for the Box environment.

        Args:
            states: States object representing the current states.
            actions: Actions object representing the actions to be taken.

        Returns:
            The previous states as a States object.
        """
        return self.States(states.tensor - actions.tensor)

    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        """Computes the L2 norm of the input tensor along the last dimension.

        Args:
            x: Input tensor of shape `(*batch_shape, 2)`.

        Returns:
            Normalized tensor of shape `batch_shape`.
        """
        return torch.norm(x, dim=-1)

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """Checks if the actions are valid (Cartesian per-dimension semantics).

        For Cartesian actions:
        - Forward: action[i] >= delta (min step) and state[i] + action[i] <= 1
        - Forward from s0: action[i] >= 0 and action[i] <= delta (sample in [0, delta])
        - Backward: action[i] >= delta and state[i] - action[i] >= 0
        - Backward to s0: action = state (go directly to origin)

        Args:
            states: The current states.
            actions: The actions to be taken.
            backward: Whether the actions are backward actions.

        Returns:
            True if the actions are valid, False otherwise.
        """
        non_exit_actions = actions[~actions.is_exit]
        non_terminal_states = states[~actions.is_exit]

        if len(non_exit_actions) == 0:
            return True

        s0_states_idx = non_terminal_states.is_initial_state

        # Can't go backward from s0
        if torch.any(s0_states_idx) and backward:
            return False

        # Forward from s0: actions must be in [0, 1] per dimension (full space coverage)
        if not backward and torch.any(s0_states_idx):
            actions_at_s0 = non_exit_actions[s0_states_idx].tensor
            if torch.any(actions_at_s0 < 0) or torch.any(
                actions_at_s0 > 1.0 + self.epsilon
            ):
                return False

        non_s0_states = non_terminal_states[~s0_states_idx].tensor
        non_s0_actions = non_exit_actions[~s0_states_idx].tensor

        if len(non_s0_actions) == 0:
            return True

        # All actions must be non-negative
        if torch.any(non_s0_actions < -self.epsilon):
            return False

        if not backward:
            # Forward from non-s0: actions >= delta and state + action <= 1
            if torch.any(non_s0_actions < self.delta - self.epsilon):
                return False
            if torch.any(non_s0_states + non_s0_actions > 1 + self.epsilon):
                return False
        else:
            # Backward: state - action >= 0
            if torch.any(non_s0_states - non_s0_actions < -self.epsilon):
                return False
            # Backward to s0: if resulting state would be in [0, delta), action must equal state
            resulting_states = non_s0_states - non_s0_actions
            near_origin = torch.all(resulting_states < self.delta, dim=-1)
            if torch.any(near_origin):
                # These should go directly to s0
                if torch.any(torch.abs(resulting_states[near_origin]) > self.epsilon):
                    return False

        return True

    def reward(self, final_states: States) -> torch.Tensor:
        """Reward is distance from the goal point.

        Args:
            final_states: States object representing the final states.

        Returns:
            The reward tensor of shape `batch_shape`.
        """
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states.tensor - 0.5)
        reward = R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2

        assert reward.shape == final_states.batch_shape
        return reward

    @property
    def log_partition(self) -> float:
        """Returns the log partition of the reward function."""
        return log(self.R0 + (2 * 0.25) ** 2 * self.R1 + (2 * 0.1) ** 2 * self.R2)
