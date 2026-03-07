import torch

from gfn.actions import Actions
from gfn.gym.box import BoxPolar
from gfn.states import States


class BoxCartesian(BoxPolar):
    """Box environment with Cartesian per-dimension action validation.

    Inherits all behavior from :class:`BoxPolar` (init, step, backward_step,
    reward, log_partition, norm, make_random_states). Overrides only
    :meth:`is_action_valid` to use per-dimension bounds instead of polar norm
    constraints.

    Use with the Cartesian estimators/distributions in
    ``box_cartesian_utils.py``.

    See Also:
        :class:`BoxPolar` for the original polar norm-based variant.
    """

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """Checks if the actions are valid (Cartesian per-dimension semantics).

        For Cartesian actions:
        - Forward from s0: action[i] >= 0 and action[i] <= 1
        - Forward from non-s0: action[i] >= delta and state[i] + action[i] <= 1
        - Backward: state[i] - action[i] >= 0
        - Backward to s0: if all resulting dims < delta, action must equal state

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
