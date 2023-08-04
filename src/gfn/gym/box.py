from math import log
from typing import ClassVar, Literal, Tuple

import torch
from torchtyping import TensorType as TT

from gfn.actions import Actions
from gfn.env import Env
from gfn.states import States


class Box(Env):
    """Box environment, corresponding to the one in Section 4.1 of https://arxiv.org/abs/2301.12594"""

    def __init__(
        self,
        delta: float = 0.1,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        epsilon: float = 1e-4,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        assert 0 < delta <= 1, "delta must be in (0, 1]"
        self.delta = delta
        self.epsilon = epsilon
        s0 = torch.tensor([0.0, 0.0], device=torch.device(device_str))

        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

        super().__init__(s0=s0)

    def make_States_class(self) -> type[States]:
        env = self

        class BoxStates(States):
            state_shape: ClassVar[Tuple[int, ...]] = (2,)
            s0 = env.s0
            sf = env.sf  # should be (-inf, -inf)

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> TT["batch_shape", 2, torch.float]:
                return torch.rand(batch_shape + (2,), device=env.device)

        return BoxStates

    def make_Actions_class(self) -> type[Actions]:
        env = self

        class BoxActions(Actions):
            action_shape: ClassVar[Tuple[int, ...]] = (2,)
            dummy_action: ClassVar[TT[2]] = torch.tensor(
                [float("inf"), float("inf")], device=env.device
            )
            exit_action: ClassVar[TT[2]] = torch.tensor(
                [-float("inf"), -float("inf")], device=env.device
            )

        return BoxActions

    def maskless_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", 2, torch.float]:
        return states.tensor + actions.tensor

    def maskless_backward_step(
        self, states: States, actions: Actions
    ) -> TT["batch_shape", 2, torch.float]:
        return states.tensor - actions.tensor

    @staticmethod
    def norm(x: TT["batch_shape", 2, torch.float]) -> torch.Tensor:
        return torch.norm(x, dim=-1)

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        non_exit_actions = actions[~actions.is_exit]
        non_terminal_states = states[~actions.is_exit]

        s0_states_idx = non_terminal_states.is_initial_state
        if torch.any(s0_states_idx) and backward:
            return False

        if not backward:
            actions_at_s0 = non_exit_actions[s0_states_idx].tensor

            if torch.any(self.norm(actions_at_s0) > self.delta):
                return False

        non_s0_states = non_terminal_states[~s0_states_idx].tensor
        non_s0_actions = non_exit_actions[~s0_states_idx].tensor

        if (
            not backward
            and torch.any(torch.abs(self.norm(non_s0_actions) - self.delta) > 1e-5)
        ) or torch.any(non_s0_actions < 0):
            return False

        if not backward and torch.any(non_s0_states + non_s0_actions > 1):
            return False

        if backward and torch.any(non_s0_states - non_s0_actions < 0):
            return False

        if backward:
            states_within_delta_radius_idx = self.norm(non_s0_states) < self.delta
            corresponding_actions = non_s0_actions[states_within_delta_radius_idx]
            corresponding_states = non_s0_states[states_within_delta_radius_idx]
            if torch.any(corresponding_actions != corresponding_states):
                return False

        return True

    def log_reward(self, final_states: States) -> TT["batch_shape", torch.float]:
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states.tensor - 0.5)
        reward = (
            R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
        )

        return reward.log()

    @property
    def log_partition(self) -> float:
        return log(self.R0 + (2 * 0.25) ** 2 * self.R1 + (2 * 0.1) ** 2 * self.R2)
