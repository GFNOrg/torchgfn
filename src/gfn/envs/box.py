from typing import ClassVar, Literal, Tuple, cast


import torch
from gymnasium.spaces import Box

from gfn.envs import Env
from gfn.states import States
from gfn.actions import Actions

from torchtyping import TensorType

# Typing
StatesTensor = TensorType["batch_shape", 2, torch.float]
TensorFloat = TensorType["batch_shape", torch.float]
OneActionTensor = TensorType[2]


class BoxEnv(Env):
    """Box environment, corresponding to the one in Section 4.1 of https://arxiv.org/abs/2301.12594"""

    def __init__(
        self,
        delta: float = 0.1,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        assert 0 < delta <= 1, "delta must be in (0, 1]"
        self.delta = delta
        s0 = torch.tensor([0.0, 0.0], device=torch.device(device_str))
        action_space = Box(low=0.0, high=delta, shape=(2,))

        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

        super().__init__(action_space=action_space, s0=s0)

    def make_States_class(self) -> type[States]:
        env = self

        class BoxStates(States):
            state_shape: ClassVar[Tuple[int, ...]] = (2,)
            s0 = env.s0
            sf = env.sf  # should be (-inf, -inf)

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                return torch.rand(batch_shape + (2,), device=env.device)

        return BoxStates

    def make_Actions_class(self) -> type[Actions]:
        env = self

        class BoxActions(Actions):
            action_shape: ClassVar[Tuple[int, ...]] = (2,)
            dummy_action: ClassVar[OneActionTensor] = torch.tensor(
                [-float("inf"), -float("inf")], device=env.device
            )
            exit_action: ClassVar[OneActionTensor] = torch.tensor(
                [-float("inf"), -float("inf")], device=env.device
            )

        return BoxActions

    def maskless_step(self, states: States, actions: Actions) -> StatesTensor:
        return states.states_tensor + actions.actions_tensor

    def maskless_backward_step(self, states: States, actions: Actions) -> StatesTensor:
        return states.states_tensor - actions.actions_tensor

    @staticmethod
    def norm(x: StatesTensor) -> torch.Tensor:
        return torch.norm(x, dim=-1)

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        non_exit_actions = actions[~actions.is_exit]
        non_terminal_states = states.states_tensor[~actions.is_exit]

        s0_states_idx = non_terminal_states.is_initial_state
        if torch.any(s0_states_idx) and backward:
            return False

        if not backward:
            actions_at_s0 = non_exit_actions[s0_states_idx]

            if torch.any(self.norm(actions_at_s0) > self.delta):
                return False

        non_s0_states = non_terminal_states[~s0_states_idx]
        non_s0_actions = non_exit_actions[~s0_states_idx]

        if torch.any(self.norm(non_s0_actions) != self.delta) or torch.any(
            non_s0_actions < 0
        ):
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

    def log_reward(self, final_states: States) -> TensorFloat:
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states.states_tensor - 0.5)
        reward = (
            R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2
        )

        return reward.log()
