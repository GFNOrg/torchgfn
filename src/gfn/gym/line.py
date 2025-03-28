from typing import Literal

import torch
from torch.distributions import Normal  # TODO: extend to Beta

from gfn.actions import Actions
from gfn.env import Env
from gfn.states import States


class Line(Env):
    """Mixture of Gaussians Line environment."""

    def __init__(
        self,
        mus: list,
        sigmas: list,
        init_value: float,
        n_sd: float = 4.5,
        n_steps_per_trajectory: int = 5,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
    ):
        assert len(mus) == len(sigmas)
        self.mus = torch.tensor(mus)
        self.sigmas = torch.tensor(sigmas)
        self.n_sd = n_sd
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.mixture = [Normal(m, s) for m, s in zip(self.mus, self.sigmas)]

        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        self.init_value = init_value  # Used in s0.
        lb = torch.min(self.mus) - self.n_sd * torch.max(self.sigmas)
        ub = torch.max(self.mus) + self.n_sd * torch.max(self.sigmas)
        assert lb < self.init_value < ub

        s0 = torch.tensor([self.init_value, 0.0], device=self.device)
        dummy_action = torch.tensor([float("inf")], device=self.device)
        exit_action = torch.tensor([-float("inf")], device=self.device)
        super().__init__(
            s0=s0,
            state_shape=(2,),  # [x_pos, step_counter].
            action_shape=(1,),  # [x_pos]
            dummy_action=dummy_action,
            exit_action=exit_action,
        )  # sf is -inf by default.

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        states.tensor[..., 0] = states.tensor[..., 0] + actions.tensor.squeeze(
            -1
        )  # x position.
        states.tensor[..., 1] = states.tensor[..., 1] + 1  # Step counter.
        assert states.tensor.shape == states.batch_shape + (2,)
        return states.tensor

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment in the backward direction.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, 2).
        """
        states.tensor[..., 0] = states.tensor[..., 0] - actions.tensor.squeeze(
            -1
        )  # x position.
        states.tensor[..., 1] = states.tensor[..., 1] - 1  # Step counter.
        assert states.tensor.shape == states.batch_shape + (2,)
        return states.tensor

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        # Can't take a backward step at the beginning of a trajectory.
        if torch.any(states[~actions.is_exit].is_initial_state) and backward:
            return False

        return True

    def log_reward(self, final_states: States) -> torch.Tensor:
        """Log reward log of the environment.

        Args:
            final_states: The final states of the environment.

        Returns the log reward as a tensor of shape `batch_shape`.
        """
        s = final_states.tensor[..., 0]
        log_rewards = torch.empty((len(self.mixture),) + final_states.batch_shape)
        for i, m in enumerate(self.mixture):
            log_rewards[i] = m.log_prob(s)

        log_rewards = torch.logsumexp(log_rewards, 0)
        assert log_rewards.shape == final_states.batch_shape
        return log_rewards

    @property
    def log_partition(self) -> torch.Tensor:
        """Log Partition log of the number of gaussians."""
        n_modes = torch.tensor(len(self.mus), device=self.device)
        return n_modes.log()
