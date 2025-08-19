from typing import Literal

import torch
from torch.distributions import Normal  # TODO: extend to Beta

from gfn.actions import Actions
from gfn.env import Env
from gfn.states import States


class Line(Env):
    """Mixture of Gaussians Line environment.

    Attributes:
        mus: The means of the Gaussians.
        sigmas: The standard deviations of the Gaussians.
        n_sd: The number of standard deviations to consider for the bounds.
        n_steps_per_trajectory: The number of steps per trajectory.
        mixture: The mixture of Gaussians.
        init_value: The initial value of the state.
    """

    def __init__(
        self,
        mus: list,
        sigmas: list,
        init_value: float,
        n_sd: float = 4.5,
        n_steps_per_trajectory: int = 5,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        check_action_validity: bool = True,
    ):
        """Initializes the Line environment.

        Args:
            mus: The means of the Gaussians.
            sigmas: The standard deviations of the Gaussians.
            init_value: The initial value of the state.
            n_sd: The number of standard deviations to consider for the bounds.
            n_steps_per_trajectory: The number of steps per trajectory.
            device: The device to use.
        """
        assert len(mus) == len(sigmas)
        self.mus = torch.tensor(mus)
        self.sigmas = torch.tensor(sigmas)
        self.n_sd = n_sd
        self.n_steps_per_trajectory = n_steps_per_trajectory
        self.mixture = [Normal(m, s) for m, s in zip(self.mus, self.sigmas)]

        self.init_value = init_value  # Used in s0.
        lb = torch.min(self.mus) - self.n_sd * torch.max(self.sigmas)
        ub = torch.max(self.mus) + self.n_sd * torch.max(self.sigmas)
        assert lb < self.init_value < ub

        s0 = torch.tensor([self.init_value, 0.0], device=device)
        dummy_action = torch.tensor([float("inf")], device=device)
        exit_action = torch.tensor([-float("inf")], device=device)
        super().__init__(
            s0=s0,
            state_shape=(2,),  # [x_pos, step_counter].
            action_shape=(1,),  # [x_pos]
            dummy_action=dummy_action,
            exit_action=exit_action,
            check_action_validity=check_action_validity,
        )  # sf is -inf by default.

    def step(self, states: States, actions: Actions) -> States:
        """Performs a step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The next states.
        """
        states.tensor[..., 0] = states.tensor[..., 0] + actions.tensor.squeeze(
            -1
        )  # x position.
        states.tensor[..., 1] = states.tensor[..., 1] + 1  # Step counter.
        assert states.tensor.shape == states.batch_shape + (2,)
        return self.States(states.tensor)

    def backward_step(self, states: States, actions: Actions) -> States:
        """Performs a backward step in the environment.

        Args:
            states: The current states.
            actions: The actions to take.

        Returns:
            The previous states.
        """
        states.tensor[..., 0] = states.tensor[..., 0] - actions.tensor.squeeze(
            -1
        )  # x position.
        states.tensor[..., 1] = states.tensor[..., 1] - 1  # Step counter.
        assert states.tensor.shape == states.batch_shape + (2,)
        return self.States(states.tensor)

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """Checks if the actions are valid.

        Args:
            states: The current states.
            actions: The actions to check.
            backward: Whether to check for backward actions.

        Returns:
            `True` if the actions are valid, `False` otherwise.
        """
        # Can't take a backward step at the beginning of a trajectory.
        if torch.any(states[~actions.is_exit].is_initial_state) and backward:
            return False

        return True

    def log_reward(self, final_states: States) -> torch.Tensor:
        """Computes the log reward of the environment.

        Args:
            final_states: The final states of the environment.

        Returns:
            The log reward.
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
        """Returns the log partition of the reward function."""
        n_modes = torch.tensor(len(self.mus), device=self.device)
        return n_modes.log()
