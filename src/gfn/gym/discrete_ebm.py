from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch
import torch.nn as nn

from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import EnumPreprocessor, IdentityPreprocessor
from gfn.states import DiscreteStates, States


class EnergyFunction(nn.Module, ABC):
    """Base class for energy functions"""

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the energy function

        Args:
            states: tensor of states of shape (*batch_shape, *state_shape)

        Returns tensor of energies of shape (*batch_shape)
        """


class IsingModel(EnergyFunction):
    """Ising model energy function"""

    def __init__(self, J: torch.Tensor):
        """Ising model energy function

        Args:
            J: interaction matrix of shape (state_shape, state_shape)
        """
        super().__init__()
        self.J = J
        self._state_shape, _ = J.shape
        assert J.shape == (self._state_shape, self._state_shape)
        self.linear = nn.Linear(self._state_shape, 1, bias=False)
        self.linear.weight.data = J

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ising model.

        Args:
            states: tensor of states of shape (*batch_shape, *state_shape)

        Returns tensor of energies of shape (*batch_shape)
        """
        assert states.shape[-1] == self._state_shape
        states = states.float()
        tmp = self.linear(states)
        return -(states * tmp).sum(-1)


class DiscreteEBM(DiscreteEnv):
    """Environment for discrete energy-based models, based on https://arxiv.org/pdf/2202.01361.pdf"""

    def __init__(
        self,
        ndim: int,
        energy: EnergyFunction | None = None,
        alpha: float = 1.0,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["Identity", "Enum"] = "Identity",
    ):
        """Discrete EBM environment.

        Args:
            ndim: dimension D of the sampling space {0, 1}^D.
            energy: energy function of the EBM. Defaults to None. If
                None, the Ising model with Identity matrix is used.
            alpha: interaction strength the EBM. Defaults to 1.0.
            device_str: "cpu" or "cuda". Defaults to "cpu".
            preprocessor_name: "KHot" or "OneHot" or "Identity".
                Defaults to "KHot".
        """
        self.ndim = ndim

        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), 2, dtype=torch.long, device=torch.device(device_str))

        if energy is None:
            energy = IsingModel(
                torch.ones((ndim, ndim), device=torch.device(device_str))
            )
        self.energy: EnergyFunction = energy
        self.alpha = alpha

        n_actions = 2 * ndim + 1
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2 * ndim - 1] corresponds to replacing s[i - ndim] with 1

        if preprocessor_name == "Identity":
            preprocessor = IdentityPreprocessor(output_dim=ndim)
        elif preprocessor_name == "Enum":
            preprocessor = EnumPreprocessor(
                get_states_indices=self.get_states_indices,
            )
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(
            s0=s0,
            state_shape=(self.ndim,),
            # dummy_action=,
            # exit_action=,
            n_actions=n_actions,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def update_masks(self, states: type[States]) -> None:
        states.forward_masks[..., : self.ndim] = states.tensor == -1
        states.forward_masks[..., self.ndim : 2 * self.ndim] = states.tensor == -1
        states.forward_masks[..., -1] = torch.all(states.tensor != -1, dim=-1)
        states.backward_masks[..., : self.ndim] = states.tensor == 0
        states.backward_masks[..., self.ndim : 2 * self.ndim] = states.tensor == 1

    def make_random_states_tensor(self, batch_shape: Tuple) -> torch.Tensor:
        """Generates random states tensor of shape (*batch_shape, ndim)."""
        return torch.randint(
            -1,
            2,
            batch_shape + (self.ndim,),
            dtype=torch.long,
            device=self.device,
        )

    def is_exit_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Determines if the actions are exit actions.

        Args:
            actions: tensor of actions of shape (*batch_shape, *action_shape)

        Returns tensor of booleans of shape (*batch_shape)
        """
        return actions == self.n_actions - 1

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Performs a step.

        Args:
            states: States object representing the current states.
            actions: Actions object representing the actions to be taken.

        Returns the next states as tensor of shape (*batch_shape, ndim).
        """
        # First, we select that actions that replace a -1 with a 0.
        # Remove singleton dimension for broadcasting.
        mask_0 = (actions.tensor < self.ndim).squeeze(-1)
        states.tensor[mask_0] = states.tensor[mask_0].scatter(
            -1, actions.tensor[mask_0], 0  # Set indices to 0.
        )
        # Then, we select that actions that replace a -1 with a 1.
        mask_1 = (
            (actions.tensor >= self.ndim) & (actions.tensor < 2 * self.ndim)
        ).squeeze(
            -1
        )  # Remove singleton dimension for broadcasting.
        states.tensor[mask_1] = states.tensor[mask_1].scatter(
            -1, (actions.tensor[mask_1] - self.ndim), 1  # Set indices to 1.
        )
        return states.tensor

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        """Performs a backward step.

        In this env, states are n-dim vectors. s0 is empty (represented as -1),
            so s0=[-1, -1, ..., -1], each action is replacing a -1 with either a
            0 or 1. Action i in [0, ndim-1] os replacing s[i] with 0, whereas
            action i in [ndim, 2*ndim-1] corresponds to replacing s[i - ndim] with 1.
            A backward action asks "what index should be set back to -1", hence the fmod
            to enable wrapping of indices.
        """
        return states.tensor.scatter(-1, actions.tensor.fmod(self.ndim), -1)

    def reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """Not used during training but provided for completeness.

        Note the effect of clipping will be seen in these values.

        Args:
            final_states: DiscreteStates object representing the final states.

        Returns the reward as tensor of shape (*batch_shape).
        """
        reward = torch.exp(self.log_reward(final_states))
        assert reward.shape == final_states.batch_shape
        return reward

    def log_reward(self, final_states: DiscreteStates) -> torch.Tensor:
        """The energy weighted by alpha is our log reward.

        Args:
            final_states: DiscreteStates object representing the final states.

        Returns the log reward as tensor of shape (*batch_shape)."""
        raw_states = final_states.tensor
        canonical = 2 * raw_states - 1
        log_reward = -self.alpha * self.energy(canonical)

        assert log_reward.shape == final_states.batch_shape
        return log_reward

    def get_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2, then we convert to base 3

        Args:
            states: DiscreteStates object representing the states.

        Returns the states indices as tensor of shape (*batch_shape).
        """
        states_raw = states.tensor
        canonical_base = 3 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        states_indices = (states_raw + 1).mul(canonical_base).sum(-1).long()
        assert states_indices.shape == states.batch_shape
        return states_indices

    def get_terminating_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """Returns the indices of the terminating states.

        Args:
            states: DiscreteStates object representing the states.

        Returns the indices of the terminating states as tensor of shape (*batch_shape).
        """
        states_raw = states.tensor
        canonical_base = 2 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        states_indices = (states_raw).mul(canonical_base).sum(-1).long()
        assert states_indices.shape == states.batch_shape
        return states_indices

    @property
    def n_states(self) -> int:
        return 3**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return 2**self.ndim

    @property
    def all_states(self) -> DiscreteStates:
        # This is brute force !
        digits = torch.arange(3, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        all_states = all_states - 1
        return self.States(all_states)

    @property
    def terminating_states(self) -> DiscreteStates:
        digits = torch.arange(2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        true_dist = self.reward(self.terminating_states)
        return true_dist / true_dist.sum()

    @property
    def log_partition(self) -> float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1).item()
