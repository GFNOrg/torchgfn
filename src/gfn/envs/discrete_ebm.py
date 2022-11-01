from typing import Callable, ClassVar, Literal, Tuple, cast
from abc import ABC, abstractmethod


import torch
import torch.nn as nn
from gfn.containers.states import States
from gfn.envs.env import Env
from gymnasium.spaces import Discrete
from torchtyping import TensorType


class EnergyFunction(nn.Module, ABC):
    """Base class for energy functions"""

    @abstractmethod
    def forward(self, states: TensorType["B", "D"]) -> TensorType["B"]:
        pass


class IsingModel(EnergyFunction):
    """Ising model energy function"""

    def __init__(self, J: TensorType["D", "D"]):
        super().__init__()
        self.J = J
        self.linear = nn.Linear(J.shape[0], 1, bias=False)
        self.linear.weight.data = J

    def forward(self, states: TensorType["B", "D"]) -> TensorType["B"]:
        states = states.float()
        tmp = self.linear(states)
        return -(states * tmp).sum(-1)


class DiscreteEBMEnv(Env):
    """Environment for discrete energy-based models, based on https://arxiv.org/pdf/2202.01361.pdf"""

    def __init__(
        self,
        ndim: int,
        energy: EnergyFunction | None = None,
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        """Discrete EBM environment.

        Args:
            ndim (int, optional): dimension D of the sampling space {0, 1}^D.
            energy (EnergyFunction): energy function of the EBM. Defaults to None. If None, the Ising model with Identity matrix is used.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim

        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), 2, dtype=torch.long, device=torch.device(device_str))

        if energy is None:
            energy = IsingModel(
                torch.ones((ndim, ndim), device=torch.device(device_str))
            )
        self.energy: EnergyFunction = energy

        action_space = Discrete(
            2 * ndim + 1
        )  # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2 * ndim - 1] corresponds to replacing s[i - ndim] with 1

        super().__init__(action_space=action_space, s0=s0, sf=sf)

    def make_States_class(self) -> type[States]:
        env = self

        class DiscreteEBMStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> TensorType["B", env.ndim]:
                return torch.randint(
                    -1,
                    2,
                    batch_shape + (env.ndim,),
                    dtype=torch.long,
                    device=env.device,
                )

            def make_masks(
                self,
            ) -> Tuple[
                TensorType["B", env.n_actions], TensorType["B", env.n_actions - 1]
            ]:
                forward_masks = torch.zeros(
                    self.batch_shape + (env.n_actions,),
                    device=env.device,
                    dtype=torch.bool,
                )
                backward_masks = torch.zeros(
                    self.batch_shape + (env.n_actions - 1,),
                    device=env.device,
                    dtype=torch.bool,
                )

                return forward_masks, backward_masks

            def update_masks(self) -> None:
                # The following two lines are for typing only.
                self.forward_masks = cast(
                    TensorType["B", env.n_actions], self.forward_masks
                )
                self.backward_masks = cast(
                    TensorType["B", env.n_actions - 1], self.backward_masks
                )

                self.forward_masks[..., : env.ndim] = self.states_tensor == -1
                self.forward_masks[..., env.ndim : 2 * env.ndim] = (
                    self.states_tensor == -1
                )
                self.forward_masks[..., -1] = torch.all(
                    self.states_tensor != -1, dim=-1
                )
                self.backward_masks[..., : env.ndim] = self.states_tensor == 0
                self.backward_masks[..., env.ndim : 2 * env.ndim] = (
                    self.states_tensor == 1
                )

        return DiscreteEBMStates

    def is_exit_actions(
        self, actions: TensorType["B", torch.long]
    ) -> TensorType["B", torch.bool]:
        return actions == self.n_actions - 1

    def maskless_step(
        self, states: TensorType["B", "D"], actions: TensorType["B", torch.long]
    ) -> None:
        # First, we select that actions that replace a -1 with a 0
        mask_0 = actions < self.ndim
        states[mask_0] = states[mask_0].scatter(-1, actions[mask_0].unsqueeze(-1), 0)
        # Then, we select that actions that replace a -1 with a 1
        mask_1 = (actions >= self.ndim) & (actions < 2 * self.ndim)
        states[mask_1] = states[mask_1].scatter(
            -1, (actions[mask_1] - self.ndim).unsqueeze(-1), 1
        )

    def maskless_backward_step(
        self, states: TensorType["B", "D"], actions: TensorType["B", torch.long]
    ) -> None:
        states.scatter_(-1, actions.unsqueeze(-1).fmod(self.ndim), -1)

    def reward(self, final_states: States) -> TensorType["B", torch.float]:
        raw_states = final_states.states_tensor
        canonical = 2 * raw_states - 1
        return torch.exp(-self.energy(canonical))

    def get_states_indices(self, states: States) -> TensorType["B", torch.long]:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2, then we convert to base 3"""
        states_raw = states.states_tensor
        canonical_base = 3 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        return (states_raw + 1).mul(canonical_base).sum(-1).long()

    def get_terminating_states_indices(
        self, states: States
    ) -> TensorType["B", torch.long]:
        states_raw = states.states_tensor
        canonical_base = 2 ** torch.arange(self.ndim - 1, -1, -1, device=self.device)
        return (states_raw).mul(canonical_base).sum(-1).long()

    @property
    def n_states(self) -> int:
        return 3**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return 2**self.ndim

    @property
    def all_states(self) -> States:
        # This is brute force !
        digits = torch.arange(3, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        all_states = all_states - 1
        return self.States(all_states)

    @property
    def terminating_states(self) -> States:
        digits = torch.arange(2, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        true_dist = self.reward(self.terminating_states)
        return true_dist / true_dist.sum()

    @property
    def log_partition(self) -> float:
        rewards = self.reward(self.terminating_states)
        return torch.log(rewards.sum()).item()
