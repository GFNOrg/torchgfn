from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.parametrizations import Parametrization

from ..containers import States, Trajectories, Transitions

# Typing
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]


class Loss(ABC):
    "Abstract Base Class for all GFN Losses"

    def __init__(self, parametrization: Parametrization):
        self.parametrization = parametrization

    @abstractmethod
    def __call__(self, *args, **kwargs) -> TensorType[0, float]:
        pass


class EdgeDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, edges: Transitions) -> TensorType[0, float]:
        pass


class StateDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, states: States) -> TensorType[0, float]:
        pass


class TrajectoryDecomposableLoss(Loss, ABC):
    def get_pfs_and_pbs(
        self, trajectories: Trajectories, fill_value: float = 0.0
    ) -> Tuple[LogPTrajectoriesTensor, LogPTrajectoriesTensor]:
        # fill value is the value used for invalid states (sink state usually)
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        valid_actions = trajectories.actions[trajectories.actions != -1]

        # uncomment next line for debugging
        # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions == -1)

        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")
        valid_pf_logits = self.actions_sampler.get_logits(valid_states)
        valid_log_pf_all = valid_pf_logits.log_softmax(dim=-1)
        valid_log_pf_actions = torch.gather(
            valid_log_pf_all, dim=-1, index=valid_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_pf_trajectories = torch.full_like(
            trajectories.actions, fill_value=fill_value, dtype=torch.float
        )
        log_pf_trajectories[trajectories.actions != -1] = valid_log_pf_actions

        valid_pb_logits = self.backward_actions_sampler.get_logits(
            valid_states[~valid_states.is_initial_state]
        )
        valid_log_pb_all = valid_pb_logits.log_softmax(dim=-1)
        non_exit_valid_actions = valid_actions[
            valid_actions != trajectories.env.n_actions - 1
        ]
        valid_log_pb_actions = torch.gather(
            valid_log_pb_all, dim=-1, index=non_exit_valid_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_pb_trajectories = torch.full_like(
            trajectories.actions, fill_value=fill_value, dtype=torch.float
        )
        log_pb_trajectories_slice = torch.full_like(
            valid_actions, fill_value=fill_value, dtype=torch.float
        )
        log_pb_trajectories_slice[
            valid_actions != trajectories.env.n_actions - 1
        ] = valid_log_pb_actions
        log_pb_trajectories[trajectories.actions != -1] = log_pb_trajectories_slice

        return log_pf_trajectories, log_pb_trajectories

    @abstractmethod
    def __call__(self, trajectories: Trajectories) -> TensorType[0, float]:
        pass
