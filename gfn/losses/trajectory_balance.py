from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.parametrizations import TBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]


class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        reward_clip_min: float = 1e-5,
    ):
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)

    def get_scores(
        self, trajectories: Trajectories
    ) -> Tuple[ScoresTensor, ScoresTensor, ScoresTensor]:

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
        log_pf_trajectories = torch.zeros_like(trajectories.actions, dtype=torch.float)
        log_pf_trajectories[trajectories.actions != -1] = valid_log_pf_actions

        log_pf_trajectories = log_pf_trajectories.sum(dim=0)

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
        log_pb_trajectories = torch.zeros_like(trajectories.actions, dtype=torch.float)
        log_pb_trajectories_slice = torch.zeros_like(valid_actions, dtype=torch.float)
        log_pb_trajectories_slice[
            valid_actions != trajectories.env.n_actions - 1
        ] = valid_log_pb_actions
        log_pb_trajectories[trajectories.actions != -1] = log_pb_trajectories_slice

        log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        rewards = trajectories.rewards
        log_rewards = torch.log(rewards.clamp_min(self.reward_clip_min))  # type: ignore

        return (
            log_pf_trajectories,
            log_pb_trajectories,
            log_pf_trajectories - log_pb_trajectories - log_rewards,
        )

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_scores(trajectories)
        loss = (scores + self.parametrization.logZ.tensor).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
