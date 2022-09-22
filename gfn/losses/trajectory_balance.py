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
        on_policy: bool = False,
    ):
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)
        self.on_policy = on_policy

    def get_scores(
        self, trajectories: Trajectories
    ) -> Tuple[ScoresTensor, ScoresTensor, ScoresTensor]:

        if self.on_policy:
            assert trajectories.log_pbs is not None
            log_pf_trajectories = trajectories.log_pfs
            log_pb_trajectories = trajectories.log_pbs
        else:
            log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
                trajectories
            )

            log_pf_trajectories = log_pf_trajectories.sum(dim=0)
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
