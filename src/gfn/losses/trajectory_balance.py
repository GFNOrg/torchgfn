from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories
from gfn.estimators import LogZEstimator
from gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]


@dataclass
class TBParametrization(PFBasedParametrization):
    r"""
    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the DAG.
    $\mathcal{O}_3$ is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Trajectory Balance Loss.
    """
    logZ: LogZEstimator


class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        reward_clip_min: float = 1e-5,
        on_policy: bool = False,
    ):
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
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
