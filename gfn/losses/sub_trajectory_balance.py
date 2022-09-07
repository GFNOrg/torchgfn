from typing import Optional, Tuple

import torch
from torchtyping import TensorType

from gfn.containers import Trajectories, States
from gfn.losses.base import TrajectoryDecomposableLoss
from gfn.parametrizations import DBParametrization
from gfn.samplers.actions_samplers import LogitPBActionsSampler, LogitPFActionsSampler

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]


class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: DBParametrization,
        hubs: Optional[States] = None,
        reward_clip_min: float = 1e-5,
    ):
        self.parametrization = parametrization
        self.reward_clip_min = reward_clip_min
        self.actions_sampler = LogitPFActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = LogitPBActionsSampler(parametrization.logit_PB)
        self.hubs = hubs

    def get_sub_trajectories(self, trajectories: Trajectories) -> Trajectories:


    def __call__(self, trajectories: Trajectories) -> LossTensor:
        scores = self._compute_scores(trajectories)
        return self._compute_loss(scores)

    def _compute_scores(self, trajectories: Trajectories) -> ScoresTensor:
        scores = torch.zeros(len(trajectories), device=self.device)
        for i, trajectory in enumerate(trajectories):
            for sub_trajectory in trajectory:
                scores[i] += self._compute_score(sub_trajectory)
        return scores

    def _compute_score(self, sub_trajectory: Trajectories) -> float: