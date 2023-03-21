"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

from dataclasses import dataclass

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
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
    ):
        """Loss object to evaluate the TB loss on a batch of trajectories.

        Args:
            log_reward_clip_min (float, optional): minimal value to clamp the reward to. Defaults to -12 (roughly log(1e-5)).
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Which should be faster than
                                        reevaluating them. Defaults to False.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )
        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.parametrization.logZ.tensor).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


class LogPartitionVarianceLoss(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: PFBasedParametrization,
        log_reward_clip_min: float = -12,
        on_policy: bool = False,
    ):
        """Loss object to evaluate the Log Partition Variance Loss (Section 3.2 of
        [ROBUST SCHEDULING WITH GFLOWNETS](https://arxiv.org/abs/2302.05446))

        Args:
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Which should be faster than
                                        reevaluating them. Defaults to False.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.actions_sampler = DiscreteActionsSampler(parametrization.logit_PF)
        self.backward_actions_sampler = BackwardDiscreteActionsSampler(
            parametrization.logit_PB
        )

        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores - scores.mean()).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss
