"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""
from dataclasses import dataclass

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.estimators import LogZEstimator
from gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss


@dataclass
class TBParametrization(PFBasedParametrization, TrajectoryDecomposableLoss):
    r"""Dataclass which holds the logZ estimate for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.logit_PB is a fixed DiscretePBEstimator.

    Attributes:
        logZ: a LogZEstimator instance.
        on_policy: boolean indicating whether we need to reevaluate the log probs.
        log_reward_clip_min: minimal value to clamp the reward to.

    """
    logZ: LogZEstimator
    on_policy: bool = False
    log_reward_clip_min: float = -12  # roughly log(1e-5)

    def loss(self, trajectories: Trajectories) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))
        """
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.logZ.tensor).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


@dataclass
class LogPartitionVarianceParametrization(
    PFBasedParametrization, TrajectoryDecomposableLoss
):
    """Dataclass which holds the logZ estimate for the Log Partition Variance loss.

    Attributes:
        on_policy: boolean indicating whether we need to reevaluate the log probs.
        log_reward_clip_min: minimal value to clamp the reward to.
    """

    on_policy: bool = False
    log_reward_clip_min: float = -12  # roughly log(1e-5)

    def loss(self, trajectories: Trajectories) -> TT[0, float]:
        """Log Partition Variance loss.

        This method is described in section 3.2 of
        [ROBUST SCHEDULING WITH GFLOWNETS](https://arxiv.org/abs/2302.05446))
        """
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores - scores.mean()).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is NaN.")

        return loss
