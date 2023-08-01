"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

import torch
from torchtyping import TensorType as TT
import torch.nn as nn

from gfn.containers import Trajectories
from gfn.gflownet.base import TrajectoryBasedGFlowNet


class TBGFlowNet(TrajectoryBasedGFlowNet):
    r"""Holds the logZ estimate for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.logit_PB is a fixed DiscretePBEstimator.

    Attributes:
        logZ: a LogZEstimator instance.
        log_reward_clip_min: minimal value to clamp the reward to.

    """

    def __init__(
        self,
        init_logZ: float = 0.0,
        log_reward_clip_min: float = -12,  # roughly log(1e-5)
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logZ = nn.Parameter(torch.tensor(init_logZ))
        self.log_reward_clip_min = log_reward_clip_min

    def loss(self, trajectories: Trajectories) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.logZ).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


class LogPartitionVarianceGFlowNet(TrajectoryBasedGFlowNet):
    """Dataclass which holds the logZ estimate for the Log Partition Variance loss.

    Attributes:
        log_reward_clip_min: minimal value to clamp the reward to.

    Raises:
        ValueError: if the loss is NaN.
    """
    def __init__(self, log_reward_clip_min: float = -12, **kwargs):
        super().__init__(**kwargs)

        self.log_reward_clip_min = log_reward_clip_min  # -12 is roughly log(1e-5)

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
