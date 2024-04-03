"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.base import TrajectoryBasedGFlowNet
from gfn.modules import GFNModule


class TBGFlowNet(TrajectoryBasedGFlowNet):
    r"""Holds the logZ estimate for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.logit_PB is a fixed DiscretePBEstimator.

    Attributes:
        logZ: a LogZEstimator instance.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        init_logZ: float = 0.0,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb)

        self.logZ = nn.Parameter(
            torch.tensor(init_logZ)
        )  # TODO: Optionally, this should be a nn.Module to support conditional GFNs.
        self.log_reward_clip_min = log_reward_clip_min

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores = self.get_trajectories_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        loss = (scores + self.logZ).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


class LogPartitionVarianceGFlowNet(TrajectoryBasedGFlowNet):
    """Dataclass which holds the logZ estimate for the Log Partition Variance loss.

    Attributes:
        log_reward_clip_min: If finite, clips log rewards to this value.

    Raises:
        ValueError: if the loss is NaN.
    """

    def __init__(
        self,
        pf: GFNModule,
        pb: GFNModule,
        log_reward_clip_min: float = -float("inf"),
    ):
        super().__init__(pf, pb)
        self.log_reward_clip_min = log_reward_clip_min

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> TT[0, float]:
        """Log Partition Variance loss.

        This method is described in section 3.2 of
        [ROBUST SCHEDULING WITH GFLOWNETS](https://arxiv.org/abs/2302.05446))
        """
        del env  # unused
        _, _, scores = self.get_trajectories_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        loss = (scores - scores.mean()).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is NaN.")

        return loss
