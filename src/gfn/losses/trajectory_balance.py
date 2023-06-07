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
from gfn.samplers import ActionsSampler


@dataclass
class TBParametrization(PFBasedParametrization):
    r"""Dataclass which holds the logZ estimate for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.logit_PB is a fixed DiscretePBEstimator.
    """
    logZ: LogZEstimator


# TODO: rename to TrajectoryBalanceLoss.
# TODO: Should this loss live within the Parameterization, as a method?
class TrajectoryBalance(TrajectoryDecomposableLoss):
    """Loss object to evaluate the TB loss on a batch of trajectories.

    This method is described in section 2.3 of [this paper](https://arxiv.org/abs/2209.12782))

    Attributes:
        parametrization: a TBParamaterization (Trajectory Balance) instance.
        log_reward_clip_min: minimal value to clamp rewards to.
        on_policy: stores whether log probabilities stored in trajectories are used.
    """

    def __init__(
        self,
        parametrization: TBParametrization,
        log_reward_clip_min: float = -12,  # roughly log(1e-5)
        on_policy: bool = False,
    ):
        """Instantiates a TrajectoryBalance instance.

        Args:
            parametrization: a TBParamaterization instance.
            log_reward_clip_min): minimal value to clamp the reward to.
            on_policy: If True, the log probs stored in the trajectories are used,
                which should be faster than reevaluating them.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> TT[0, float]:
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores + self.parametrization.logZ.tensor).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss


# TODO: All methods should have a nice reference like this.
# TODO: Should this loss live within the Parameterization, as a method?
class LogPartitionVarianceLoss(TrajectoryDecomposableLoss):
    """Loss object to evaluate the Log Partition Variance Loss.

    This method is described in section 3.2 of
    [ROBUST SCHEDULING WITH GFLOWNETS](https://arxiv.org/abs/2302.05446))

    Attributes:
        parametrization: a PFBasedParameterization instance.
        log_reward_clip_min: Threshold below which all log rewards will be clamped.
        actions_sampler: Forward policy actions sampler.
        backward_actions_sampler: Backward policy actions sampler.
    """

    def __init__(
        self,
        parametrization: PFBasedParametrization,
        log_reward_clip_min: float = -12,  # roughly log(1e-5)
        on_policy: bool = False,
    ):
        """Instantiate the LogPartitionVarianceLoss given a paramaterization.

        Args:
            parametrization: a PFBasedParameterization instance.
            log_reward_clip_min: Threshold below which all log rewards will be clipped
                to 0.
            on_policy: If True, the log probs stored in the trajectories are used.
                Which should be faster than reevaluating them.
        """
        self.parametrization = parametrization
        self.log_reward_clip_min = log_reward_clip_min
        self.on_policy = on_policy

    def __call__(self, trajectories: Trajectories) -> TT[0, float]:
        """Given a batch of trajectories, return a batch of losses.

        Args:
            trajectories: a batch of trajectories.

        Returns: a batch of losses.

        Raises:
            ValueError: If ever the loss is NaN.
        """
        _, _, scores = self.get_trajectories_scores(trajectories)
        loss = (scores - scores.mean()).pow(2).mean()
        if torch.isnan(loss):
            raise ValueError("loss is NaN.")

        return loss
