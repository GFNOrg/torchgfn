"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

import warnings
from typing import cast

import torch
import torch.nn as nn

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import Estimator, ScalarEstimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.utils.handlers import (
    is_callable_exception_handler,
    warn_about_recalculating_logprobs,
)


class TBGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet for the Trajectory Balance loss.

    $\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3$, where
    $\mathcal{O}_1 = \mathbb{R}$ represents the possible values for logZ,
    and $\mathcal{O}_2$ is the set of forward probability functions consistent with the
    DAG. $\mathcal{O}_3$ is the set of backward probability functions consistent with
    the DAG, or a singleton thereof, if self.pb is a fixed DiscretePBEstimator.

    See [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)
    for more details.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        logZ: A learnable parameter or a ScalarEstimator instance (for conditional GFNs).
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        init_logZ: float = 0.0,
        log_reward_clip_min: float = -float("inf"),
        dag_is_tree: bool = False,
    ):
        """Initializes a TBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            logZ: A learnable parameter or a ScalarEstimator instance (for
                conditional GFNs).
            init_logZ: The initial value for the logZ parameter (used if logZ is None).
            log_reward_clip_min: If finite, clips log rewards to this value.
            dag_is_tree: Whether the gflownet DAG is a tree, and pb is therefore always
                1. Must be set explicitly by user to ensure that pb is an Estimator
                except under this special case.
        """
        if pb is None and not dag_is_tree:
            raise ValueError(
                "pb must be an Estimator unless dag_is_tree is True. "
                "If the gflownet DAG is a tree, set dag_is_tree to True."
            )
        if isinstance(pb, Estimator) and dag_is_tree:
            warnings.warn(
                "The user specified that the GFlowNet DAG is a tree, and specified a "
                "backward policy estimator. Under normal circumstances, pb should be "
                "None if the GFlowNet DAG is a tree, because the backward policy "
                "probability is always 1, and therefore learning a backward policy "
                "estimator is not necessary and will slow down training. Please ensure "
                "this is the intended experimental setup."
            )

        super().__init__(pf, pb)

        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))
        self.log_reward_clip_min = log_reward_clip_min

    def logz_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of named parameters containing 'logZ' in their name.

        Returns:
            A dictionary of named parameters containing 'logZ' in their name.
        """
        return {k: v for k, v in dict(self.named_parameters()).items() if "logZ" in k}

    def logz_parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters containing 'logZ' in their name.

        Returns:
            A list of parameters containing 'logZ' in their name.
        """
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the trajectory balance loss.

        The trajectory balance loss is described in section 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed trajectory balance loss as a tensor. The shape depends on the
            reduction method.
        """
        del env  # unused
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # If the conditioning values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.conditioning is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditioning)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)
        scores = (scores + logZ.squeeze()).pow(2)
        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class LogPartitionVarianceGFlowNet(TrajectoryBasedGFlowNet):
    """GFlowNet for the Log Partition Variance loss.

    The log partition variance loss is described in section 3.2 of
    [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator,
        log_reward_clip_min: float = -float("inf"),
    ):
        """Initializes a LogPartitionVarianceGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator.
            log_reward_clip_min: If finite, clips log rewards to this value.
        """
        super().__init__(pf, pb)
        self.log_reward_clip_min = log_reward_clip_min

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the log partition variance loss.

        The log partition variance loss is described in section 3.2 of
        [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed log partition variance loss as a tensor. The shape depends on
            the reduction method.
        """
        del env  # unused
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        scores = (scores - scores.mean()).pow(2)
        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss
