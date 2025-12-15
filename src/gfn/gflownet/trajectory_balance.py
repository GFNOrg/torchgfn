"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

import math
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
from gfn.utils.prob_calculations import get_trajectory_pfs


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
        constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
            is therefore always 1. Must be set explicitly by user to ensure that pb
            is an Estimator except under this special case.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        init_logZ: float = 0.0,
        constant_pb: bool = False,
        log_reward_clip_min: float = -float("inf"),
    ):
        """Initializes a TBGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            logZ: A learnable parameter or a ScalarEstimator instance (for
                conditional GFNs).
            init_logZ: The initial value for the logZ parameter (used if logZ is None).
            constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
                is therefore always 1. Must be set explicitly by user to ensure that pb
                is an Estimator except under this special case.
            log_reward_clip_min: If finite, clips log rewards to this value.
        """
        super().__init__(
            pf, pb, constant_pb=constant_pb, log_reward_clip_min=log_reward_clip_min
        )

        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))

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

        # If the conditions values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditions)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)
        scores = (scores + logZ.squeeze()).pow(2)
        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class RelativeTrajectoryBalanceGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet for the Relative Trajectory Balance (RTB) loss.

    This objective matches a posterior sampler to a prior diffusion (or other
    sequential) model by minimizing

    .. math::

        \left(\log Z_\phi + \log p_\phi(\tau) - \log p_\theta(\tau)
              - \beta \log r(x_T)\right)^2,

    where :math:`p_\theta` is a fixed prior process, :math:`p_\phi` is the
    learnable posterior, :math:`r` is a positive reward/constraint on the
    terminal state :math:`x_T`, and :math:`\log Z_\phi` is a learned scalar
    normalizer.
    """

    def __init__(
        self,
        pf: Estimator,
        prior_pf: Estimator,
        *,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        init_logZ: float = 0.0,
        beta: float = 1.0,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
    ):
        """Initializes an RTB GFlowNet.

        Args:
            pf: Posterior forward policy estimator :math:`p_\\phi`.
            prior_pf: Fixed prior forward policy estimator :math:`p_\\theta`.
            logZ: Learnable log-partition parameter or ScalarEstimator for
                conditional settings. Defaults to a scalar parameter.
            init_logZ: Initial value for logZ if ``logZ`` is None.
            beta: Optional scaling applied to the terminal log-reward.
            log_reward_clip_min: If finite, clips terminal log-rewards.
            debug: if True, enables extra checks at the cost of execution speed.
        """
        super().__init__(
            pf=pf,
            pb=None,
            constant_pb=True,
            log_reward_clip_min=log_reward_clip_min,
        )
        self.prior_pf = prior_pf
        self.beta = beta
        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))
        self.debug = debug  # TODO: to be passed to base classes.

    def logz_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns named parameters containing 'logZ'."""
        return {k: v for k, v in dict(self.named_parameters()).items() if "logZ" in k}

    def logz_parameters(self) -> list[torch.Tensor]:
        """Returns parameters containing 'logZ'."""
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    def _prior_log_pf(
        self,
        trajectories: Trajectories,
        *,
        fill_value: float = 0.0,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """Computes prior forward log-probs along provided trajectories."""
        # The prior is fixed; evaluate it without tracking gradients to keep its
        # parameters out of the RTB optimization graph.
        with torch.no_grad():
            log_pf = get_trajectory_pfs(
                self.prior_pf,
                trajectories,
                fill_value=fill_value,
                recalculate_all_logprobs=recalculate_all_logprobs,
            )
        return log_pf.sum(dim=0)

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the RTB loss on a batch of trajectories."""
        del env  # unused
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)

        # Posterior log-probs (forward; backward ignored in RTB score).
        log_pf_post = self.trajectory_log_probs_forward(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )
        if self.debug:
            assert log_pf_post is not None

        total_log_pf_post = log_pf_post.sum(dim=0)

        # Prior log-probs along the same trajectories.
        total_log_pf_prior = self._prior_log_pf(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

        log_rewards = trajectories.log_rewards
        if self.debug:
            assert log_rewards is not None
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)  # type: ignore

        if trajectories.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditions)
        else:
            logZ = self.logZ
        logZ = cast(torch.Tensor, logZ).squeeze()

        scores = (
            logZ + total_log_pf_post - total_log_pf_prior - self.beta * log_rewards.squeeze()  # type: ignore
        ).pow(2)
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
        constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
            is therefore always 1. Must be set explicitly by user to ensure that pb
            is an Estimator except under this special case.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

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
