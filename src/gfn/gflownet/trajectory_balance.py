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
from gfn.gflownet.losses import RegressionLoss
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
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
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
            debug: If True, keep runtime safety checks active; disable in compiled runs.
            loss_fn: Regression loss applied to balance residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss`.
        """
        super().__init__(
            pf,
            pb,
            constant_pb=constant_pb,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
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
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # If the conditions values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.states.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                # Guard isinstance behind debug: logZ type is fixed at __init__ time,
                # but isinstance checks cause graph breaks in torch.compile.
                if self.debug:
                    assert isinstance(self.logZ, ScalarEstimator)
                # cast: when conditions exist, logZ is always a ScalarEstimator (set at init).
                logZ = cast(ScalarEstimator, self.logZ)(
                    trajectories.states.conditions[0]
                )  # [N] or [..., 1]
        else:
            logZ = self.logZ  # []

        logZ = cast(torch.Tensor, logZ).squeeze()  # [] or [N]
        scores = self.loss_fn(scores + logZ)  # [N]
        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class RelativeTBBase(TrajectoryBasedGFlowNet):
    r"""Shared base for Relative Trajectory Balance variants.

    Manages the prior forward policy and ``beta`` scaling.  Subclasses only
    need to implement :meth:`loss` (deciding how to handle ``logZ`` and
    reduction).
    """

    def __init__(
        self,
        pf: Estimator,
        prior_pf: Estimator,
        *,
        beta: float = 1.0,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ):
        super().__init__(
            pf=pf,
            pb=None,
            constant_pb=True,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )
        # Store the prior as a plain attribute (not an nn.Module submodule)
        # so that its parameters don't leak into self.parameters() /
        # self.pf_pb_named_parameters().  The prior is frozen and evaluated
        # under torch.no_grad() in loss(); registering it would silently
        # include its weights in optimizer state dicts and checkpoints.
        object.__setattr__(self, "_prior_pf", prior_pf)
        self.register_buffer("beta", torch.tensor(beta))

    @property
    def prior_pf(self) -> Estimator:
        """The fixed prior forward policy (not registered as a submodule)."""
        return self._prior_pf  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Shared score computation
    # ------------------------------------------------------------------

    def _compute_rtb_scores(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """RTB residuals: ``log_pf_post - log_pf_prior - beta * log_rewards``.

        Returns:
            Shape ``(N,)`` per-trajectory scores.
        """
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)

        # Posterior log-probs (T, N) → sum to (N,).
        log_pf_post = self.trajectory_log_probs_forward(
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        ).sum(dim=0)

        # Prior log-probs (T, N) → sum to (N,), detached.
        with torch.no_grad():
            log_pf_prior = get_trajectory_pfs(
                self.prior_pf,
                trajectories,
                fill_value=0.0,
                recalculate_all_logprobs=True,
            ).sum(dim=0)

        log_rewards = trajectories.log_rewards
        if self.debug:
            assert log_rewards is not None
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)  # type: ignore

        return log_pf_post - log_pf_prior - self.beta * log_rewards  # type: ignore


class RelativeTrajectoryBalanceGFlowNet(RelativeTBBase):
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
        loss_fn: RegressionLoss | None = None,
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
            loss_fn: Regression loss applied to balance residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss`.
        """
        super().__init__(
            pf=pf,
            prior_pf=prior_pf,
            beta=beta,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )
        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))

    def logz_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns named parameters containing 'logZ'."""
        return {k: v for k, v in dict(self.named_parameters()).items() if "logZ" in k}

    def logz_parameters(self) -> list[torch.Tensor]:
        """Returns parameters containing 'logZ'."""
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the RTB loss on a batch of trajectories."""
        scores = self._compute_rtb_scores(env, trajectories, recalculate_all_logprobs)

        # Get logZ.
        if trajectories.states.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                if self.debug:
                    assert isinstance(self.logZ, ScalarEstimator)
                logZ = cast(ScalarEstimator, self.logZ)(
                    trajectories.states.conditions[0]
                )
        else:
            logZ = self.logZ
        logZ = cast(torch.Tensor, logZ).squeeze()

        # The 0.5 factor makes the gradient equal to the residual itself
        # (d/dt [0.5·g(t)] = g'(t)/2; for g=t², this gives t instead of 2t),
        # matching the convention in Venkatraman et al. (2024).
        scores = 0.5 * self.loss_fn(scores + logZ)

        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
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
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        scores = scores.sub_(scores.mean())  # [N], in-place mean-centering.
        scores = self.loss_fn(scores)  # [N]
        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss


class RelativeLogPartitionVarianceGFlowNet(RelativeTBBase):
    r"""RTB variant that eliminates the learned logZ via variance minimization.

    Analogous to how :class:`LogPartitionVarianceGFlowNet` relates to
    :class:`TBGFlowNet`, this class mean-centers the RTB residuals within each
    batch so that no explicit ``logZ`` parameter is needed.

    The loss minimizes

    .. math::

        \operatorname{Var}_{\tau}\!\bigl[\log p_\phi(\tau)
              - \log p_\theta(\tau) - \beta\,\log r(x_T)\bigr],

    which equals the RTB loss evaluated at the batch-optimal
    :math:`\log Z^* = -\overline{s}` (the negative batch mean of scores).
    """

    def __init__(
        self,
        pf: Estimator,
        prior_pf: Estimator,
        *,
        beta: float = 1.0,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ):
        """Initializes a Relative LPV GFlowNet.

        Args:
            pf: Posterior forward policy estimator :math:`p_\\phi`.
            prior_pf: Fixed prior forward policy estimator :math:`p_\\theta`.
            beta: Scaling applied to the terminal log-reward.
            log_reward_clip_min: If finite, clips terminal log-rewards.
            debug: If True, enables extra checks at the cost of execution speed.
            loss_fn: Regression loss applied to balance residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss`.
        """
        super().__init__(
            pf=pf,
            prior_pf=prior_pf,
            beta=beta,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the Relative LPV loss on a batch of trajectories."""
        scores = self._compute_rtb_scores(env, trajectories, recalculate_all_logprobs)
        scores = scores - scores.mean()
        # The 0.5 factor makes the gradient equal to the residual itself;
        # see RelativeTrajectoryBalanceGFlowNet.loss() for details.
        scores = 0.5 * self.loss_fn(scores)

        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss
