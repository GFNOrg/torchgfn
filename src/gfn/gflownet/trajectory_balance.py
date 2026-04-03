"""
Implementations of the [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)
and the [Log Partition Variance loss](https://arxiv.org/abs/2302.05446).
"""

import math
from logging import warning
from typing import cast

import torch
import torch.nn as nn

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import Estimator, ScalarEstimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.gflownet.losses import HalfSquaredLoss, RegressionLoss
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

    # logz_named_parameters() and logz_parameters() are inherited from
    # TrajectoryBasedGFlowNet — they filter self.named_parameters() for 'logZ'.

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the trajectory balance loss.

        The trajectory balance loss is described in section 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                Useful for intrinsic rewards (see
                "Towards Improving Exploration through Sibling Augmented
                GFlowNets", Madan et al., ICLR 2025).

        Returns:
            The computed trajectory balance loss as a tensor. The shape depends on the
            reduction method.
        """
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
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
        """Initializes the shared RTB base.

        Args:
            pf: Posterior forward policy estimator (trainable).
            prior_pf: Fixed prior forward policy estimator.  All parameters
                must have ``requires_grad=False``; the constructor enforces
                this and raises if any are trainable.
            beta: Reward scaling factor.
            log_reward_clip_min: If finite, clips terminal log-rewards.
            debug: If True, enables extra runtime checks.
            loss_fn: Regression loss applied to balance residuals.
        """
        super().__init__(
            pf=pf,
            pb=None,
            constant_pb=True,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )

        # Enforce that the prior is frozen.  A trainable prior would silently
        # receive gradients through the RTB loss (despite torch.no_grad() on
        # the forward pass) if its parameters were accidentally shared.
        trainable = [n for n, p in prior_pf.named_parameters() if p.requires_grad]
        if trainable:
            raise ValueError(
                f"prior_pf has {len(trainable)} trainable parameter(s) "
                f"(first: {trainable[0]!r}).  Freeze all prior parameters with "
                f"`for p in prior_pf.parameters(): p.requires_grad_(False)` "
                f"before constructing the RTB objective."
            )

        # Store the prior as a plain attribute (not an nn.Module submodule)
        # so that its parameters don't leak into self.parameters() /
        # self.pf_pb_named_parameters().  The prior is frozen and evaluated
        # under torch.no_grad() in _compute_rtb_scores(); registering it as a
        # submodule would silently include its weights in optimizer state dicts
        # and checkpoints.
        object.__setattr__(self, "_prior_pf", prior_pf)
        if beta == 0.0:
            import warnings

            warnings.warn(
                "beta=0 makes the RTB loss insensitive to rewards. "
                "This is valid but unusual — verify this is intended.",
                stacklevel=2,
            )
        self.register_buffer("beta", torch.tensor(beta))

    @property
    def prior_pf(self) -> Estimator:
        """The fixed prior forward policy (not registered as a submodule)."""
        return self._prior_pf  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Shared score computation
    # ------------------------------------------------------------------

    def get_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        env: Env | None = None,
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """RTB residuals (without logZ): ``log_pf_post - log_pf_prior - beta * log_R``.

        This is the public interface to the RTB balance residuals, analogous to
        :meth:`TrajectoryBasedGFlowNet.get_scores` for standard TB.

        Returns:
            Shape ``(N,)`` per-trajectory scores.
        """
        return self._compute_rtb_scores(
            env, trajectories, log_rewards=log_rewards, recalculate_all_logprobs=recalculate_all_logprobs
        )

    def _compute_rtb_scores(
        self,
        env: Env | None,
        trajectories: Trajectories,
        log_rewards: torch.Tensor | None = None,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """RTB residuals: ``log_pf_post - log_pf_prior - beta * log_rewards``.

        Args:
            env: The environment (unused, kept for API consistency).
            trajectories: The Trajectories object to evaluate.
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                Useful for intrinsic rewards (see
                "Towards Improving Exploration through Sibling Augmented
                GFlowNets", Madan et al., ICLR 2025).
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

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
        # Always recalculate: the prior is frozen so there are no cached
        # logprobs to reuse, and we need fresh evaluations under the prior.
        with torch.no_grad():
            log_pf_prior = get_trajectory_pfs(
                self.prior_pf,
                trajectories,
                recalculate_all_logprobs=True,
            ).sum(dim=0)

        if log_rewards is None:
            if self.debug:
                assert trajectories.log_rewards is not None
            log_rewards = cast(torch.Tensor, trajectories.log_rewards)
        if self.debug:
            assert log_rewards.shape == (trajectories.batch_size,)
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

        return log_pf_post - log_pf_prior - cast(torch.Tensor, self.beta) * log_rewards


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
                Defaults to :class:`~gfn.gflownet.losses.HalfSquaredLoss`.
        """
        super().__init__(
            pf=pf,
            prior_pf=prior_pf,
            beta=beta,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn or HalfSquaredLoss(),
        )
        self.logZ = logZ or nn.Parameter(torch.tensor(init_logZ))

    # logz_named_parameters() and logz_parameters() are inherited from
    # TrajectoryBasedGFlowNet — they filter self.named_parameters() for 'logZ'.

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the RTB loss on a batch of trajectories."""
        scores = self._compute_rtb_scores(
            env,
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

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

        scores = self.loss_fn(scores + logZ)

        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class TrustPCLGFlowNet(RelativeTrajectoryBalanceGFlowNet):
    r"""Trust-PCL view of Relative Trajectory Balance.

    Deleu et al. (2025) proved that RTB is mathematically equivalent to
    Trust-PCL, an off-policy RL method with KL regularization toward a
    reference policy.  This class provides an **RL-native interface** to
    the same algorithm, using reinforcement learning terminology.

    The equivalence (Proposition 3.1 of Deleu et al.):

    .. math::

        \mathcal{L}_{\text{Trust-PCL}}(\phi, \psi)
            = \alpha^2 \,\mathcal{L}_{\text{RTB}}(\phi, \psi)

    where :math:`\alpha = 1/\beta` is the Trust-PCL temperature.

    **Parameter correspondence:**

    +---------------------+------------------------------+---------------------------+
    | Concept             | RTB name                     | Trust-PCL name            |
    +=====================+==============================+===========================+
    | Temperature          | ``beta``                     | ``alpha = 1/beta``        |
    +---------------------+------------------------------+---------------------------+
    | Learned scalar       | ``logZ``                     | ``v_soft_s0 = alpha*logZ``|
    +---------------------+------------------------------+---------------------------+
    | Trainable model      | ``pf`` (posterior)           | ``policy``                |
    +---------------------+------------------------------+---------------------------+
    | Fixed reference      | ``prior_pf``                 | ``reference_policy``      |
    +---------------------+------------------------------+---------------------------+

    **Interpretation of the learned scalar:**

    In RTB, ``logZ`` estimates the log-partition function
    :math:`\log \int p_\theta(x)\,r(x)\,dx`.  In Trust-PCL, the same
    quantity is the **soft value function** at the initial state:
    :math:`V^{\text{soft}}_\psi(s_0) = \alpha \cdot \log Z_\psi`.
    This connects GFlowNet training to entropy-regularized RL, where the
    soft value satisfies the soft Bellman equation.

    **Why this class exists:**

    The underlying computation is identical to
    :class:`RelativeTrajectoryBalanceGFlowNet` (the loss is just scaled by
    :math:`\alpha^2`).  This class exists to:

    1. Provide an RL-native constructor (``policy``, ``reference_policy``,
       ``alpha``, ``init_v_soft_s0``) for researchers familiar with
       Trust-PCL / SAC / entropy-regularized RL.
    2. Expose :attr:`alpha` and :attr:`v_soft_s0` properties for
       interpretability and monitoring.
    3. Serve as a pedagogical bridge between the GFlowNet and RL communities.

    References:
        Deleu et al. "Relative Trajectory Balance is equivalent to
        Trust-PCL" (2025, arXiv:2509.01632).

        Nachum et al. "Trust-PCL: An Off-Policy Trust Region Method for
        Continuous Control" (NeurIPS 2017, arXiv:1707.01891).

        Venkatraman et al. "Amortizing intractable inference in diffusion
        models for vision, language, and control" (NeurIPS 2024,
        arXiv:2405.20971).
    """

    def __init__(
        self,
        policy: Estimator,
        reference_policy: Estimator,
        *,
        alpha: float = 1.0,
        init_v_soft_s0: float = 0.0,
        logZ: nn.Parameter | ScalarEstimator | None = None,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ):
        """Initializes a Trust-PCL GFlowNet.

        Args:
            policy: The trainable policy :math:`\\pi_\\phi` (= RTB posterior).
            reference_policy: The fixed reference policy :math:`\\pi_{\\text{ref}}`
                (= RTB prior).  Must have all parameters frozen
                (``requires_grad=False``); evaluated under ``torch.no_grad()``.
            alpha: Trust-PCL temperature (must be > 0). Controls the strength
                of KL regularization toward the reference policy. Corresponds
                to ``1/beta`` in RTB.  Higher alpha → more regularization
                (policy stays closer to reference).
            init_v_soft_s0: Initial value for the soft value function at
                :math:`s_0`.  Converted to ``logZ = v_soft_s0 / alpha``
                internally.  Mutually exclusive with ``logZ``.
            logZ: Explicit logZ parameter for advanced use (e.g. conditional
                generation with a :class:`ScalarEstimator`).  Mutually
                exclusive with ``init_v_soft_s0``.
            log_reward_clip_min: If finite, clips terminal log-rewards.
            debug: If True, enables extra runtime checks.
            loss_fn: Regression loss applied to balance residuals.
                Defaults to :class:`~gfn.gflownet.losses.HalfSquaredLoss`.

        Raises:
            ValueError: If ``alpha <= 0`` or if both ``logZ`` and a non-default
                ``init_v_soft_s0`` are provided.
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if logZ is not None and init_v_soft_s0 != 0.0:
            raise ValueError(
                "logZ and init_v_soft_s0 are mutually exclusive. "
                "Pass logZ directly for conditional generation or advanced use; "
                "otherwise set init_v_soft_s0 and let the constructor derive logZ."
            )
        beta = 1.0 / alpha
        init_logZ = (
            init_v_soft_s0 * beta
        )  # v_soft = alpha * logZ → logZ = v / alpha = v * beta

        if not isinstance(logZ, type(None)) and init_logZ != 0.0:
            # If logZ is explicitly provided, we ignore init_v_soft_s0 and use logZ directly.
            warning.warn(
                "TrustPCLGFlowNet's init_v_soft_s0 is ignored because logZ is explicitly provided. Ensure this is intentional."
            )

        super().__init__(
            pf=policy,
            prior_pf=reference_policy,
            beta=beta,
            init_logZ=init_logZ,
            logZ=logZ,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )

    @property
    def alpha(self) -> torch.Tensor:
        r"""Trust-PCL temperature: :math:`\alpha = 1/\beta`.

        Controls the strength of KL regularization toward the reference
        policy. At convergence, the learned policy satisfies:

        .. math::

            \pi_\phi(a|s) \propto \pi_{\text{ref}}(a|s)
            \exp\!\bigl(Q^{\text{soft}}(s,a) / \alpha\bigr)

        Higher alpha → policy stays closer to the reference (more
        regularization). Lower alpha → policy deviates more toward
        reward-maximizing behavior.
        """
        return 1.0 / cast(torch.Tensor, self.beta)

    @property
    def v_soft_s0(self) -> torch.Tensor:
        r"""Soft value function at the initial state: :math:`V^{\text{soft}}_\psi(s_0) = \alpha \cdot \log Z_\psi`.

        This is the expected return under the optimal entropy-regularized
        policy, starting from :math:`s_0`:

        .. math::

            V^{\text{soft}}(s_0) = \mathbb{E}_{\pi_\phi}\!\left[
                \sum_t r(s_t, a_t)
                + \alpha \sum_t \log \frac{\pi_{\text{ref}}(a_t|s_t)}
                                          {\pi_\phi(a_t|s_t)}
            \right]

        The KL regularization term :math:`\alpha \log(\pi_{\text{ref}} / \pi_\phi)`
        in the sum emerges from the ratio of prior to posterior log-probabilities
        in the RTB balance condition.

        Monitoring this value during training shows how the expected
        (regularized) return evolves.  At convergence it equals
        :math:`\alpha \log \int p_\\theta(x)\,r(x)\,dx`.
        """
        logZ = self.logZ
        if isinstance(logZ, ScalarEstimator):
            raise ValueError(
                "v_soft_s0 cannot be computed when logZ is a ScalarEstimator "
                "(conditional logZ). Compute alpha * logZ(conditions) manually "
                "via self.alpha and self.logZ."
            )
        # Detach so that accessing v_soft_s0 for monitoring does not
        # accidentally create a gradient path through logZ.
        return self.alpha * cast(torch.Tensor, logZ).detach()

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Computes the Trust-PCL loss: :math:`\alpha^2 \cdot \mathcal{L}_{\text{RTB}}`.

        The scaling by :math:`\alpha^2` is the only difference from
        :meth:`RelativeTrajectoryBalanceGFlowNet.loss`.  It ensures
        gradient magnitudes match the Trust-PCL formulation.
        """
        rtb_loss = super().loss(
            env,
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
            reduction=reduction,
        )
        return self.alpha**2 * rtb_loss


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
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the log partition variance loss.

        The log partition variance loss is described in section 3.2 of
        [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').
            log_rewards: Optional custom log rewards tensor of shape
                (n_trajectories,). When None, uses the environment rewards.
                Useful for intrinsic rewards (see
                "Towards Improving Exploration through Sibling Augmented
                GFlowNets", Madan et al., ICLR 2025).

        Returns:
            The computed log partition variance loss as a tensor. The shape depends on
            the reduction method.
        """
        if self.debug:
            warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
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
                Defaults to :class:`~gfn.gflownet.losses.HalfSquaredLoss`.
        """
        super().__init__(
            pf=pf,
            prior_pf=prior_pf,
            beta=beta,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn or HalfSquaredLoss(),
        )

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the Relative LPV loss on a batch of trajectories."""
        scores = self._compute_rtb_scores(
            env,
            trajectories,
            log_rewards=log_rewards,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )
        scores = scores - scores.mean()
        scores = self.loss_fn(scores)

        loss = loss_reduce(scores, reduction)
        if self.debug and torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss
