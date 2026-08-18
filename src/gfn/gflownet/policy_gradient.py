r"""Soft policy-gradient GFlowNet objectives: VPG and Entropic PPO.

Implements the two headline methods of Zykova-Myzina et al., "Proximal Policy
Optimization for Amortized Discrete Sampling" (2026, arXiv:2606.15793), which train the
GFlowNet forward policy as an entropy-regularized RL agent rather than by enforcing a
flow-balance condition:

- :class:`PolicyGradientGFlowNet` — Vanilla Policy Gradient (their Algorithm 1) with a
  selectable advantage estimator: the full soft return, reward-to-go, reward-to-go with a
  learned value baseline, or GAE. The paper's Section 5.2 finds the variance ordering
  matches the RL literature, with GAE best.
- :class:`EntPPOGFlowNet` — Entropic PPO (their Algorithm 2), the first PPO variant that
  works for GFlowNets. Soft policy improvement contributes an *analytic* KL trust region
  against the rollout policy, which standard PPO lacks; the paper shows that both this KL
  term (their Fig. 7) and the usual ratio clipping (their Fig. 8) are load-bearing, and
  removing either causes collapse once several update epochs are taken per batch.

The RL correspondence holds at :math:`\alpha = \gamma = 1` with a *fixed* backward policy:
changing :math:`P_B` changes the MDP reward itself, so these objectives admit no joint
forward-backward loss. To learn :math:`P_B` alongside them, use :func:`tlm_loss`
(Trajectory Likelihood Maximization) as a separate update, as the paper does.

Usage sketch (the user owns the optimizer loop, as elsewhere in torchgfn)::

    gflownet = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, gae_lambda=0.7)
    for it in range(n_iterations):
        traj = sampler.sample_trajectories(env, n=16, save_estimator_outputs=True)
        batch = gflownet.to_training_samples(traj)   # freezes advantages / targets
        for _ in range(K):                           # inner update epochs
            for split in batch_splits(batch, S):
                policy_opt.zero_grad()
                gflownet.policy_loss(split).backward()
                policy_opt.step()
                value_opt.zero_grad()
                gflownet.value_loss(env, split).backward()
                value_opt.step()

See ``tutorials/examples/train_hypergrid_ppo.py`` for a complete script.
"""

from __future__ import annotations

from typing import Literal, Tuple, cast

import torch
from torch.distributions import Distribution, kl_divergence

from gfn.containers import PolicyGradientTrajectories, Trajectories
from gfn.env import Env
from gfn.estimators import (
    ConditionalScalarEstimator,
    Estimator,
    PolicyEstimatorProtocol,
    ScalarEstimator,
    validate_policy_estimator,
)
from gfn.gflownet.base import TrajectoryBasedGFlowNet, loss_reduce
from gfn.gflownet.losses import RegressionLoss
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet, SubTBWeighting
from gfn.states import States
from gfn.utils.handlers import call_estimator_with_conditions
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs
from gfn.utils.soft_rl import (
    gae_advantages,
    reward_to_go,
    soft_state_values,
    soft_step_returns,
    soft_step_rewards,
)

AdvantageEstimator = Literal["total", "reward_to_go", "baseline", "gae"]
CriticLoss = Literal["mse", "sub_eb"]
KLEstimator = Literal["analytic", "importance"]

# The importance ratio is exp() of a log-prob difference, which overflows to inf at
# ~89 in float32. An inf ratio yields a finite-looking clipped loss but NaN gradients,
# because exp's backward still multiplies through by inf. Anything beyond this bound
# is far outside any sane trust region, so clamping only affects samples the clip
# would have flattened anyway.
MAX_LOG_RATIO = 20.0


def ppo_clip(ratio: torch.Tensor, advantages: torch.Tensor, eps: float) -> torch.Tensor:
    r"""The PPO clipped surrogate $\min(\rho A, \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon) A)$.

    Args:
        ratio: The importance ratio $\rho_t = \pi_\theta / \pi_{\text{old}}$.
        advantages: The (detached) advantage estimates, broadcastable to ``ratio``.
        eps: The clipping parameter $\epsilon$.

    Returns:
        A tensor of the same shape as ``ratio`` holding the clipped surrogate. This is
        *maximized*, so callers negate it to form a loss.
    """
    return torch.minimum(
        ratio * advantages, ratio.clamp(1.0 - eps, 1.0 + eps) * advantages
    )


def masked_categorical_kl(
    dist_new: Distribution, dist_old: Distribution, debug: bool = False
) -> torch.Tensor:
    r"""Computes $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{old}})$ over masked categoricals.

    ``torch.distributions.kl_divergence`` cannot be used directly here. Invalid actions
    carry ``-inf`` logits, and its implementation forms ``logits_p - logits_q`` before
    masking the result, so those entries are ``nan`` in the forward pass. The forward
    value survives (the masked assignment overwrites it), but the backward pass
    multiplies a zero upstream gradient by that ``nan`` and poisons every policy
    gradient. This computes the same quantity while keeping the arithmetic finite.

    Args:
        dist_new: The current policy's action distribution, exposing ``logits``.
        dist_old: The rollout policy's action distribution, exposing ``logits``.
        debug: If True, asserts that both distributions mask the same actions.

    Returns:
        A tensor of shape ``dist_new.batch_shape`` holding the per-state KL.
    """
    log_p = dist_new.logits  # type: ignore[attr-defined]  # Normalized by Categorical.
    log_q = dist_old.logits  # type: ignore[attr-defined]

    valid = torch.isfinite(log_p)
    if debug:
        assert (valid == torch.isfinite(log_q)).all(), (
            "The current and rollout policies mask different actions, so their KL is "
            "infinite. This should not happen: both are built from the same states."
        )

    zero = torch.zeros_like(log_p)
    safe_p = torch.where(valid, log_p, zero)
    safe_q = torch.where(valid, log_q, zero)
    return torch.where(valid, safe_p.exp() * (safe_p - safe_q), zero).sum(-1)


def tlm_loss(
    pb: Estimator,
    trajectories: Trajectories,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Trajectory Likelihood Maximization loss for the backward policy (Eq. 24).

    .. math::

        \mathrm{TLM} = -\mathbb{E}_{\tau \sim P_F}
            \Big[ \sum_t \log P_B(s_t \mid s_{t+1}, \varphi) \Big],
        \qquad
        \nabla_\varphi \mathrm{TLM} = \nabla_\varphi \mathrm{KL}(P_F \,\|\, P_B).

    Objectives derived from the soft-RL formulation (VPG, Ent-PPO) do not admit a joint
    forward-backward objective, because changing $P_B$ changes the MDP reward itself.
    TLM sidesteps this by fitting $P_B$ to the trajectory distribution induced by the
    current $P_F$, and can equally be used alongside DB / TB / SubTB.

    Only the backward policy receives gradient: the expectation is over trajectories
    sampled from $P_F$ and involves no reparameterization, so no detaching is required.

    Args:
        pb: The backward policy estimator to train.
        trajectories: The batch of forward trajectories sampled from $P_F$.
        reduction: The reduction method to use ('mean', 'sum', or 'none').

    Returns:
        The TLM loss. Its shape depends on the reduction method.
    """
    log_pb = get_trajectory_pbs(pb, trajectories)
    return loss_reduce(-log_pb.sum(dim=0), reduction)


def _validate_soft_value_estimator(logV: Estimator | None, required: bool) -> None:
    """Validates the soft value estimator against the selected advantage estimator.

    Args:
        logV: The soft value estimator, or None.
        required: Whether the selected advantage estimator needs a value function.

    Raises:
        ValueError: If a value estimator is required but missing, or supplied but
            unused.
        TypeError: If ``logV`` is not a scalar estimator.
    """
    if required and logV is None:
        raise ValueError(
            "A `logV` estimator is required for the 'baseline' and 'gae' advantage "
            "estimators. Pass a ScalarEstimator, or select 'total'/'reward_to_go'."
        )
    if not required and logV is not None:
        raise ValueError(
            "A `logV` estimator was passed but the selected advantage estimator does "
            "not use a value function. Select 'baseline' or 'gae', or drop `logV`."
        )
    if logV is not None and not isinstance(
        logV, (ScalarEstimator, ConditionalScalarEstimator)
    ):
        raise TypeError(
            f"logV must be a ScalarEstimator or ConditionalScalarEstimator, "
            f"got {type(logV).__name__}"
        )


class PolicyGradientGFlowNet(TrajectoryBasedGFlowNet):
    r"""GFlowNet trained by Vanilla Policy Gradient on the soft-RL objective.

    Maximizes the soft return $\tilde{J}(\pi_\theta) = \tilde{V}^{\pi_\theta}(s_0)$ via
    the policy gradient theorem (their Eq. 8),

    .. math::

        \nabla_\theta \tilde{J}(\pi_\theta) = \mathbb{E}_{\pi_\theta}
            \Big[ \sum_{t} \nabla_\theta \log \pi_\theta(s_{t+1} \mid s_t)\, \Psi_t \Big],

    where $\Psi_t$ is one of four estimators of increasing sophistication, selected by
    ``advantage``:

    - ``"total"``: $\Psi_t = \sum_k g_k$, the full soft return. Unbiased, high variance.
      Note $\sum_k g_k$ is exactly the negated Trajectory Balance score, so this
      estimator is REINFORCE on the reverse KL — with the learned $\log Z$ of TB playing
      the role of a constant baseline (Malkin et al., 2023).
    - ``"reward_to_go"``: $\Psi_t = \hat{R}_t = \sum_{k \geq t} g_k$ (Eq. 10).
    - ``"baseline"``: $\Psi_t = \hat{R}_t - V_\varphi(s_t)$ (Eq. 11).
    - ``"gae"``: $\Psi_t = \sum_k \lambda^k \delta_{t+k}$ (Eq. 13). The paper's choice.

    The advantages and value-fit targets are computed once per rollout by
    :meth:`to_training_samples` and frozen in a
    :class:`~gfn.containers.PolicyGradientTrajectories`, so the same batch can be reused
    for several gradient steps without the targets drifting.

    Attributes:
        pf: The forward policy estimator (the RL agent's policy).
        pb: The backward policy estimator, which defines the MDP reward and must be held
            fixed by the policy-gradient update. Train it separately with
            :func:`tlm_loss` if desired.
        logV: A ScalarEstimator for the soft value $\tilde{V}_\varphi$, or None when the
            selected advantage estimator does not use one. Per the soft-RL dictionary,
            $\tilde{V}^{\pi^\star}(s) = \log \mathcal{F}(s)$ and
            $\tilde{V}^{\pi^\star}(s_0) = \log Z$, so this is the same object DB/SubTB
            call ``logF`` — named ``logV`` here because on-policy it tracks
            $\tilde{V}^{\pi_\theta}$, not the optimal flow.
        advantage: Which estimator of $\Psi_t$ to use.
        gae_lambda: The GAE bias-variance parameter (paper's tuned value: 0.7).
        critic_loss: How to fit ``logV``: ``"mse"`` regression onto the frozen targets,
            or ``"sub_eb"``, the Sub-EB balance objective of Eqs. 14-15.
        value_loss_weight: Weight of the critic term inside the combined :meth:`loss`.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            GFlowNet DAG is a tree, and pb is therefore always 1.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logV: ScalarEstimator | ConditionalScalarEstimator | None = None,
        advantage: AdvantageEstimator = "gae",
        gae_lambda: float = 0.7,
        critic_loss: CriticLoss = "mse",
        sub_eb_weighting: SubTBWeighting = "geometric_within",
        sub_eb_lamda: float = 0.9,
        value_loss_weight: float = 1.0,
        constant_pb: bool = False,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ) -> None:
        """Initializes a PolicyGradientGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the GFlowNet DAG is a tree.
            logV: A ScalarEstimator for the soft value function. Required for the
                'baseline' and 'gae' advantage estimators, and rejected otherwise.
            advantage: The advantage estimator, one of 'total', 'reward_to_go',
                'baseline', 'gae'.
            gae_lambda: The GAE parameter, used when advantage='gae'.
            critic_loss: 'mse' regression onto the frozen targets, or 'sub_eb' for the
                Sub-EB balance objective. The paper reports 'sub_eb' slightly worse than
                plain GAE, so 'mse' is the default.
            sub_eb_weighting: Sub-trajectory weighting scheme for the Sub-EB critic; see
                :class:`~gfn.gflownet.sub_trajectory_balance.SubTBGFlowNet`.
            sub_eb_lamda: Discount for longer sub-trajectories in the Sub-EB critic.
            value_loss_weight: Weight applied to the critic term in :meth:`loss`.
            constant_pb: Whether to ignore the backward policy estimator.
            log_reward_clip_min: If finite, clips log rewards to this value.
            debug: If True, keep runtime safety checks active.
            loss_fn: Regression loss applied to the critic residuals. Defaults to
                :class:`~gfn.gflownet.losses.SquaredLoss`.

        Raises:
            ValueError: If ``advantage`` or ``critic_loss`` is unrecognized, or if
                ``logV`` is missing when required (or supplied when unused).
            TypeError: If ``logV`` is not a scalar estimator.
        """
        super().__init__(
            pf,
            pb,
            constant_pb=constant_pb,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )
        if advantage not in ("total", "reward_to_go", "baseline", "gae"):
            raise ValueError(
                "advantage must be one of 'total', 'reward_to_go', 'baseline', 'gae'; "
                f"got {advantage!r}"
            )
        if critic_loss not in ("mse", "sub_eb"):
            raise ValueError(
                f"critic_loss must be 'mse' or 'sub_eb'; got {critic_loss!r}"
            )
        needs_value = advantage in ("baseline", "gae")
        _validate_soft_value_estimator(logV, needs_value)
        if critic_loss == "sub_eb" and not needs_value:
            raise ValueError(
                "critic_loss='sub_eb' trains a value function, which the "
                f"{advantage!r} advantage estimator does not use."
            )

        self.logV = logV
        self.advantage = advantage
        self.gae_lambda = gae_lambda
        self.critic_loss = critic_loss
        self.value_loss_weight = value_loss_weight

        if debug and logV is not None and hasattr(logV, "debug"):
            logV.debug = True

        # The Sub-EB critic (Eqs. 14-15) is the SubTB residual with log F replaced by V
        # and the policies frozen, so reuse SubTB rather than reimplementing its
        # sub-trajectory bookkeeping.
        self._sub_eb: SubTBGFlowNet | None = None
        if critic_loss == "sub_eb":
            assert logV is not None  # Guaranteed by _validate_soft_value_estimator.
            sub_eb = SubTBGFlowNet(
                pf=pf,
                pb=pb,
                logF=logV,
                weighting=sub_eb_weighting,
                lamda=sub_eb_lamda,
                log_reward_clip_min=log_reward_clip_min,
                constant_pb=constant_pb,
                debug=debug,
                loss_fn=loss_fn,
            )
            # Bypass nn.Module registration: the helper owns no parameters of its own,
            # it borrows pf/pb/logV, which are already registered here. Registering it
            # would duplicate every one of them under `_sub_eb.*` in `state_dict()`,
            # making checkpoints twice the size and un-loadable into an `mse` model.
            object.__setattr__(self, "_sub_eb", sub_eb)

    def logV_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns named parameters containing 'logV' in their name.

        Returns:
            A dictionary of the soft value function's named parameters.
        """
        return {k: v for k, v in self.named_parameters() if "logV" in k}

    def logV_parameters(self) -> list[torch.Tensor]:
        """Returns parameters containing 'logV' in their name.

        Returns:
            A list of the soft value function's parameters.
        """
        return [v for k, v in self.named_parameters() if "logV" in k]

    # ------------------------------------------------------------------
    # Per-rollout preparation (Algorithm 1 lines 3-4 / Algorithm 2 line 4)
    # ------------------------------------------------------------------

    def to_training_samples(
        self,
        trajectories: Trajectories,
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> PolicyGradientTrajectories:
        r"""Freezes the per-rollout quantities the update must hold fixed.

        Assigns the soft-MDP rewards of Eq. 7, forms the one-step soft returns $g_t$,
        and computes both the selected advantage estimate $\hat{A}_t$ and the matching
        value-fit target $y_t$ — the Monte-Carlo soft return for the non-GAE estimators,
        and the bootstrapped $\mathrm{sg}[\hat{A}_t + V_\varphi(s_t)]$ for GAE.

        All of this is evaluated at the rollout-time policy under ``no_grad`` and stored,
        so that taking several gradient steps on the batch does not move the targets.

        Args:
            trajectories: The freshly sampled forward trajectories. Passing an already
                prepared :class:`~gfn.containers.PolicyGradientTrajectories` is a no-op.
            log_rewards: Optional custom log rewards of shape (n_trajectories,). When
                None, uses the environment rewards carried by the trajectories. Useful
                for intrinsic rewards (see "Towards Improving Exploration through
                Sibling Augmented GFlowNets", Madan et al., ICLR 2025).

        Returns:
            A PolicyGradientTrajectories carrying the frozen advantages, value targets,
            and rollout-time log-probabilities.
        """
        if isinstance(trajectories, PolicyGradientTrajectories):
            return trajectories

        with torch.no_grad():
            # Recompute rather than reuse `trajectories.log_probs`: the saved values are
            # the *behavior* log-probs, which differ from pi_{theta_k} whenever the
            # rollout used exploration kwargs. The soft return, the PPO ratio and the KL
            # are all defined against the policy, not the behavior distribution.
            log_pf = get_trajectory_pfs(
                self.pf, trajectories, recalculate_all_logprobs=True
            )
            log_pb = get_trajectory_pbs(self.pb, trajectories)
            rewards = soft_step_rewards(
                trajectories,
                log_pb,
                log_rewards=log_rewards,
                log_reward_clip_min=self.log_reward_clip_min,
            )
            soft_returns = soft_step_returns(log_pf, rewards)
            advantages, value_targets = self._advantages_and_targets(
                trajectories, soft_returns
            )

            # Padded steps carry no signal; zero them so the frozen tensors are exactly
            # what the masked loss sees.
            valid = ~trajectories.actions.is_dummy
            advantages = advantages * valid
            value_targets = value_targets * valid

        prepared = PolicyGradientTrajectories.from_trajectories(
            trajectories, advantages, value_targets
        )
        prepared.log_probs = log_pf
        return prepared

    def _advantages_and_targets(
        self, trajectories: Trajectories, soft_returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the selected advantage estimate and its value-fit target.

        Args:
            trajectories: The batch of trajectories.
            soft_returns: Tensor of shape (T, N) of one-step soft returns.

        Returns:
            A tuple ``(advantages, value_targets)``, each of shape (T, N).
        """
        if self.advantage == "total":
            total = soft_returns.sum(dim=0, keepdim=True)
            return total.expand_as(soft_returns).clone(), torch.zeros_like(soft_returns)

        rtg = reward_to_go(soft_returns)
        if self.advantage == "reward_to_go":
            return rtg, torch.zeros_like(soft_returns)

        assert self.logV is not None  # Guaranteed by the constructor validation.
        values = soft_state_values(self.logV, trajectories, debug=self.debug)
        if self.advantage == "baseline":
            # Monte-Carlo target, matching Algorithm 1 line 4.
            return rtg - values[:-1], rtg

        advantages = gae_advantages(soft_returns, values, self.gae_lambda)
        # Bootstrapped target sg[A_t + V(s_t)], which keeps the critic consistent with
        # the bias-variance trade-off that lambda selects (Algorithm 2 line 4).
        return advantages, advantages + values[:-1]

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def policy_loss(
        self,
        trajectories: PolicyGradientTrajectories,
        reduction: str = "mean",
    ) -> torch.Tensor:
        r"""Computes the VPG surrogate loss $-\sum_t \log \pi_\theta(a_t|s_t)\hat{A}_t$.

        The advantage is detached: the score-function identity guarantees that gradients
        flowing through $\Psi_t$ vanish in expectation, so the paper (and standard RL
        practice) drops them.

        Args:
            trajectories: A prepared batch from :meth:`to_training_samples`.
            reduction: How to reduce over trajectories ('mean', 'sum' or 'none'). Steps
                are always summed within a trajectory, matching Algorithm 1 line 5.

        Returns:
            The policy-gradient loss. Its shape depends on the reduction method.

        Raises:
            TypeError: If ``trajectories`` has not been prepared.
        """
        batch = self._require_prepared(trajectories)
        log_pf = get_trajectory_pfs(self.pf, batch, recalculate_all_logprobs=True)
        valid = ~batch.actions.is_dummy
        per_step = log_pf * batch.advantages.detach() * valid
        return loss_reduce(-per_step.sum(dim=0), reduction)

    def value_loss(
        self,
        env: Env,
        trajectories: PolicyGradientTrajectories,
        reduction: str = "mean",
    ) -> torch.Tensor:
        r"""Fits the soft value function $\tilde{V}_\varphi$ on the frozen targets.

        With ``critic_loss="mse"`` this is Algorithm 1 line 7 / Algorithm 2 line 8: mean
        squared error onto $y_t$, averaged over valid steps. With
        ``critic_loss="sub_eb"`` it is instead the Sub-EB balance objective (Eqs. 14-15),
        evaluated with the policies frozen so that only $\varphi$ receives gradient.

        Args:
            env: The environment the trajectories were sampled from.
            trajectories: A prepared batch from :meth:`to_training_samples`.
            reduction: The reduction method to use.

        Returns:
            The critic loss, or a detached zero when the objective uses no critic.

        Raises:
            TypeError: If ``trajectories`` has not been prepared.
            ValueError: If the Sub-EB critic is selected and the batch carries no
                rollout-time log-probabilities.
        """
        batch = self._require_prepared(trajectories)
        if self.logV is None:
            return torch.zeros((), device=batch.device)

        if self._sub_eb is not None:
            # Eq. 15 holds theta fixed for the critic update, so the residual must be
            # formed at the rollout policy — which the batch already carries — and not
            # at whatever the policy has drifted to during the K inner epochs.
            if batch.log_probs is None:
                raise ValueError(
                    "The Sub-EB critic needs the rollout-time log-probabilities; the "
                    "batch must come from `to_training_samples`."
                )
            log_pf = batch.log_probs.detach()
            with torch.no_grad():
                log_pb = get_trajectory_pbs(self.pb, batch)
            return self._sub_eb.loss(
                env,
                batch,
                reduction=reduction,
                log_pf_trajectories=log_pf,
                log_pb_trajectories=log_pb,
            )

        values = soft_state_values(self.logV, batch, debug=self.debug)
        valid = ~batch.actions.is_dummy
        residuals = (values[:-1] - batch.value_targets.detach())[valid]
        return loss_reduce(self.loss_fn(residuals), reduction)

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
        *,
        log_rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the combined policy and critic loss.

        Provided so the class works with the standard ``loss_from_trajectories`` path and
        a single optimizer with per-component parameter groups. Training loops that take
        a different number of policy and critic steps (the paper's K, E and S) should
        call :meth:`policy_loss` and :meth:`value_loss` directly.

        Args:
            env: The environment the trajectories were sampled from.
            trajectories: Trajectories, prepared or not.
            recalculate_all_logprobs: Unused; the policy log-probs are always
                re-evaluated, because the loss is taken at the current parameters while
                the frozen batch holds the rollout-time ones.
            reduction: The reduction method to use.
            log_rewards: Optional custom log rewards, forwarded to
                :meth:`to_training_samples`.

        Returns:
            ``policy_loss + value_loss_weight * value_loss``.

        Raises:
            ValueError: If ``reduction`` is 'none' while a critic is in use. The policy
                term is one value per trajectory and the critic term one per valid step,
                so there is no elementwise sum of the two; call :meth:`policy_loss` and
                :meth:`value_loss` separately for unreduced values.
        """
        if reduction == "none" and self.logV is not None:
            raise ValueError(
                "reduction='none' is not supported for the combined loss: the policy "
                "term is per-trajectory and the critic term is per-step. Call "
                "`policy_loss` and `value_loss` separately instead."
            )
        batch = self.to_training_samples(trajectories, log_rewards=log_rewards)
        loss = self.policy_loss(batch, reduction=reduction)
        if self.logV is not None:
            loss = loss + self.value_loss_weight * self.value_loss(
                env, batch, reduction=reduction
            )
        return loss

    def _require_prepared(
        self, trajectories: Trajectories
    ) -> PolicyGradientTrajectories:
        """Checks that the batch carries frozen advantages.

        Args:
            trajectories: The batch to check.

        Returns:
            The batch, narrowed to PolicyGradientTrajectories.

        Raises:
            TypeError: If the batch has not been prepared.
        """
        if not isinstance(trajectories, PolicyGradientTrajectories):
            raise TypeError(
                "Policy-gradient losses need a PolicyGradientTrajectories carrying "
                "frozen advantages; call `gflownet.to_training_samples(trajectories)` "
                "once per rollout and pass the result."
            )
        return trajectories


class EntPPOGFlowNet(PolicyGradientGFlowNet):
    r"""GFlowNet trained by Entropic Proximal Policy Optimization.

    Maximizes the surrogate of Eq. 20,

    .. math::

        \mathcal{L}_{\mathrm{Ent\text{-}PPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{old}}
            \Big[ \sum_t \mathrm{PPOClip}(\rho_t(\theta), \hat{A}_t)
                  - \mathrm{KL}(\pi_\theta(\cdot|s_t) \,\|\, \pi_{old}(\cdot|s_t)) \Big],

    with $\rho_t = \pi_\theta(a_t|s_t) / \pi_{old}(a_t|s_t)$ and GAE advantages.

    The KL term is not a heuristic regularizer bolted onto PPO. At $\alpha = 1$ the soft
    policy improvement operator maximizes
    $\mathbb{E}_{\pi_\theta}[\tilde{Q}^{\pi_{old}}] + \mathcal{H}(\pi_\theta)$; writing
    $\tilde{Q}$ in terms of the soft advantage leaves a cross-entropy against
    $\pi_{old}$, which combines with the entropy into exactly
    $-\mathrm{KL}(\pi_\theta \| \pi_{old})$ (their Eq. 18). Standard PPO instead carries a
    free entropy coefficient $c_2$ and no cross-entropy term, and its optimum concentrates
    on $\arg\max_x R(x)$ rather than sampling from $R/Z$ — which is why earlier attempts
    at PPO for GFlowNets reported mode collapse.

    Clipping is still required on top: the analytic KL pins down the right optimum but
    does not bound the per-sample importance ratio, whose fluctuations across update
    epochs destabilize the gradient estimator.

    ``use_kl=False`` reduces this to naive PPO and ``use_clipping=False`` removes the
    ratio clip; both exist to reproduce the paper's ablations, not for production use.

    Attributes:
        clip_eps: The PPO clipping parameter $\epsilon$ (paper: 0.2).
        use_kl: Whether to include the analytic KL penalty.
        use_clipping: Whether to clip the importance ratio.
        kl_estimator: ``"analytic"`` computes the exact KL between the current and
            rollout action distributions (requires ``save_estimator_outputs=True`` at
            sampling time and policies torch can take a KL between, i.e. the discrete
            Categorical policies). ``"importance"`` uses the single-sample unbiased
            estimator $\mathbb{E}_{a \sim \pi_{old}}[\rho \log \rho]$, which needs only
            log-probabilities and therefore works for any policy.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        logV: ScalarEstimator | ConditionalScalarEstimator,
        gae_lambda: float = 0.7,
        clip_eps: float = 0.2,
        use_kl: bool = True,
        use_clipping: bool = True,
        kl_estimator: KLEstimator = "analytic",
        critic_loss: CriticLoss = "mse",
        sub_eb_weighting: SubTBWeighting = "geometric_within",
        sub_eb_lamda: float = 0.9,
        value_loss_weight: float = 1.0,
        constant_pb: bool = False,
        log_reward_clip_min: float = -float("inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ) -> None:
        """Initializes an EntPPOGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the GFlowNet DAG is a tree.
            logV: A ScalarEstimator for the soft value function. Required: Ent-PPO is
                defined on GAE advantages.
            gae_lambda: The GAE parameter (paper: 0.7).
            clip_eps: The PPO clipping parameter (paper: 0.2).
            use_kl: Whether to include the analytic KL penalty. False reproduces the
                naive-PPO ablation.
            use_clipping: Whether to clip the importance ratio. False reproduces the
                no-clipping ablation.
            kl_estimator: 'analytic' or 'importance'; see the class docstring.
            critic_loss: 'mse' or 'sub_eb'.
            sub_eb_weighting: Sub-trajectory weighting for the Sub-EB critic.
            sub_eb_lamda: Discount for longer sub-trajectories in the Sub-EB critic.
            value_loss_weight: Weight applied to the critic term in :meth:`loss`.
            constant_pb: Whether to ignore the backward policy estimator.
            log_reward_clip_min: If finite, clips log rewards to this value.
            debug: If True, keep runtime safety checks active.
            loss_fn: Regression loss applied to the critic residuals.
        """
        super().__init__(
            pf,
            pb,
            logV=logV,
            advantage="gae",
            gae_lambda=gae_lambda,
            critic_loss=critic_loss,
            sub_eb_weighting=sub_eb_weighting,
            sub_eb_lamda=sub_eb_lamda,
            value_loss_weight=value_loss_weight,
            constant_pb=constant_pb,
            log_reward_clip_min=log_reward_clip_min,
            debug=debug,
            loss_fn=loss_fn,
        )
        if kl_estimator not in ("analytic", "importance"):
            raise ValueError(
                f"kl_estimator must be 'analytic' or 'importance'; got {kl_estimator!r}"
            )
        if clip_eps <= 0.0:
            raise ValueError(f"clip_eps must be positive; got {clip_eps}")
        self.clip_eps = clip_eps
        self.use_kl = use_kl
        self.use_clipping = use_clipping
        self.kl_estimator = kl_estimator

    def policy_loss(
        self,
        trajectories: PolicyGradientTrajectories,
        reduction: str = "mean",
    ) -> torch.Tensor:
        r"""Computes the Ent-PPO loss (the negated surrogate of Eq. 20).

        Args:
            trajectories: A prepared batch from :meth:`to_training_samples`. Its
                ``log_probs`` hold $\log \pi_{\text{old}}$ and, for the analytic KL,
                its ``estimator_outputs`` hold the $\pi_{\text{old}}$ logits.
            reduction: How to reduce over trajectories. Steps are summed within a
                trajectory, matching Algorithm 2 line 6.

        Returns:
            The Ent-PPO loss. Its shape depends on the reduction method.

        Raises:
            TypeError: If ``trajectories`` has not been prepared.
            ValueError: If the batch carries no rollout-time log-probabilities.
        """
        batch = self._require_prepared(trajectories)
        log_pf_old = batch.log_probs
        if log_pf_old is None:
            raise ValueError(
                "The prepared batch is missing rollout-time log-probabilities; it must "
                "come from `to_training_samples`."
            )
        log_pf_old = log_pf_old.detach()

        log_pf_new, kl = self._current_logprobs_and_kl(batch)
        valid = ~batch.actions.is_dummy
        ratio = torch.exp((log_pf_new - log_pf_old).clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO))

        advantages = batch.advantages.detach()
        if self.use_clipping:
            surrogate = ppo_clip(ratio, advantages, self.clip_eps)
        else:
            surrogate = ratio * advantages
        if kl is not None:
            surrogate = surrogate - kl

        return loss_reduce(-(surrogate * valid).sum(dim=0), reduction)

    def _current_logprobs_and_kl(
        self, batch: PolicyGradientTrajectories
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        r"""Evaluates $\log \pi_\theta(a_t|s_t)$ and the KL against $\pi_{old}$.

        ``get_trajectory_pfs`` returns only the chosen-action log-probability, but the
        analytic KL needs the full action distribution, so this rebuilds both
        distributions on the valid states in one vectorized pass.

        Args:
            batch: A prepared batch.

        Returns:
            A tuple ``(log_pf_new, kl)`` of tensors of shape (T, N); ``kl`` is None when
            ``use_kl`` is False.

        Raises:
            NotImplementedError: If the forward policy is not vectorized (e.g. a
                recurrent estimator), or if no analytic KL is registered for the
                policy's distribution type.
            ValueError: If the analytic KL is requested but the batch carries no
                rollout-time estimator outputs.
        """
        policy_pf = cast(PolicyEstimatorProtocol, self.pf)
        validate_policy_estimator(policy_pf, "pf")

        if not self.use_kl:
            # Naive-PPO ablation: only the chosen action's log-prob is needed.
            return (
                get_trajectory_pfs(self.pf, batch, recalculate_all_logprobs=True),
                None,
            )

        if self.kl_estimator == "importance":
            # rho * log(rho) with rho = pi_theta / pi_old is an unbiased single-sample
            # estimator of KL(pi_theta || pi_old) under a ~ pi_old, and needs only
            # log-probabilities.
            log_pf_new = get_trajectory_pfs(
                self.pf, batch, recalculate_all_logprobs=True
            )
            assert batch.log_probs is not None
            log_ratio = (log_pf_new - batch.log_probs.detach()).clamp(
                -MAX_LOG_RATIO, MAX_LOG_RATIO
            )
            return log_pf_new, torch.exp(log_ratio) * log_ratio

        if not bool(getattr(policy_pf, "is_vectorized", True)):
            raise NotImplementedError(
                "Ent-PPO's analytic KL requires a vectorized policy estimator; "
                "recurrent / per-step estimators are not supported yet. Pass "
                "kl_estimator='importance' to use the single-sample estimator instead."
            )

        state_mask = ~batch.states.is_sink_state
        action_mask = ~batch.actions.is_dummy
        if self.debug:
            assert (state_mask[:-1] == action_mask).all()

        log_pf_new = torch.zeros(
            action_mask.shape,
            dtype=torch.get_default_dtype(),
            device=batch.states.device,
        )
        kl = torch.zeros_like(log_pf_new)

        valid_states = batch.states[state_mask]
        valid_actions = batch.actions[action_mask]
        if len(valid_states) == 0:
            return log_pf_new, kl

        conditions = batch.states.conditions
        valid_conditions = None if conditions is None else conditions[state_mask]

        dist_new = self._distribution(valid_states, None, valid_conditions)
        log_pf_new[action_mask] = dist_new.log_prob(valid_actions.tensor).to(
            log_pf_new.dtype
        )

        if batch.estimator_outputs is None:
            raise ValueError(
                "The analytic KL needs the rollout policy's estimator outputs. "
                "Sample with `save_estimator_outputs=True`, or pass "
                "kl_estimator='importance'."
            )
        # Detach: pi_old is a fixed reference. The sampler records estimator outputs
        # with their autograd graph attached, which would otherwise be freed by the
        # first of the K inner backward passes.
        dist_old = self._distribution(
            valid_states, batch.estimator_outputs[action_mask].detach(), valid_conditions
        )
        if hasattr(dist_new, "logits") and hasattr(dist_old, "logits"):
            kl_valid = masked_categorical_kl(dist_new, dist_old, debug=self.debug)
        else:
            try:
                kl_valid = kl_divergence(dist_new, dist_old)
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"No analytic KL is registered for {type(dist_new).__name__}. "
                    f"Pass kl_estimator='importance' to use the single-sample "
                    f"estimator instead."
                ) from error
        kl[action_mask] = kl_valid.to(kl.dtype)

        return log_pf_new, kl

    def _distribution(
        self,
        states: States,
        module_output: torch.Tensor | None,
        conditions: torch.Tensor | None,
    ) -> Distribution:
        """Builds the forward policy's action distribution over a flat batch of states.

        Args:
            states: The states to evaluate, with a one-dimensional batch shape.
            module_output: Precomputed estimator outputs (the rollout policy's logits),
                or None to run the current forward policy.
            conditions: Conditions aligned with ``states``, or None.

        Returns:
            The action distribution.
        """
        if module_output is None:
            module_output = call_estimator_with_conditions(
                self.pf, "pf", states, conditions
            )
        return self.pf.to_probability_distribution(states, module_output)
