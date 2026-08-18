"""Container for policy-gradient rollouts with frozen advantages and value targets."""

from __future__ import annotations

from typing import Sequence

import torch
from tensordict.base import TensorDictBase

from gfn.containers.trajectories import Trajectories, pad_dim0_if_needed
from gfn.env import Env


class PolicyGradientTrajectories(Trajectories):
    r"""Trajectories augmented with the per-rollout quantities a PPO update must freeze.

    Policy-gradient GFlowNet training (Zykova-Myzina et al., 2026, arXiv:2606.15793)
    takes several gradient steps on a single batch of rollouts. Algorithm 2 line 4 is
    explicit that the advantages $\hat{A}_t$, the value-fit targets $y_t$ and the
    rollout-time log-probabilities are computed **once** and held fixed across all $K$
    inner epochs: recomputing them against the moving $V_\varphi$ would change the
    objective mid-update and break the trust-region argument.

    This container carries those frozen quantities alongside the trajectories, so a
    training loop can slice it into mini-batches and re-evaluate the loss as many times
    as it likes without any hidden state on the GFlowNet. The rollout-time policy
    $\pi_{\text{old}}$ is carried by the inherited fields:

    - ``log_probs`` holds $\log \pi_{\text{old}}(a_t | s_t)$, which the PPO ratio divides
      by. Note that :meth:`PolicyGradientGFlowNet.to_training_samples
      <gfn.gflownet.policy_gradient.PolicyGradientGFlowNet.to_training_samples>`
      *overwrites* whatever the sampler saved here: the ratio and the KL are defined
      against the policy $\pi_{\theta_k}$, which differs from the behavior distribution
      whenever the rollout used exploration kwargs.
    - ``estimator_outputs`` (from ``save_estimator_outputs=True``) are the
      $\pi_{\text{old}}$ logits, needed to rebuild the full old action distribution for
      the analytic KL penalty.

    Attributes:
        advantages: Tensor of shape (max_length, batch_size) with the frozen soft
            advantage estimates $\hat{A}_t$.
        value_targets: Tensor of shape (max_length, batch_size) with the frozen
            regression targets $y_t$ for the soft value function.
    """

    def __init__(
        self,
        env: Env,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Initializes a PolicyGradientTrajectories instance.

        Args:
            env: The environment where the states and actions are defined.
            advantages: Frozen advantages of shape (max_length, batch_size).
            value_targets: Frozen value-fit targets of shape (max_length, batch_size).
            **kwargs: Forwarded to :class:`~gfn.containers.Trajectories`.

        Raises:
            ValueError: If the frozen tensors do not match the actions batch shape.
        """
        super().__init__(env, **kwargs)
        self.advantages = advantages
        self.value_targets = value_targets
        self._check_frozen_shapes()

    def _check_frozen_shapes(self) -> None:
        """Validates that the frozen tensors align with the actions batch shape."""
        expected = self.actions.batch_shape
        for name, tensor in (
            ("advantages", self.advantages),
            ("value_targets", self.value_targets),
        ):
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, expected {tuple(expected)} "
                    "to match the actions batch shape (max_length, batch_size)."
                )

    @classmethod
    def from_trajectories(
        cls,
        trajectories: Trajectories,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> "PolicyGradientTrajectories":
        r"""Wraps existing trajectories together with their frozen PG quantities.

        The rollout-policy tensors are detached. The sampler records
        ``estimator_outputs`` with its autograd graph still attached, which would pin
        every per-step forward activation for the whole update, break ``deepcopy``, and
        make the second inner epoch fail with "backward through the graph a second
        time". Nothing downstream wants a gradient through $\pi_{\text{old}}$.

        Args:
            trajectories: The rollout to wrap. Its ``log_probs`` (and, for Ent-PPO, its
                ``estimator_outputs``) define $\pi_{\text{old}}$.
            advantages: Frozen advantages of shape (max_length, batch_size).
            value_targets: Frozen value-fit targets of shape (max_length, batch_size).

        Returns:
            A new PolicyGradientTrajectories sharing the input's states and actions.
        """
        log_probs = trajectories.log_probs
        estimator_outputs = trajectories.estimator_outputs
        return cls(
            env=trajectories.env,
            advantages=advantages.detach(),
            value_targets=value_targets.detach(),
            states=trajectories.states,
            actions=trajectories.actions,
            terminating_idx=trajectories.terminating_idx,
            is_backward=trajectories.is_backward,
            log_rewards=trajectories._log_rewards,
            log_probs=None if log_probs is None else log_probs.detach(),
            estimator_outputs=(
                None if estimator_outputs is None else estimator_outputs.detach()
            ),
        )

    def __getitem__(
        self, index: int | slice | tuple | Sequence[int] | Sequence[bool] | torch.Tensor
    ) -> "PolicyGradientTrajectories":
        """Returns a subset of the trajectories along the batch dimension.

        This is what a training loop uses to build the ``S`` mini-batch splits of an
        update epoch. The frozen tensors are sliced and truncated exactly like
        ``log_probs`` in the base class.

        Args:
            index: Indices to select trajectories.

        Returns:
            A new PolicyGradientTrajectories with the selected trajectories.
        """
        sub = super().__getitem__(index)
        if isinstance(index, int):
            index = [index]
        new_max_length = sub.actions.batch_shape[0]
        return PolicyGradientTrajectories.from_trajectories(
            sub,
            self.advantages[:, index][:new_max_length],
            self.value_targets[:, index][:new_max_length],
        )

    def extend(self, other: Trajectories) -> None:
        """Extends this container with another PolicyGradientTrajectories.

        Args:
            other: Another PolicyGradientTrajectories to append.

        Raises:
            TypeError: If ``other`` does not carry frozen PG quantities.
        """
        if not isinstance(other, PolicyGradientTrajectories):
            raise TypeError(
                "PolicyGradientTrajectories can only be extended with another "
                f"PolicyGradientTrajectories, got {type(other).__name__}."
            )
        if len(other) == 0:
            return
        if len(self) == 0:
            self.advantages = torch.full(size=(0, 0), fill_value=0.0, device=self.device)
            self.value_targets = torch.full(
                size=(0, 0), fill_value=0.0, device=self.device
            )

        super().extend(other)

        # Pad to a common number of steps (0.0 is the neutral value for both, matching
        # how padded steps contribute nothing to the loss) before concatenating.
        self.advantages, other_adv = pad_dim0_if_needed(
            self.advantages, other.advantages, 0.0
        )
        self.advantages = torch.cat((self.advantages, other_adv), dim=1)
        self.value_targets, other_tgt = pad_dim0_if_needed(
            self.value_targets, other.value_targets, 0.0
        )
        self.value_targets = torch.cat((self.value_targets, other_tgt), dim=1)
        self._check_frozen_shapes()

    def to_tensordict(self) -> TensorDictBase:
        """Serializes the container, including the frozen PG quantities."""
        td = super().to_tensordict()
        td["advantages"] = self.advantages
        td["value_targets"] = self.value_targets
        return td

    @classmethod
    def from_tensordict(
        cls, env: Env, td: TensorDictBase
    ) -> "PolicyGradientTrajectories":
        """Reconstructs the container from a TensorDict.

        Args:
            env: The environment needed to reconstruct States/Actions.
            td: The TensorDict produced by :meth:`to_tensordict`.

        Returns:
            A new PolicyGradientTrajectories instance.
        """
        trajectories = Trajectories.from_tensordict(env, td)
        return cls.from_trajectories(
            trajectories,
            td["advantages"].clone(),
            td["value_targets"].clone(),
        )
