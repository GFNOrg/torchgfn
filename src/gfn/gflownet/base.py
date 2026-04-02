import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar, cast

import torch
import torch.nn as nn

from gfn.containers import Container, Trajectories
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.gflownet.losses import RegressionLoss, SquaredLoss
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.prob_calculations import (
    get_trajectory_pbs,
    get_trajectory_pfs,
    get_trajectory_pfs_and_pbs,
)

TrainingSampleType = TypeVar("TrainingSampleType", bound=Container)


def loss_reduce(loss: torch.Tensor, method: str) -> torch.Tensor:
    """Utility function to handle loss aggregation strategies.

    Args:
        loss: The tensor to reduce.
        method: The reduction method to use ('mean', 'sum', or 'none').

    Returns:
        The reduced tensor.
    """
    if method == "mean":
        return loss.mean()
    elif method == "sum":
        return loss.sum()
    elif method == "none":
        return loss
    else:
        raise ValueError(f"Invalid loss reduction method: {method}")


class GFlowNet(ABC, nn.Module, Generic[TrainingSampleType]):
    """Abstract base class for GFlowNets.

    A formal definition of GFlowNets is given in Sec. 3 of
    [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266).
    """

    log_reward_clip_min = float("-inf")  # Default off.

    def __init__(
        self,
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ) -> None:
        """Initialize shared GFlowNet state.

        Args:
            debug: If True, keep runtime safety checks and warnings active. Set False
                in compiled hot paths to avoid graph breaks; use True in tests/debugging.
            loss_fn: Regression loss applied to balance condition residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss` (standard
                ``t²``, corresponding to reverse KL). See
                :mod:`gfn.gflownet.losses` for alternatives.
        """
        super().__init__()
        self.debug = debug
        self.loss_fn = loss_fn or SquaredLoss()

    @abstractmethod
    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditions: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples a specific number of complete trajectories from the environment.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample.
            conditions: Optional conditions tensor for conditional environments.
            save_logprobs: Whether to save the logprobs of the actions (useful for
                on-policy learning).
            save_estimator_outputs: Whether to save the estimator outputs (useful for
                off-policy learning with a tempered policy).

        Returns:
            A Trajectories object containing the sampled trajectories.
        """

    def sample_terminating_states(self, env: Env, n: int) -> States:
        """Rolls out the policy and returns the terminating states.

        Args:
            env: The environment to sample terminating states from.
            n: Number of terminating states to sample.

        Returns:
            The sampled terminating states as a States object.
        """
        trajectories = self.sample_trajectories(
            env, n, save_estimator_outputs=False, save_logprobs=False
        )
        return trajectories.terminating_states

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories) -> TrainingSampleType:
        """Converts trajectories to training samples.

        Args:
            trajectories: The Trajectories object to convert.

        Returns:
            The training samples, type depends on the type of GFlowNet subclass.
        """

    @abstractmethod
    def loss(
        self,
        env: Env,
        training_objects: Any,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """Computes the loss given the training objects.

        Args:
            env: The environment where the training objects are sampled from.
            training_objects: The objects to compute the loss with.
            recalculate_all_logprobs: If True, always recalculate logprobs even if they
                exist. If False, use existing logprobs when available.

        Returns:
            The computed loss as a tensor.
        """

    def loss_from_trajectories(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """Helper method to compute loss directly from trajectories.

        This method converts trajectories to the appropriate training samples and computes
        the loss with the correct arguments based on the type of GFlowNet subclass.

        Args:
            env: The environment where the training objects are sampled from.
            trajectories: The trajectories to compute the loss with.
            recalculate_all_logprobs: If True, always recalculate logprobs even if they
                exist. If False, use existing logprobs when available.

        Returns:
            The computed loss as a tensor.
        """
        training_samples = self.to_training_samples(trajectories)

        return self.loss(
            env,
            training_samples,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

    def assert_finite_gradients(self):
        """Asserts that the gradients are finite."""
        if not self.debug:
            return
        for p in self.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise RuntimeError("GFlowNet has non-finite gradients")

    def assert_finite_parameters(self):
        """Asserts that the parameters are finite."""
        if not self.debug:
            return
        for p in self.parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError("GFlowNet has non-finite parameters")


class PFBasedGFlowNet(GFlowNet[TrainingSampleType], ABC):
    """A GFlowNet that uses forward (PF) and backward (PB) policy networks.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if it can be ignored (e.g., the
            gflownet DAG is a tree, and pb is therefore always 1).
        constant_pb: Whether to ignore the backward policy estimator.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        constant_pb: bool = False,
        log_reward_clip_min: float = float("-inf"),
        debug: bool = False,
        loss_fn: RegressionLoss | None = None,
    ) -> None:
        """Initializes a PFBasedGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree,
                and pb is therefore always 1.
            constant_pb: Whether to ignore the backward policy estimator, e.g., if the
                gflownet DAG is a tree, and pb is therefore always 1. Must be set
                explicitly by user to ensure that pb is an Estimator except under this
                special case.
            log_reward_clip_min: If finite, clips log rewards to this value.
            debug: If True, keep runtime safety checks active; disable in compiled runs.
            loss_fn: Regression loss applied to balance condition residuals.
                Defaults to :class:`~gfn.gflownet.losses.SquaredLoss`.

        """
        super().__init__(debug=debug, loss_fn=loss_fn)
        # Technical note: pb may be constant for a variety of edge cases, for example,
        # if all terminal states can be reached with exactly the same number of
        # trajectories, and we assume a uniform backward policy, then we can omit the pb
        # term (see section 6 of Discrete Probabilistic Inference as Control in
        # Multi-path Environments by Tristan Deleu, Padideh Nouri, Nikolay Malkin,
        # Doina Precup, Yoshua Bengio for more details). We do not intend to document
        # all of these cases for now.
        if pb is None and not constant_pb:
            raise ValueError(
                "pb must be an Estimator unless constant_pb is True. "
                "If you intend to ignore pb, e.g., the gflownet DAG is a tree, "
                "set constant_pb to True."
            )
        if isinstance(pb, Estimator) and constant_pb:
            warnings.warn(
                "The user specified that pb should be ignored, and specified a "
                "backward policy estimator. Under normal circumstances, pb should be "
                "None if pb is constant, (e.g., the GFlowNet DAG is a tree and "
                "the backward policy probability is always 1), because learning a "
                "backward policy estimator is not necessary and will slow down "
                "training. Please ensure this is the intended experimental setup."
            )

        self.pf = pf
        self.pb = pb

        # Propagate debug flag to estimators so that gflownet.debug
        # acts as a single entry point for enabling all validation checks.
        if debug:
            if hasattr(pf, "debug"):
                pf.debug = True
            if pb is not None and hasattr(pb, "debug"):
                pb.debug = True
        self.constant_pb = constant_pb
        self.log_reward_clip_min = log_reward_clip_min

        # Advisory: recurrent PF with non-recurrent PB is unusual
        # (tree DAGs typically prefer pb=None with constant_pb=True).
        # Import locally to avoid circular imports during module import time.
        from gfn.estimators import RecurrentDiscretePolicyEstimator

        if isinstance(self.pf, RecurrentDiscretePolicyEstimator) and isinstance(
            self.pb, Estimator
        ):
            warnings.warn(
                "Using a recurrent PF, which is only valid for tree DAGs, with a "
                "non-recurrent PB is unusual. "
                "Consider using pb=None with constant_pb=True for tree DAGs.",
            )
        # Disallow recurrent PB estimators universally.
        # I'm not actually sure we should disallow this.
        if isinstance(self.pb, RecurrentDiscretePolicyEstimator):
            raise TypeError(
                "Recurrent PB estimators are not supported. Use a non-recurrent PB "
                "or set pb=None with constant_pb=True for tree DAGs."
            )

    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditions: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples trajectories using the forward policy network.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample.
            conditions: Optional conditions tensor for conditional environments.
            save_logprobs: Whether to save the logprobs of the actions.
            save_estimator_outputs: Whether to save the estimator outputs.
            **policy_kwargs: Additional keyword arguments for the sampler.

        Returns:
            A Trajectories object containing the sampled trajectories.
        """
        sampler = Sampler(estimator=self.pf)
        trajectories = sampler.sample_trajectories(
            env,
            n=n,
            conditions=conditions,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
            **policy_kwargs,
        )

        return trajectories

    def pf_pb_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of named parameters containing 'pf' or 'pb' in their name.

        Returns:
            A dictionary of named parameters containing 'pf' or 'pb' in their name.
        """
        return {k: v for k, v in self.named_parameters() if "pb" in k or "pf" in k}

    def pf_pb_parameters(self) -> list[torch.Tensor]:
        """Returns a list of parameters containing 'pf' or 'pb' in their name.

        Returns:
            A list of parameters containing 'pf' or 'pb' in their name.
        """
        return [v for k, v in self.named_parameters() if "pb" in k or "pf" in k]


class TrajectoryBasedGFlowNet(PFBasedGFlowNet[Trajectories], ABC):
    """A GFlowNet that operates on complete trajectories.

    Attributes:
        pf: The forward policy module.
        pb: The backward policy module, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        constant_pb: Whether to ignore the backward policy estimator, e.g., if the
            gflownet DAG is a tree, and pb is therefore always 1.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates forward and backward logprobs for each trajectory in the batch.

        More specifically, it evaluates $\log P_F(s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        If recalculate_all_logprobs=True, we re-evaluate the logprobs of the
        trajectories using the current self.pf. Otherwise, the following applies:
            - If trajectories have logprobs attribute, use them - this is usually for
                on-policy learning.
            - Elif trajectories have estimator_outputs attribute, transform them into
                logprobs - this is usually for off-policy learning with a tempered policy.
            - Else (trajectories have none of them), re-evaluate the logprobs using
                the current self.pf - this is usually for off-policy learning with
                replay buffer.

        Args:
            trajectories: The Trajectories object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tuple of tensors of shape (max_length, batch_size) containing
            the log_pf and log_pb for each action in each trajectory.
        """
        return get_trajectory_pfs_and_pbs(
            self.pf,
            self.pb,
            trajectories,
            recalculate_all_logprobs,
        )

    def trajectory_log_probs_forward(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        """Evaluates forward logprobs only for each trajectory in the batch."""
        return get_trajectory_pfs(
            self.pf,
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
        )

    def trajectory_log_probs_backward(
        self,
        trajectories: Trajectories,
    ) -> torch.Tensor:
        """Evaluates backward logprobs only for each trajectory in the batch."""
        return get_trajectory_pbs(
            self.pb,
            trajectories,
        )

    def logz_named_parameters(self) -> dict[str, torch.Tensor]:
        """Returns named parameters containing 'logZ' in their name.

        Works for any subclass that registers a logZ parameter (e.g.
        :class:`TBGFlowNet`, :class:`RelativeTrajectoryBalanceGFlowNet`).
        Returns an empty dict for subclasses without logZ.
        """
        return {k: v for k, v in self.named_parameters() if "logZ" in k}

    def logz_parameters(self) -> list[torch.Tensor]:
        """Returns parameters containing 'logZ' in their name.

        Works for any subclass that registers a logZ parameter (e.g.
        :class:`TBGFlowNet`, :class:`RelativeTrajectoryBalanceGFlowNet`).
        Returns an empty list for subclasses without logZ.
        """
        return [v for k, v in self.named_parameters() if "logZ" in k]

    def get_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        env: Env | None = None,
    ) -> torch.Tensor:
        r"""Calculates scores for a batch of trajectories.

        The scores for each trajectory are defined as:
        $\log \left( \frac{P_F(\tau)}{P_B(\tau \mid x) R(x)} \right)$.

        Args:
            trajectories: The Trajectories object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            env: The environment (unused in base TB, but required by some
                subclasses such as RTB and SubTB).

        Returns:
            A tensor of shape (batch_size,) containing the scores for each trajectory.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # Guard None-checks behind debug to avoid graph breaks in torch.compile;
        # get_pfs_and_pbs always returns non-None tensors in normal operation.
        if self.debug:
            assert log_pf_trajectories is not None
        total_log_pf_trajectories = log_pf_trajectories.sum(dim=0)  # [N]
        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0)  # [N]

        # cast: log_rewards is always set for terminating trajectories;
        # assert is behind debug to avoid graph breaks in torch.compile.
        log_rewards = cast(torch.Tensor, trajectories.log_rewards)
        if self.debug:
            assert log_rewards is not None
        # Fast path: skip clamp when log_reward_clip_min is disabled (default).
        if math.isfinite(self.log_reward_clip_min):
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

        # Keep runtime safety checks under `debug` to avoid graph breaks in torch.compile.
        if self.debug:
            if torch.any(torch.isinf(total_log_pf_trajectories)):
                raise ValueError("Infinite pf logprobs found")
            if torch.any(torch.isinf(total_log_pb_trajectories)):
                raise ValueError("Infinite pb logprobs found")
            if not torch.isfinite(log_rewards).all():
                non_finite = ~torch.isfinite(log_rewards)
                raise ValueError(
                    f"Non-finite log_rewards found ({non_finite.sum().item()} "
                    f"of {log_rewards.numel()} values). This typically means "
                    f"env.reward() returned zero or negative values. "
                    f"Consider using log_reward_clip_min to clamp extreme values."
                )
            assert total_log_pf_trajectories.shape == (trajectories.batch_size,)
            assert total_log_pb_trajectories.shape == (trajectories.batch_size,)

        # Fused (pf - pb) then subtract rewards; keep it branch-free/out-of-place
        # to stay friendly to torch.compile graphs.
        scores = torch.sub(
            total_log_pf_trajectories, total_log_pb_trajectories, alpha=1.0
        )
        # Subtract rewards in a separate op to avoid in-place mutations (graph-stable)
        # while still keeping only one extra temporary.
        scores = scores - log_rewards
        return scores

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        """Returns the input trajectories as training samples.

        Args:
            trajectories: The Trajectories object to use as training samples.

        Returns:
            The same Trajectories object.
        """
        return trajectories
