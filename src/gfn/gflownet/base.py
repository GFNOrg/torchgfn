import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar

import torch
import torch.nn as nn

from gfn.containers import Container, Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.common import has_log_probs
from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs

TrainingSampleType = TypeVar("TrainingSampleType", bound=Container)


def loss_reduce(loss: torch.Tensor, method: str) -> torch.Tensor:
    """Utility function to handle loss aggregation strategies."""
    reduction_methods = {
        "mean": torch.mean,
        "sum": torch.sum,
        "none": lambda x: x,
    }
    if method in reduction_methods:
        return reduction_methods[method](loss)
    else:
        raise ValueError(
            f"Invalid loss reduction method: {method} not in {reduction_methods.keys()}"
        )


class GFlowNet(ABC, nn.Module, Generic[TrainingSampleType]):
    """Abstract Base Class for GFlowNets.

    A formal definition of GFlowNets is given in Sec. 3 of [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266).
    """

    log_reward_clip_min = float("-inf")  # Default off.

    @abstractmethod
    def sample_trajectories(
        self,
        env: Env,
        n: int,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
    ) -> Trajectories:
        """Sample a specific number of complete trajectories.

        Args:
            env: the environment to sample trajectories from.
            n: number of trajectories to be sampled.
            save_logprobs: whether to save the logprobs of the actions - useful for on-policy learning.
            save_estimator_outputs: whether to save the estimator outputs - useful for off-policy learning
                        with tempered policy
        Returns:
            Trajectories: sampled trajectories object.
        """

    def sample_terminating_states(self, env: Env, n: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            env: the environment to sample terminating states from.
            n: number of terminating states to be sampled.
        Returns:
            States: sampled terminating states object.
        """
        trajectories = self.sample_trajectories(
            env, n, save_estimator_outputs=False, save_logprobs=False
        )
        return trajectories.last_states

    def logz_named_parameters(self):
        return {k: v for k, v in dict(self.named_parameters()).items() if "logZ" in k}

    def logz_parameters(self):
        return [v for k, v in dict(self.named_parameters()).items() if "logZ" in k]

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories) -> TrainingSampleType:
        """Converts trajectories to training samples. The type depends on the GFlowNet."""

    @abstractmethod
    def loss(
        self, env: Env, training_objects: Any, reduction: str | None = None
    ) -> torch.Tensor:
        """Computes the loss given the training objects."""

    def loss_from_trajectories(
        self, env: Env, trajectories: Trajectories
    ) -> torch.Tensor:
        """Helper method to compute loss directly from trajectories.

        This method handles converting trajectories to the appropriate training samples
        and computing the loss with the correct arguments based on the GFlowNet type.

        Args:
            env: The environment to compute the loss for
            trajectories: The trajectories to compute the loss from

        Returns:
            torch.Tensor: The computed loss
        """
        training_samples = self.to_training_samples(trajectories)
        if isinstance(self, PFBasedGFlowNet):
            # Check if trajectories already have log_probs
            if has_log_probs(trajectories):
                warnings.warn(
                    "Recalculating logprobs for trajectories that already have them. "
                    "This may be inefficient for on-policy trajectories. "
                    "If the training is done on-policy, you should call loss() directly "
                    "with recalculate_all_logprobs=False instead of loss_from_trajectories()."
                )

            # We know this is safe because PFBasedGFlowNet's loss accepts these arguments
            return self.loss(env, training_samples, recalculate_all_logprobs=True)
        return self.loss(env, training_samples)


class PFBasedGFlowNet(GFlowNet[TrainingSampleType], ABC):
    """A GFlowNet that uses forward (PF) and backward (PB) policy networks."""

    def __init__(self, pf: GFNModule, pb: GFNModule):
        super().__init__()
        self.pf = pf
        self.pb = pb

    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditioning: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples trajectories, optionally with specified policy kwargs."""
        sampler = Sampler(estimator=self.pf)
        trajectories = sampler.sample_trajectories(
            env,
            n=n,
            conditioning=conditioning,
            save_logprobs=save_logprobs,
            save_estimator_outputs=save_estimator_outputs,
            **policy_kwargs,
        )

        return trajectories

    def pf_pb_named_parameters(self):
        return {k: v for k, v in self.named_parameters() if "pb" in k or "pf" in k}

    def pf_pb_parameters(self):
        return [v for k, v in self.named_parameters() if "pb" in k or "pf" in k]

    @abstractmethod
    def loss(
        self,
        env: Env,
        training_objects: Any,
        recalculate_all_logprobs: bool = False,
        reduction: str | None = None,
    ) -> torch.Tensor:
        """Computes the loss given the training objects.

        Args:
            env: The environment to compute the loss for
            training_objects: The objects to compute the loss on
            recalculate_all_logprobs: If True, always recalculate logprobs even if they exist.
                                     If False, use existing logprobs when available.
            reduction: The reduction to apply to the loss.
            **kwargs: Additional arguments specific to the loss
        """


class TrajectoryBasedGFlowNet(PFBasedGFlowNet[Trajectories]):
    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
        recalculate_all_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates logprobs for each transition in each trajectory in the batch.

        More specifically it evaluates $\log P_F (s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        Unless recalculate_all_logprobs=True, in which case we re-evaluate the logprobs of the trajectories with
        the current self.pf. The following applies:
            - If trajectories have log_probs attribute, use them - this is usually for on-policy learning
            - Else, if trajectories have estimator_outputs attribute, transform them
                into log_probs - this is usually for off-policy learning with a tempered policy
            - Else, if trajectories have none of them, re-evaluate the log_probs
                using the current self.pf - this is usually for off-policy learning with replay buffer

        Args:
            trajectories: Trajectories to evaluate.
            fill_value: Value to use for invalid states (i.e. $s_f$ that is added to
                shorter trajectories).
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns: A tuple of float tensors of shape (max_length, n_trajectories) containing
            the log_pf and log_pb for each action in each trajectory. The first one can be None.

        Raises:
            ValueError: if the trajectories are backward.
            AssertionError: when actions and states dimensions mismatch.
        """
        return get_trajectory_pfs_and_pbs(
            self.pf, self.pb, trajectories, fill_value, recalculate_all_logprobs
        )

    def get_trajectories_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a batch of trajectories, calculate forward & backward policy scores.

        Args:
            trajectories: Trajectories to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns: A tuple of float tensors of shape (n_trajectories,)
            containing the total log_pf, total log_pb, and the total
            log-likelihood of the trajectories.
        """
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        assert log_pf_trajectories is not None
        total_log_pf_trajectories = log_pf_trajectories.sum(dim=0)
        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        log_rewards = trajectories.log_rewards

        if math.isfinite(self.log_reward_clip_min) and log_rewards is not None:
            log_rewards = log_rewards.clamp_min(self.log_reward_clip_min)

        if torch.any(torch.isinf(total_log_pf_trajectories)) or torch.any(
            torch.isinf(total_log_pb_trajectories)
        ):
            raise ValueError("Infinite logprobs found")

        assert total_log_pf_trajectories.shape == (trajectories.n_trajectories,)
        assert total_log_pb_trajectories.shape == (trajectories.n_trajectories,)
        assert log_rewards is not None
        return (
            total_log_pf_trajectories,
            total_log_pb_trajectories,
            total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        )

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        return trajectories
