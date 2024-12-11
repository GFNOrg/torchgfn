import math
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar, Union

import torch
import torch.nn as nn

from gfn.containers import Trajectories
from gfn.containers.base import Container
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs

TrainingSampleType = TypeVar(
    "TrainingSampleType", bound=Union[Container, tuple[States, ...]]
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
        save_logprobs: bool = True,
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
    def loss(self, env: Env, training_objects: Any) -> torch.Tensor:
        """Computes the loss given the training objects."""


class PFBasedGFlowNet(GFlowNet[TrainingSampleType]):
    r"""Base class for gflownets that explicitly uses $P_F$.

    Attributes:
        pf: GFNModule
        pb: GFNModule
    """

    def __init__(self, pf: GFNModule, pb: GFNModule):
        super().__init__()
        self.pf = pf
        self.pb = pb

    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditioning: torch.Tensor | None = None,
        save_logprobs: bool = True,
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
        return (
            total_log_pf_trajectories,
            total_log_pb_trajectories,
            total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        )

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        return trajectories
