import math
from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

from gfn.containers import Trajectories
from gfn.containers.base import Container
from gfn.env import DiscreteEnv, Env
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from gfn.states import DiscreteStates, States
from gfn.utils.common import has_log_probs

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
        n_samples: int,
        save_logprobs: bool = True,
        save_estimator_outputs: bool = False,
    ) -> Trajectories:
        """Sample a specific number of complete trajectories.

        Args:
            env: the environment to sample trajectories from.
            n_samples: number of trajectories to be sampled.
            save_logprobs: whether to save the logprobs of the actions - useful for on-policy learning.
            save_estimator_outputs: whether to save the estimator outputs - useful for off-policy learning
                        with tempered policy
        Returns:
            Trajectories: sampled trajectories object.
        """

    @overload
    def sample_terminating_states(
        self, env: DiscreteEnv, n_samples: int
    ) -> DiscreteStates: ...

    @overload
    def sample_terminating_states(self, env: Env, n_samples: int) -> States: ...

    def sample_terminating_states(self, env: Env, n_samples: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            env: the environment to sample terminating states from.
            n_samples: number of terminating states to be sampled.
        Returns:
            States: sampled terminating states object.
        """
        trajectories = self.sample_trajectories(
            env, n_samples, save_estimator_outputs=False, save_logprobs=False
        )
        return trajectories.last_states

    def logz_named_parameters(self):
        return {"logZ": dict(self.named_parameters())["logZ"]}

    def logz_parameters(self):
        return [dict(self.named_parameters())["logZ"]]

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories) -> TrainingSampleType:
        """Converts trajectories to training samples. The type depends on the GFlowNet."""

    @abstractmethod
    def loss(self, env: Env, training_objects):
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
        n_samples: int,
        save_logprobs: bool = True,
        save_estimator_outputs: bool = False,
        **policy_kwargs,
    ) -> Trajectories:
        """Samples trajectories, optionally with specified policy kwargs."""
        sampler = Sampler(estimator=self.pf)
        trajectories = sampler.sample_trajectories(
            env,
            n_trajectories=n_samples,
            save_estimator_outputs=save_estimator_outputs,
            save_logprobs=save_logprobs,
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
    ) -> Tuple[
        Tensor,
        Tensor,
    ]:
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

        Returns: A tuple of float tensors of shape (max_length, n_trajectories) containing
            the log_pf and log_pb for each action in each trajectory. The first one can be None.

        Raises:
            ValueError: if the trajectories are backward.
            AssertionError: when actions and states dimensions mismatch.
        """
        # fill value is the value used for invalid states (sink state usually)
        if trajectories.is_backward:
            raise ValueError("Backward trajectories are not supported")

        valid_states = trajectories.states[~trajectories.states.is_sink_state]
        valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

        # uncomment next line for debugging
        # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions.is_dummy)

        if valid_states.batch_shape != tuple(valid_actions.batch_shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")

        if has_log_probs(trajectories) and not recalculate_all_logprobs:
            log_pf_trajectories = trajectories.log_probs
        else:
            if (
                trajectories.estimator_outputs is not None
                and not recalculate_all_logprobs
            ):
                estimator_outputs = trajectories.estimator_outputs[
                    ~trajectories.actions.is_dummy
                ]
            else:
                estimator_outputs = self.pf(valid_states)

            # Calculates the log PF of the actions sampled off policy.
            valid_log_pf_actions = self.pf.to_probability_distribution(
                valid_states, estimator_outputs
            ).log_prob(
                valid_actions.tensor
            )  # Using the actions sampled off-policy.
            log_pf_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.float,
            )
            log_pf_trajectories[~trajectories.actions.is_dummy] = valid_log_pf_actions

        non_initial_valid_states = valid_states[~valid_states.is_initial_state]
        non_exit_valid_actions = valid_actions[~valid_actions.is_exit]

        # Using all non-initial states, calculate the backward policy, and the logprobs
        # of those actions.
        estimator_outputs = self.pb(non_initial_valid_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            non_initial_valid_states, estimator_outputs
        ).log_prob(non_exit_valid_actions.tensor)

        log_pb_trajectories = torch.full_like(
            trajectories.actions.tensor[..., 0],
            fill_value=fill_value,
            dtype=torch.float,
        )
        log_pb_trajectories_slice = torch.full_like(
            valid_actions.tensor[..., 0], fill_value=fill_value, dtype=torch.float
        )
        log_pb_trajectories_slice[~valid_actions.is_exit] = valid_log_pb_actions
        log_pb_trajectories[~trajectories.actions.is_dummy] = log_pb_trajectories_slice

        return log_pf_trajectories, log_pb_trajectories

    def get_trajectories_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Given a batch of trajectories, calculate forward & backward policy scores."""
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
        return (
            total_log_pf_trajectories,
            total_log_pb_trajectories,
            total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        )

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        return trajectories
