import math
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchtyping import TensorType as TT

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import GFNModule
from gfn.samplers import Sampler
from gfn.states import States


class GFlowNet(nn.Module):
    """Abstract Base Class for GFlowNets. This is always On Policy.

    A formal definition of GFlowNets is given in Sec. 3 of [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266).
    """

    @abstractmethod
    def sample_trajectories(self, env: Env, n_samples: int) -> Trajectories:
        """Sample a specific number of complete trajectories.

        Args:
            env: the environment to sample trajectories from.
            n_samples: number of trajectories to be sampled.
        Returns:
            Trajectories: sampled trajectories object.
        """

    def sample_terminating_states(self, env: Env, n_samples: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            env: the environment to sample terminating states from.
            n_samples: number of terminating states to be sampled.
        Returns:
            States: sampled terminating states object.
        """
        trajectories = self.sample_trajectories(env, n_samples, sample_off_policy=False)
        return trajectories.last_states

    def pf_pb_named_parameters(self):
        return {k: v for k, v in self.named_parameters() if "pb" in k or "pf" in k}

    def pf_pb_parameters(self):
        return [v for k, v in self.named_parameters() if "pb" in k or "pf" in k]

    def logz_named_parameters(self):
        return {"logZ": dict(self.named_parameters())["logZ"]}

    def logz_parameters(self):
        return [dict(self.named_parameters())["logZ"]]

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories):
        """Converts trajectories to training samples. The type depends on the GFlowNet."""

    @abstractmethod
    def loss(self, env: Env, training_objects):
        """Computes the loss given the training objects."""


class PFBasedGFlowNet(GFlowNet):
    r"""Base class for gflownets that explicitly uses $P_F$.

    Attributes:
        pf: GFNModule
        pb: GFNModule
    """

    def __init__(self, pf: GFNModule, pb: GFNModule, off_policy: bool):
        super().__init__()
        self.pf = pf
        self.pb = pb
        self.off_policy = off_policy

    def sample_trajectories(
        self, env: Env, n_samples: int, sample_off_policy: bool, **policy_kwargs
    ) -> Trajectories:
        """Samples trajectories, optionally with specified policy kwargs."""
        sampler = Sampler(estimator=self.pf)
        trajectories = sampler.sample_trajectories(
            env,
            n_trajectories=n_samples,
            off_policy=sample_off_policy,
            **policy_kwargs,
        )

        return trajectories


class TrajectoryBasedGFlowNet(PFBasedGFlowNet):
    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
    ) -> Tuple[
        TT["max_length", "n_trajectories", torch.float],
        TT["max_length", "n_trajectories", torch.float],
    ]:
        r"""Evaluates logprobs for each transition in each trajectory in the batch.

        More specifically it evaluates $\log P_F (s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        Useful when the policy used to sample the trajectories is different from
        the one used to evaluate the loss. Otherwise we can use the logprobs directly
        from the trajectories.

        Note - for off policy exploration, the trajectories submitted to this method
        will be sampled off policy.

        Args:
            trajectories: Trajectories to evaluate.
            estimator_outputs: Optional stored estimator outputs from previous forward
                sampling (encountered, for example, when sampling off policy).
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

        if self.off_policy:
            # We re-use the values calculated in .sample_trajectories().
            if not isinstance(trajectories.estimator_outputs, type(None)):
                estimator_outputs = trajectories.estimator_outputs[
                    ~trajectories.actions.is_dummy
                ]
            else:
                raise Exception(
                    "GFlowNet is off policy, but no estimator_outputs found in Trajectories!"
                )

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

        else:
            log_pf_trajectories = trajectories.log_probs

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
        self, trajectories: Trajectories
    ) -> Tuple[
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
    ]:
        """Given a batch of trajectories, calculate forward & backward policy scores."""
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(trajectories)

        assert log_pf_trajectories is not None
        total_log_pf_trajectories = log_pf_trajectories.sum(dim=0)
        total_log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        log_rewards = trajectories.log_rewards
        if math.isfinite(self.log_reward_clip_min) and not isinstance(
            log_rewards, type(None)
        ):
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
