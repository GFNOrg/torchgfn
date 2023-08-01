from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn
from torchtyping import TensorType as TT
import torch

from gfn.containers import Trajectories
from gfn.modules import GFNModule
from gfn.samplers import ActionsSampler, TrajectoriesSampler
from gfn.states import States


class GFlowNet(nn.Module):
    @abstractmethod
    def sample_trajectories(self, n_samples: int) -> Trajectories:
        pass

    def sample_terminating_states(self, n_samples: int) -> States:
        trajectories = self.sample_trajectories(n_samples)
        return trajectories.last_states


class PFBasedGFlowNet(GFlowNet):
    r"""Base class for gflownets that explicitly uses $P_F$.

    Attributes:
        pf: GFNModule
        pb: GFNModule
    """

    def __init__(self, pf: GFNModule, pb: GFNModule, on_policy: bool = False):
        super().__init__()
        self.pf = pf
        self.pb = pb
        self.on_policy = on_policy

    def sample_trajectories(self, n_samples: int = 1000) -> Trajectories:
        actions_sampler = ActionsSampler(self.pf)
        trajectories_sampler = TrajectoriesSampler(actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return trajectories


# TODO: Add "EdgeBasedGFlowNet" and "StateBasedGFlowNet".
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

        if self.on_policy:
            log_pf_trajectories = trajectories.log_probs
        else:
            module_output = self.pf(valid_states)
            valid_log_pf_actions = self.pf.to_probability_distribution(
                valid_states,
                module_output
            ).log_prob(
                valid_actions.tensor
            )
            log_pf_trajectories = torch.full_like(
                trajectories.actions.tensor[..., 0],
                fill_value=fill_value,
                dtype=torch.float,
            )
            log_pf_trajectories[~trajectories.actions.is_dummy] = valid_log_pf_actions

        non_initial_valid_states = valid_states[~valid_states.is_initial_state]
        non_exit_valid_actions = valid_actions[~valid_actions.is_exit]

        module_output = self.pb(non_initial_valid_states)
        valid_log_pb_actions = self.pb.to_probability_distribution(
            non_initial_valid_states,
            module_output
        ).log_prob(
            non_exit_valid_actions.tensor
        )

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

        log_rewards = trajectories.log_rewards.clamp_min(self.log_reward_clip_min)  # type: ignore
        if torch.any(torch.isinf(total_log_pf_trajectories)) or torch.any(
            torch.isinf(total_log_pb_trajectories)
        ):
            raise ValueError("Infinite logprobs found")
        return (
            total_log_pf_trajectories,
            total_log_pb_trajectories,
            total_log_pf_trajectories - total_log_pb_trajectories - log_rewards,
        )
