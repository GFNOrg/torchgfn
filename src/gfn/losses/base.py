import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.estimators import ProbabilityEstimator
from gfn.samplers import ActionsSampler, TrajectoriesSampler
from gfn.states import States


@dataclass
class Parametrization(ABC):
    """Abstract Base Class for Flow Parametrizations.

    Flow paramaterizations are defined in Sec. 3 of [GFlowNets Foundations](link).

    All attributes should be estimators, and should either have a GFNModule or attribute
    called `module`, or torch.Tensor attribute called `tensor` with requires_grad=True.
    """

    @abstractmethod
    def sample_trajectories(self, n_samples: int) -> Trajectories:
        """Sample a specific number of complete trajectories.

        Args:
            n_samples: number of trajectories to be sampled.

        Returns:
            Trajectories: sampled trajectories object.
        """
        pass

    def sample_terminating_states(self, n_samples: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            n_samples: number of terminating states to be sampled.

        Returns:
            States: sampled terminating states object.
        """
        trajectories = self.sample_trajectories(n_samples)
        return trajectories.last_states

    @property
    def parameters(self) -> dict:
        """
        Return a dictionary of all parameters of the parametrization.
        Note that there might be duplicate parameters (e.g. when two NNs share parameters),
        in which case the optimizer should take as input set(self.parameters.values()).
        """
        # TODO: use parameters of the fields instead, loop through them here
        parameters_dict = {}
        for name, estimator in self.__dict__.items():
            parameters_dict.update(
                {
                    f"{name}_{key}": value  # TODO: fix name for logZ
                    for key, value in estimator.named_parameters().items()
                }
            )
        return parameters_dict

    def save_state_dict(self, path: str):
        for name, estimator in self.__dict__.items():
            torch.save(estimator.named_parameters(), os.path.join(path, name + ".pt"))

    def load_state_dict(self, path: str):
        for name, estimator in self.__dict__.items():
            estimator.load_state_dict(torch.load(os.path.join(path, name + ".pt")))


@dataclass
class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly uses $P_F$"
    pf: ProbabilityEstimator
    pb: ProbabilityEstimator

    def sample_trajectories(self, n_samples: int = 1000) -> Trajectories:
        actions_sampler = ActionsSampler(self.pf)
        trajectories_sampler = TrajectoriesSampler(actions_sampler)
        trajectories = trajectories_sampler.sample_trajectories(
            n_trajectories=n_samples
        )
        return trajectories


class Loss(ABC):
    """Abstract Base Class for all GFN Losses."""

    def __init__(self, parametrization: Parametrization):
        self.parametrization = parametrization

    @abstractmethod
    def __call__(self, *args, **kwargs) -> TT[0, float]:
        pass


class EdgeDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, edges: Transitions) -> TT[0, float]:
        pass


class StateDecomposableLoss(Loss, ABC):
    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TT[0, float]:
        """Given a batch of non-terminal and terminal states, compute a loss.

        Unlike the GFlowNets Foundations paper, we allow more flexibility by passing a
        tuple of states, the first one being the internal states of the trajectories
        (i.e. non-terminal states), and the second one being the terminal states of the
        trajectories. If these two are not handled differently, then they should be
        concatenated together.
        """
        pass


class TrajectoryDecomposableLoss(Loss, ABC):
    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
        no_pf: bool = False,
    ) -> Tuple[
        TT["max_length", "n_trajectories", torch.float] | None,
        TT["max_length", "n_trajectories", torch.float],
    ]:
        """Evaluates logprobs for each transition in each trajectory in the batch.

        More specifically it evaluates $\log P_F (s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        This is useful when the policy used to sample the trajectories is different from
        the one used to evaluate the loss. Otherwise we can use the logprobs directly
        from the trajectories.

        Args:
            trajectories: Trajectories to evaluate.
            fill_value: Value to use for invalid states (i.e. $s_f$ that is added to
                shorter trajectories).
            no_pf: Whether to evaluate log_pf as well.

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
        # assert trajectories.states.is_sink_state[:-1].equal(trajectories.actions == -1)

        if valid_states.batch_shape != tuple(valid_actions.batch_shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")

        log_pf_trajectories = None
        if not no_pf:
            valid_log_pf_actions = self.parametrization.pf(valid_states).log_prob(
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

        valid_log_pb_actions = self.parametrization.pb(
            non_initial_valid_states
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
        log_pf_trajectories, log_pb_trajectories = self.get_pfs_and_pbs(
            trajectories, no_pf=self.on_policy
        )
        if self.on_policy:
            log_pf_trajectories = trajectories.log_probs

        assert log_pf_trajectories is not None
        log_pf_trajectories = log_pf_trajectories.sum(dim=0)
        log_pb_trajectories = log_pb_trajectories.sum(dim=0)

        log_rewards = trajectories.log_rewards.clamp_min(self.log_reward_clip_min)  # type: ignore

        return (
            log_pf_trajectories,
            log_pb_trajectories,
            log_pf_trajectories - log_pb_trajectories - log_rewards,
        )

    @abstractmethod
    def __call__(self, trajectories: Trajectories) -> TT[0, float]:
        pass
