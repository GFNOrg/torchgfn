import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar

import torch
import torch.nn as nn

from gfn.containers import Container, Trajectories
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.prob_calculations import get_trajectory_pfs_and_pbs

TrainingSampleType = TypeVar("TrainingSampleType", bound=Container)


def loss_reduce(loss: torch.Tensor, method: str) -> torch.Tensor:
    """Utility function to handle loss aggregation strategies.

    Args:
        loss: The tensor to reduce.
        method: The reduction method to use ('mean', 'sum', or 'none').

    Returns:
        The reduced tensor.
    """
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
    """Abstract base class for GFlowNets.

    A formal definition of GFlowNets is given in Sec. 3 of
    [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266).
    """

    log_reward_clip_min = float("-inf")  # Default off.

    @abstractmethod
    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditioning: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples a specific number of complete trajectories from the environment.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample.
            conditioning: Optional conditioning tensor for conditional environments.
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


class PFBasedGFlowNet(GFlowNet[TrainingSampleType], ABC):
    """A GFlowNet that uses forward (PF) and backward (PB) policy networks.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        dag_is_tree: Whether the gflownet DAG is a tree, and pb is therefore always 1.
    """

    def __init__(
        self, pf: Estimator, pb: Estimator | None, dag_is_tree: bool = False
    ) -> None:
        """Initializes a PFBasedGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree,
                and pb is therefore always 1.
            dag_is_tree: Whether the gflownet DAG is a tree, and pb is therefore always 1.
                Must be set explicitly by user to ensure that pb is an Estimator
                except under this special case.
        """
        super().__init__()

        if pb is None and not dag_is_tree:
            raise ValueError(
                "pb must be an Estimator unless dag_is_tree is True. "
                "If the gflownet DAG is a tree, set dag_is_tree to True."
            )
        if isinstance(pb, Estimator) and dag_is_tree:
            warnings.warn(
                "The user specified that the GFlowNet DAG is a tree, and specified a "
                "backward policy estimator. Under normal circumstances, pb should be "
                "None if the GFlowNet DAG is a tree, because the backward policy "
                "probability is always 1, and therefore learning a backward policy "
                "estimator is not necessary and will slow down training. Please ensure "
                "this is the intended experimental setup."
            )

        self.pf = pf
        self.pb = pb
        self.dag_is_tree = dag_is_tree

    def sample_trajectories(
        self,
        env: Env,
        n: int,
        conditioning: torch.Tensor | None = None,
        save_logprobs: bool = False,
        save_estimator_outputs: bool = False,
        **policy_kwargs: Any,
    ) -> Trajectories:
        """Samples trajectories using the forward policy network.

        Args:
            env: The environment to sample trajectories from.
            n: Number of trajectories to sample.
            conditioning: Optional conditioning tensor for conditional environments.
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
            conditioning=conditioning,
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


class TrajectoryBasedGFlowNet(PFBasedGFlowNet[Trajectories]):
    """A GFlowNet that operates on complete trajectories.

    Attributes:
        pf: The forward policy module.
        pb: The backward policy module, or None if the gflownet DAG is a tree, and
            pb is therefore always 1.
        dag_is_tree: Whether the gflownet DAG is a tree, and pb is therefore always 1.
    """

    def __init__(
        self, pf: Estimator, pb: Estimator | None, dag_is_tree: bool = False
    ) -> None:
        """Initializes a TrajectoryBasedGFlowNet instance.

        Args:
            pf: The forward policy estimator.
            pb: The backward policy estimator, or None if the gflownet DAG is a tree, and
                pb is therefore always 1.
            dag_is_tree: Whether the gflownet DAG is a tree, and pb is therefore always 1.
                Must be set explicitly by user to ensure that pb is an Estimator
                except under this special case.
        """
        super().__init__(pf, pb, dag_is_tree=dag_is_tree)

    def get_pfs_and_pbs(
        self,
        trajectories: Trajectories,
        fill_value: float = 0.0,
        recalculate_all_logprobs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates forward and backward logprobs for each trajectory in the batch.

        More specifically, it evaluates $\log P_F(s' \mid s)$ and $\log P_B(s \mid s')$
        for each transition in each trajectory in the batch.

        If recalculate_all_logprobs=True, we re-evaluate the logprobs of the trajectories
        using the current self.pf. Otherwise, the following applies:
            - If trajectories have logprobs attribute, use them - this is usually for
                on-policy learning.
            - Elif trajectories have estimator_outputs attribute, transform them into
                logprobs - this is usually for off-policy learning with a tempered policy.
            - Else (trajectories have none of them), re-evaluate the logprobs using
                the current self.pf - this is usually for off-policy learning with
                replay buffer.

        Args:
            trajectories: The Trajectories object to evaluate.
            fill_value: Value to use for invalid states (e.g., $s_f$ added to shorter
                trajectories).
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tuple of tensors of shape (max_length, n_trajectories) containing
            the log_pf and log_pb for each action in each trajectory.
        """
        return get_trajectory_pfs_and_pbs(
            self.pf, self.pb, trajectories, fill_value, recalculate_all_logprobs
        )

    def get_scores(
        self,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
    ) -> torch.Tensor:
        r"""Calculates scores for a batch of trajectories.

        The scores for each trajectory are defined as:
        $\log \left( \frac{P_F(\tau)}{P_B(\tau \mid x) R(x)} \right)$.

        Args:
            trajectories: The Trajectories object to evaluate.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.

        Returns:
            A tensor of shape (n_trajectories,) containing the scores for each trajectory.
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
        return total_log_pf_trajectories - total_log_pb_trajectories - log_rewards

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:
        """Returns the input trajectories as training samples.

        Args:
            trajectories: The Trajectories object to use as training samples.

        Returns:
            The same Trajectories object.
        """
        return trajectories
