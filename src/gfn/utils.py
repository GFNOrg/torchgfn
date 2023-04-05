from typing import Dict, Optional

import torch

from gfn.containers import States, Trajectories, Transitions
from gfn.distributions import EmpiricalTerminatingStatesDistribution
from gfn.envs import Env
from gfn.losses import (
    EdgeDecomposableLoss,
    Loss,
    Parametrization,
    StateDecomposableLoss,
    TBParametrization,
    TrajectoryDecomposableLoss,
)


def trajectories_to_training_samples(
    trajectories: Trajectories, loss_fn: Loss
) -> States | Transitions | Trajectories:
    """Converts a Trajectories container to a States, Transitions or Trajectories container,
    depending on the loss.
    """
    if isinstance(loss_fn, StateDecomposableLoss):
        # return trajectories.to_states()
        return trajectories.to_non_initial_intermediary_and_terminating_states()
    elif isinstance(loss_fn, TrajectoryDecomposableLoss):
        return trajectories
    elif isinstance(loss_fn, EdgeDecomposableLoss):
        return trajectories.to_transitions()
    else:
        raise ValueError(f"Loss {loss_fn} is not supported.")


def validate(
    env: Env,
    parametrization: Parametrization,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    """Evaluates the current parametrization on the given environment.
    This is for environments with known target reward. The validation is done by computing the l1 distance between the
    learned empirical and the target distributions.

    Args:
        env: The environment to evaluate the parametrization on.
        parametrization: The parametrization to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the l1 validation metric. If the parametrization is a TBParametrization,
        i.e. contains LogZ, then the (absolute) difference between the learned and the target LogZ is also returned in the
        dictionary.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the parametrization
        return {}

    logZ = None
    if isinstance(parametrization, TBParametrization):
        logZ = parametrization.logZ.tensor.item()
    if visited_terminating_states is None:
        final_states_dist = parametrization.P_T(env, n_validation_samples)
    else:
        final_states_dist = EmpiricalTerminatingStatesDistribution(
            env, visited_terminating_states[-n_validation_samples:]
        )
        n_visited_states = visited_terminating_states.batch_shape[0]
        n_validation_samples = min(n_visited_states, n_validation_samples)

    final_states_dist_pmf = final_states_dist.pmf()
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info
