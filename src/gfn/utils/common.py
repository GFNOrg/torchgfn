import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import torch
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.envs import Env
from gfn.losses import (
    DBParametrization,
    FMParametrization,
    Parametrization,
    TBParametrization,
    TrajectoryDecomposableLoss,
)
from gfn.states import States


def trajectories_to_training_samples(
    trajectories: Trajectories, parametrization: Parametrization
) -> States | Transitions | Trajectories:
    """Converts Trajectories into States, Transitions or Trajectories.

    This converts a Trajectories container into a States, Transitions, or Trajectories
    container, depending on the parametrization.

    Args:
        trajectories: a Trajectories container.
        parametrization: Parametrization instance.

    Raises:
        ValueError: if the submitted Loss is not currently suppored by the function.
    """
    if isinstance(parametrization, FMParametrization):
        # return trajectories.to_states()
        return trajectories.to_non_initial_intermediary_and_terminating_states()
    elif isinstance(parametrization, TrajectoryDecomposableLoss):
        return trajectories
    elif isinstance(parametrization, DBParametrization):
        return trajectories.to_transitions()
    else:
        raise ValueError(f"Parametrization {parametrization} not supported.")


def get_terminating_state_dist_pmf(env: Env, states: States) -> TT["n_states", float]:
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(states_indices)
    counter_list = [
        counter[state_idx] if state_idx in counter else 0
        for state_idx in range(env.n_terminating_states)
    ]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: Env,
    parametrization: Parametrization,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    """Evaluates the current parametrization on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the parametrization on.
        parametrization: The parametrization to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns: A dictionary containing the l1 validation metric. If the parametrization
        is a TBParametrization, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary.
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
        terminating_states = parametrization.sample_terminating_states(
            n_validation_samples
        )
    else:
        terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info


def get_root() -> Path:
    """
    Returns the root directory of the project.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def add_root_to_path():
    """
    Adds the root directory of the project to the python path.
    """
    sys.path.append(str(get_root()))
