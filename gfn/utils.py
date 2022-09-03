from typing import Dict, Optional

from gfn.containers.states import States
from gfn.trajectories.dist import EmpiricalTerminatingStatesDistribution

from .envs import HyperGrid
from .envs.utils import get_flat_grid, get_true_dist_pmf
from .parametrizations import Parametrization, TBParametrization


def get_hypergrid_statistics(env: HyperGrid):
    true_dist_pmf = get_true_dist_pmf(env)

    flat_grid = get_flat_grid(env)

    all_rewards = env.reward(flat_grid)
    true_logZ = all_rewards.sum().log().item()
    return true_logZ, true_dist_pmf


def hypergrid_validate(
    env: HyperGrid,
    parametrization: Parametrization,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    true_logZ, true_dist_pmf = get_hypergrid_statistics(env)
    true_dist_pmf = true_dist_pmf.cpu()

    logZ = None
    if isinstance(parametrization, TBParametrization):
        logZ = parametrization.logZ.tensor.item()
    if visited_terminating_states is None:
        final_states_dist = parametrization.P_T(env, n_validation_samples)
    else:
        final_states_dist = EmpiricalTerminatingStatesDistribution(
            env, visited_terminating_states[-n_validation_samples:]
        )
    final_states_dist_pmf = final_states_dist.pmf()
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = logZ - true_logZ
    return validation_info
