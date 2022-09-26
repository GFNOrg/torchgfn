from typing import Dict, Optional

from gfn.containers.states import States
from gfn.trajectories.dist import EmpiricalTerminatingStatesDistribution

from .envs import Env
from .parametrizations import Parametrization, TBParametrization


def validate(
    env: Env,
    parametrization: Parametrization,
    n_validation_samples: int = 1000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
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
        n_visited_states = visited_terminating_states.batch_shape[0]
        n_validation_samples = min(n_visited_states, n_validation_samples)

    final_states_dist_pmf = final_states_dist.pmf()
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    validation_info = {"l1_dist": l1_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = max(logZ - true_logZ, true_logZ - logZ)
    return validation_info
