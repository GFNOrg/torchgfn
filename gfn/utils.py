from .envs import HyperGrid
from .envs.utils import get_flat_grid, get_true_dist_pmf
from .parametrizations import TBParametrization


def get_hypergrid_statistics(env: HyperGrid):
    true_dist_pmf = get_true_dist_pmf(env)

    flat_grid = get_flat_grid(env)

    all_rewards = env.reward(flat_grid)
    true_logZ = all_rewards.sum().log().item()
    return true_logZ, true_dist_pmf


def validate_TB_parametrization_for_HyperGrid(
    env: HyperGrid, parametrization: TBParametrization, n_validation_samples=1000
):
    true_logZ, true_dist_pmf = get_hypergrid_statistics(env)
    logZ = parametrization.logZ.tensor.item()

    final_states_dist = parametrization.P_T(env, n_validation_samples)
    final_states_dist_pmf = final_states_dist.pmf()
    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    return {"logZ": logZ, "l1_dist": l1_dist, "true_logZ": true_logZ}
