import torch

from gfn.gym.diffusion_sampling import (
    DiffusionSampling,
    Grid25GaussianMixture,
    Posterior9of25GaussianMixture,
)


def test_gmm25_prior_basic_sampling_and_log_reward():
    env = DiffusionSampling(
        target_str="gmm25_prior",
        target_kwargs=None,
        num_discretization_steps=8,
        device=torch.device("cpu"),
        debug=True,
    )
    assert isinstance(env.target, Grid25GaussianMixture)
    x = env.target.sample(batch_size=16)
    assert x.shape == (16, env.dim)
    log_r = env.target.log_reward(x)
    assert log_r.shape == (16,)
    assert torch.isfinite(log_r).all()


def test_gmm25_posterior9_log_reward_matches_ratio():
    env = DiffusionSampling(
        target_str="gmm25_posterior9",
        target_kwargs=None,
        num_discretization_steps=8,
        device=torch.device("cpu"),
        debug=True,
    )
    assert isinstance(env.target, Posterior9of25GaussianMixture)
    x = env.target.sample(batch_size=8)
    assert x.shape == (8, env.dim)

    log_r = env.target.log_reward(x)
    posterior_log = env.target.posterior.log_prob(x).flatten()
    prior_log = env.target.prior.log_reward(x)

    assert torch.allclose(log_r, posterior_log - prior_log, atol=1e-5)
    assert torch.isfinite(log_r).all()
