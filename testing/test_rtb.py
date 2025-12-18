import torch

from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import RelativeTrajectoryBalanceGFlowNet
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP


def _make_hypergrid_estimators():
    """Build simple forward policies for HyperGrid prior/posterior."""
    env = HyperGrid(ndim=2, height=4)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)

    pf_module_post = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)
    pf_module_prior = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)

    pf_post = DiscretePolicyEstimator(
        module=pf_module_post,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pf_prior = DiscretePolicyEstimator(
        module=pf_module_prior,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    return env, pf_post, pf_prior


def test_rtb_loss_backward_and_grads():
    torch.manual_seed(0)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=1.0,
    )
    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=8, save_logprobs=True, save_estimator_outputs=False
    )

    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=True)
    assert torch.isfinite(loss)

    loss.backward()

    # Posterior parameters and logZ should receive gradients.
    assert any(p.grad is not None for p in pf_post.parameters())
    assert any(p.grad is not None for p in gfn.logz_parameters())

    # Prior parameters are not part of the RTB graph and should have no grads.
    assert all(p.grad is None for p in pf_prior.parameters())


def test_rtb_loss_forward_only_path():
    """Ensure RTB loss works with recalculate_all_logprobs=False."""
    torch.manual_seed(1)
    env, pf_post, pf_prior = _make_hypergrid_estimators()

    gfn = RelativeTrajectoryBalanceGFlowNet(
        pf=pf_post,
        prior_pf=pf_prior,
        init_logZ=0.0,
        beta=0.5,
    )
    sampler = Sampler(estimator=pf_post)
    trajectories = sampler.sample_trajectories(
        env, n=4, save_logprobs=True, save_estimator_outputs=False
    )

    # Use cached log_probs; should not rely on any backward policy.
    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)
    loss.backward()
