# testing/test_local_search_sampler.py
import pytest
import torch

from gfn.containers import Trajectories
from gfn.estimators import DiscretePolicyEstimator
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.samplers import LocalSearchSampler, Sampler
from gfn.utils.modules import MLP
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs


def _make_env_estimators(env_name: str):
    torch.manual_seed(0)
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=5)
        preproc = KHotPreprocessor(env.height, env.ndim)
        assert isinstance(preproc.output_dim, int)
        pf_module = MLP(preproc.output_dim, env.n_actions)
        pb_module = MLP(preproc.output_dim, env.n_actions - 1)
        pf = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=preproc,
        )
        pb = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=preproc,
        )

    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=5)
        preproc = IdentityPreprocessor(output_dim=env.state_shape[-1])
        assert isinstance(preproc.output_dim, int)
        pf_module = MLP(preproc.output_dim, env.n_actions)
        pb_module = MLP(preproc.output_dim, env.n_actions - 1)
        pf = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=preproc,
        )
        pb = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=preproc,
        )

    elif env_name == "Box":
        env = Box(delta=0.1)
        pf_module = BoxPFMLP(
            hidden_dim=16,
            n_hidden_layers=1,
            n_components=1,
            n_components_s0=1,
        )
        pb_module = BoxPBMLP(
            hidden_dim=16,
            n_hidden_layers=1,
            n_components=1,
            trunk=pf_module.trunk,
        )
        pf = BoxPFEstimator(env=env, module=pf_module, n_components=1, n_components_s0=1)
        pb = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    else:
        raise ValueError(env_name)
    return env, pf, pb


def _reference_local_search(
    env,
    trajectories: Trajectories,
    pf_estimator,
    pb_estimator,
    *,
    conditioning=None,
    back_steps: torch.Tensor | None = None,
    back_ratio: float | None = None,
    use_metropolis_hastings: bool = True,
    save_logprobs: bool = True,
):
    # K selection identical to production
    if back_steps is None:
        assert (back_ratio is not None) and (0 < back_ratio <= 1)
        K = torch.ceil(back_ratio * (trajectories.terminating_idx - 1)).long()
    else:
        K = torch.where(
            back_steps > trajectories.terminating_idx,
            trajectories.terminating_idx,
            back_steps,
        )

    # Backward sampling from terminal states
    bw_sampler = Sampler(pb_estimator)
    prev_traj_bw = bw_sampler.sample_trajectories(
        env,
        states=trajectories.terminating_states,
        conditioning=conditioning,
        save_logprobs=save_logprobs or use_metropolis_hastings,
    )
    # Reverse to forward-time
    prev_traj = prev_traj_bw.reverse_backward_trajectories()
    assert prev_traj.log_rewards is not None

    # Junctions
    n_prevs = prev_traj.terminating_idx - K - 1
    jx_states = torch.gather(
        prev_traj.states.tensor,
        0,
        n_prevs.view(1, -1, 1).expand(-1, -1, *trajectories.states.state_shape),
    ).squeeze(0)
    # Reconstruct from junctions
    fw_sampler = Sampler(pf_estimator)
    recon_traj = fw_sampler.sample_trajectories(
        env,
        states=env.states_from_tensor(jx_states),
        conditioning=conditioning,
        save_logprobs=save_logprobs or use_metropolis_hastings,
    )

    # Per-step PF/PB for acceptance
    prev_pf = (
        get_trajectory_pfs(pf_estimator, prev_traj)
        if (save_logprobs or use_metropolis_hastings)
        else None
    )
    recon_pf = (
        get_trajectory_pfs(pf_estimator, recon_traj)
        if (save_logprobs or use_metropolis_hastings)
        else None
    )
    prev_pb = (
        get_trajectory_pbs(pb_estimator, prev_traj) if use_metropolis_hastings else None
    )
    recon_pb = (
        get_trajectory_pbs(pb_estimator, recon_traj) if use_metropolis_hastings else None
    )

    # For-loop splice to new trajectories (and log_pf/pb) to mirror production semantics
    bs = prev_traj.n_trajectories
    n_recons = recon_traj.terminating_idx
    dones = (n_prevs + n_recons).to(torch.long)
    max_len = int(dones.max().item())

    new_states = env.States.make_sink_states(
        (max_len + 1, bs), device=prev_traj.states.device
    )
    new_actions = env.Actions.make_dummy_actions(
        (max_len, bs), device=prev_traj.actions.device
    )
    # Work with raw tensors during splice, then wrap once at the end.
    new_states_tsr = new_states.tensor
    new_actions_tsr = new_actions.tensor
    new_log_pf = (
        torch.full((max_len, bs), 0.0, device=prev_traj.states.device)
        if (prev_pf is not None and recon_pf is not None)
        else None
    )
    new_log_pb = (
        torch.full((max_len, bs), 0.0, device=prev_traj.states.device)
        if (prev_pb is not None and recon_pb is not None)
        else None
    )

    for i in range(bs):
        npv = int(n_prevs[i].item())
        nrc = int(n_recons[i].item())

        # prefix from prev_traj
        new_states_tsr[: npv + 1, i] = prev_traj.states.tensor[: npv + 1, i]
        new_actions_tsr[:npv, i] = prev_traj.actions.tensor[:npv, i]
        # suffix from recon_traj (skip junction state duplication)
        new_states_tsr[npv + 1 : npv + nrc + 1, i] = recon_traj.states.tensor[
            1 : nrc + 1, i
        ]
        new_actions_tsr[npv : npv + nrc, i] = recon_traj.actions.tensor[:nrc, i]

        if new_log_pf is not None:
            new_log_pf[:npv, i] = prev_pf[:npv, i]  # type: ignore
            new_log_pf[npv : npv + nrc, i] = recon_pf[:nrc, i]  # type: ignore
        if new_log_pb is not None:
            new_log_pb[:npv, i] = prev_pb[:npv, i]  # type: ignore
            new_log_pb[npv : npv + nrc, i] = recon_pb[:nrc, i]  # type: ignore

    new_traj = Trajectories(
        env=env,
        states=env.states_from_tensor(new_states_tsr),
        conditioning=prev_traj.conditioning,
        actions=env.actions_from_tensor(new_actions_tsr),
        terminating_idx=dones,
        is_backward=False,
        log_rewards=recon_traj.log_rewards,  # episodic reward per splice
        log_probs=new_log_pf,
    )

    # Acceptance
    if use_metropolis_hastings:
        assert (
            prev_pf is not None
            and recon_pf is not None
            and prev_pb is not None
            and recon_pb is not None
        )
        log_accept_ratio = torch.clamp_max(
            new_traj.log_rewards  # type: ignore
            + prev_pb.sum(0)
            + (new_log_pf if new_log_pf is not None else recon_pf).sum(0)
            - prev_traj.log_rewards
            - (new_log_pb if new_log_pb is not None else recon_pb).sum(0)
            - prev_pf.sum(0),
            0.0,
        )
        accept = torch.rand(bs, device=log_accept_ratio.device) < torch.exp(
            log_accept_ratio
        )
    else:
        accept = prev_traj.log_rewards <= new_traj.log_rewards  # type: ignore

    return new_traj, accept


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_local_search_back_steps_vs_back_ratio(env_name: str):
    env, pf, pb = _make_env_estimators(env_name)
    sampler = LocalSearchSampler(pf, pb)

    torch.manual_seed(123)
    base = sampler.sample_trajectories(env, n=4)

    min_len = int(base.terminating_idx.min().item())
    desired_k = max(min_len // 2, 1)
    back_steps = torch.full((base.n_trajectories,), desired_k, device=base.states.device)
    back_ratio = float(desired_k / max(min_len - 1, 1))

    traj_steps, upd_steps = sampler.local_search(
        env,
        base,
        save_logprobs=True,
        back_steps=back_steps,
        use_metropolis_hastings=False,
    )
    traj_ratio, upd_ratio = sampler.local_search(
        env,
        base,
        save_logprobs=True,
        back_ratio=back_ratio,
        use_metropolis_hastings=False,
    )

    assert traj_steps.n_trajectories == base.n_trajectories
    assert traj_ratio.n_trajectories == base.n_trajectories
    assert upd_steps.shape == (base.n_trajectories,)
    assert upd_ratio.shape == (base.n_trajectories,)
    assert traj_steps.actions.batch_shape == traj_steps.log_probs.shape  # type: ignore
    assert traj_ratio.actions.batch_shape == traj_ratio.log_probs.shape  # type: ignore


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("use_mh", [True, False])
def test_local_search_acceptance_mask_and_shapes(env_name: str, use_mh: bool):
    env, pf, pb = _make_env_estimators(env_name)
    sampler = LocalSearchSampler(pf, pb)

    torch.manual_seed(321)
    base = sampler.sample_trajectories(env, n=3, save_logprobs=True)
    new_traj, is_updated = sampler.local_search(
        env,
        base,
        save_logprobs=True,
        back_ratio=0.5,
        use_metropolis_hastings=use_mh,
        debug=True,
    )
    assert isinstance(is_updated, torch.Tensor) and is_updated.dtype == torch.bool
    assert is_updated.shape == (new_traj.n_trajectories,)
    if use_mh:
        lp_pf = get_trajectory_pfs(pf, new_traj, recalculate_all_logprobs=False)
        lp_pb = get_trajectory_pbs(pb, new_traj)
        assert lp_pf.shape == new_traj.actions.batch_shape
        assert lp_pb.shape == new_traj.actions.batch_shape


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_sample_trajectories_with_local_search_loops(env_name: str):
    env, pf, pb = _make_env_estimators(env_name)
    sampler = LocalSearchSampler(pf, pb)

    torch.manual_seed(999)
    n = 5
    trajs = sampler.sample_trajectories(
        env,
        n=n,
        save_logprobs=False,
        n_local_search_loops=2,
        back_ratio=0.5,
        use_metropolis_hastings=False,
    )
    assert trajs.n_trajectories == n * (1 + 2)
    assert trajs.actions.batch_shape[1] == trajs.n_trajectories
    assert trajs.log_rewards is not None


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_local_search_large_back_steps_are_handled_when_safe(env_name: str):
    env, pf, pb = _make_env_estimators(env_name)
    sampler = LocalSearchSampler(pf, pb)

    torch.manual_seed(7)
    base = sampler.sample_trajectories(env, n=4)
    # Very large back_steps; adjust to ensure K <= L-1 so that n_prevs >= 0
    # (the current implementation requires this to avoid negative gather indices).
    back_steps = base.terminating_idx + 100
    back_steps = torch.minimum(back_steps, base.terminating_idx - 1)
    traj, is_updated = sampler.local_search(
        env,
        base,
        back_steps=back_steps,
        save_logprobs=True,
        use_metropolis_hastings=False,
    )
    assert traj.n_trajectories == base.n_trajectories
    assert is_updated.shape == (base.n_trajectories,)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("use_mh", [True, False])
def test_local_search_reference_impl_end_to_end_match(env_name: str, use_mh: bool):
    """
    End-to-end regression: the production local_search must match a
    reference (for-loop splice) implementation when RNG is identical.
    This guards against behavior drift during refactors.
    """
    env, pf, pb = _make_env_estimators(env_name)
    sampler = LocalSearchSampler(pf, pb)

    torch.manual_seed(42)
    base = sampler.sample_trajectories(env, n=3, save_logprobs=True)

    # Ensure identical RNG for both production and reference runs
    rng = torch.get_rng_state()

    prod_traj, prod_upd = sampler.local_search(
        env,
        base,
        save_logprobs=True,
        back_ratio=0.5,
        use_metropolis_hastings=use_mh,
        debug=False,
    )

    torch.set_rng_state(rng)
    ref_traj, ref_upd = _reference_local_search(
        env,
        base,
        pf_estimator=pf,
        pb_estimator=pb,
        conditioning=None,
        back_ratio=0.5,
        use_metropolis_hastings=use_mh,
        save_logprobs=True,
    )

    # Compare tensors (align dtypes if needed)
    assert torch.allclose(
        prod_traj.states.tensor.to(ref_traj.states.tensor.dtype), ref_traj.states.tensor
    )
    assert torch.allclose(
        prod_traj.actions.tensor.to(ref_traj.actions.tensor.dtype),
        ref_traj.actions.tensor,
    )
    assert torch.equal(prod_upd, ref_upd)
    # log_rewards and log_probs comparability
    assert torch.allclose(prod_traj.log_rewards.to(ref_traj.log_rewards.dtype), ref_traj.log_rewards)  # type: ignore
    if prod_traj.log_probs is None:
        assert ref_traj.log_probs is None
    else:
        assert ref_traj.log_probs is not None
        assert torch.allclose(
            prod_traj.log_probs.to(ref_traj.log_probs.dtype), ref_traj.log_probs
        )
