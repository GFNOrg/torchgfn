"""Tests for the soft policy-gradient GFlowNets (VPG and Ent-PPO).

Covers arXiv:2606.15793: the soft-MDP reward of Eq. 7, the advantage estimators of
Eqs. 9-13, the Ent-PPO objective of Eq. 20, the Sub-EB critic of Eqs. 14-15, and the TLM
backward-policy loss of Eq. 24. The unit tests check each piece against its defining
equation rather than against a golden number, so they pin the math, not the seed.
"""

import functools

import pytest
import torch

from gfn.containers import PolicyGradientTrajectories, Trajectories
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import (
    EntPPOGFlowNet,
    PolicyGradientGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
    ppo_clip,
    tlm_loss,
)
from gfn.gym import BitSequence, HyperGrid
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP
from gfn.utils.prob_calculations import get_trajectory_pbs, get_trajectory_pfs
from gfn.utils.soft_rl import (
    gae_advantages,
    reward_to_go,
    soft_state_values,
    soft_step_returns,
    soft_step_rewards,
    soft_td_residuals,
)


# Enumerating all states and the partition function is the expensive part of building a
# HyperGrid, and the environment is read-only here, so build each one once.
@functools.lru_cache(maxsize=None)
def get_hypergrid(ndim: int, height: int) -> HyperGrid:
    """Returns a cached HyperGrid supporting exact validation."""
    return HyperGrid(
        ndim=ndim,
        height=height,
        validate_modes=False,
        store_all_states=True,
        calculate_partition=True,
    )


def make_hypergrid_setup(ndim: int = 2, height: int = 5, seed: int = 0):
    """Builds a small HyperGrid environment with PF, PB and value estimators."""
    torch.manual_seed(seed)
    env = get_hypergrid(ndim, height)
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    pf = DiscretePolicyEstimator(
        MLP(preprocessor.output_dim, env.n_actions),
        env.n_actions,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        MLP(preprocessor.output_dim, env.n_actions - 1),
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )
    logV = ScalarEstimator(MLP(preprocessor.output_dim, 1), preprocessor=preprocessor)
    return env, pf, pb, logV


def sample_rollout(env: HyperGrid, pf: DiscretePolicyEstimator, n: int = 8):
    """Samples a rollout with everything Ent-PPO needs recorded."""
    return Sampler(estimator=pf).sample_trajectories(
        env, n=n, save_logprobs=True, save_estimator_outputs=True
    )


@pytest.fixture
def setup():
    """A HyperGrid environment, estimators, and one rollout."""
    env, pf, pb, logV = make_hypergrid_setup()
    return env, pf, pb, logV, sample_rollout(env, pf)


# ----------------------------------------------------------------------
# Soft-RL quantities (Eqs. 7, 9, 10, 12, 13)
# ----------------------------------------------------------------------


def test_soft_step_rewards_match_equation_7(setup):
    """r_t is log P_B off exit steps and log R(x) on the exit step."""
    _, _, pb, _, traj = setup
    log_pb = get_trajectory_pbs(pb, traj)
    rewards = soft_step_rewards(traj, log_pb)

    is_exit = traj.actions.is_exit
    assert torch.equal(rewards[~is_exit], log_pb[~is_exit])
    # Exactly one exit action per trajectory, at index terminating_idx - 1.
    assert torch.equal(is_exit.sum(dim=0), torch.ones(traj.batch_size, dtype=torch.long))
    exit_step = traj.terminating_idx - 1
    trajectory = torch.arange(traj.batch_size)
    assert is_exit[exit_step, trajectory].all()
    assert torch.allclose(rewards[exit_step, trajectory], traj.log_rewards)


def test_padded_steps_are_neutral(setup):
    """Dummy steps contribute exactly zero to the soft return and TD residual."""
    _, pf, pb, logV, traj = setup
    dummy = traj.actions.is_dummy
    assert dummy.any(), "Test needs a batch with trajectories of differing lengths."

    with torch.no_grad():
        log_pf = get_trajectory_pfs(pf, traj)
        log_pb = get_trajectory_pbs(pb, traj)
        g = soft_step_returns(log_pf, soft_step_rewards(traj, log_pb))
        values = soft_state_values(logV, traj)
        deltas = soft_td_residuals(g, values)

    assert torch.equal(g[dummy], torch.zeros_like(g[dummy]))
    assert torch.equal(deltas[dummy], torch.zeros_like(deltas[dummy]))
    # The sink state is pinned to V = 0; terminating states are not.
    assert torch.equal(
        values[traj.states.is_sink_state],
        torch.zeros_like(values[traj.states.is_sink_state]),
    )
    assert (values[~traj.states.is_sink_state] != 0).any()


def test_total_soft_return_equals_negated_tb_score(setup):
    """The paper's identity: sum_t g_t is the negated Trajectory Balance score."""
    _, pf, pb, _, traj = setup
    tb = TBGFlowNet(pf=pf, pb=pb)
    with torch.no_grad():
        score = tb.get_scores(traj, recalculate_all_logprobs=True)
        log_pf = get_trajectory_pfs(pf, traj)
        log_pb = get_trajectory_pbs(pb, traj)
        g = soft_step_returns(log_pf, soft_step_rewards(traj, log_pb))
    assert torch.allclose(g.sum(dim=0), -score, atol=1e-5)


def test_reward_to_go_is_a_reverse_cumulative_sum():
    """R_hat_t = sum_{k >= t} g_k."""
    g = torch.arange(12, dtype=torch.get_default_dtype()).reshape(4, 3)
    rtg = reward_to_go(g)
    for t in range(4):
        assert torch.allclose(rtg[t], g[t:].sum(dim=0))


def test_gae_limits_match_their_closed_forms(setup):
    """lambda=1 recovers reward-to-go minus the baseline; lambda=0 the TD residual."""
    _, pf, pb, logV, traj = setup
    with torch.no_grad():
        log_pf = get_trajectory_pfs(pf, traj)
        log_pb = get_trajectory_pbs(pb, traj)
        g = soft_step_returns(log_pf, soft_step_rewards(traj, log_pb))
        values = soft_state_values(logV, traj)

    deltas = soft_td_residuals(g, values)
    assert torch.allclose(gae_advantages(g, values, 0.0), deltas, atol=1e-6)
    assert torch.allclose(
        gae_advantages(g, values, 1.0), reward_to_go(g) - values[:-1], atol=1e-5
    )


def test_gae_rejects_lambda_outside_unit_interval(setup):
    """lambda must be a valid bias-variance interpolation weight."""
    _, _, _, _, traj = setup
    g = torch.zeros(traj.max_length, traj.batch_size)
    values = torch.zeros(traj.max_length + 1, traj.batch_size)
    with pytest.raises(ValueError, match="lamda"):
        gae_advantages(g, values, 1.5)


# ----------------------------------------------------------------------
# PPO clipping (Eq. 5)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio, advantage, expected",
    [
        (1.0, 3.0, 3.0),  # Unclipped.
        (1.5, 3.0, 3.6),  # Positive advantage, ratio above 1+eps -> clipped.
        (0.5, 3.0, 1.5),  # Positive advantage, ratio below 1-eps -> the min picks rho*A.
        (1.5, -3.0, -4.5),  # Negative advantage: the min picks the unclipped branch.
        (0.5, -3.0, -2.4),  # Negative advantage, ratio below 1-eps -> clipped.
    ],
)
def test_ppo_clip_branches(ratio, advantage, expected):
    """PPOClip takes the pessimistic branch on both signs of the advantage."""
    got = ppo_clip(torch.tensor(ratio), torch.tensor(advantage), eps=0.2)
    assert torch.allclose(got, torch.tensor(expected), atol=1e-6)


# ----------------------------------------------------------------------
# Batch preparation and the frozen container
# ----------------------------------------------------------------------


@pytest.mark.parametrize("advantage", ["total", "reward_to_go", "baseline", "gae"])
def test_to_training_samples_freezes_the_expected_quantities(setup, advantage):
    """Advantages and targets are detached and match the selected estimator."""
    _, pf, pb, logV, traj = setup
    kwargs = {"logV": logV} if advantage in ("baseline", "gae") else {}
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, advantage=advantage, **kwargs)
    batch = gfn.to_training_samples(traj)

    assert isinstance(batch, PolicyGradientTrajectories)
    assert not batch.advantages.requires_grad
    assert not batch.value_targets.requires_grad
    assert batch.advantages.shape == traj.actions.batch_shape

    dummy = traj.actions.is_dummy
    assert torch.equal(
        batch.advantages[dummy], torch.zeros_like(batch.advantages[dummy])
    )

    with torch.no_grad():
        g = soft_step_returns(
            get_trajectory_pfs(pf, traj),
            soft_step_rewards(traj, get_trajectory_pbs(pb, traj)),
        )
    valid = ~dummy
    if advantage == "total":
        expected = g.sum(dim=0, keepdim=True).expand_as(g)
    elif advantage == "reward_to_go":
        expected = reward_to_go(g)
    else:
        with torch.no_grad():
            values = soft_state_values(logV, traj)
        expected = (
            reward_to_go(g) - values[:-1]
            if advantage == "baseline"
            else gae_advantages(g, values, gfn.gae_lambda)
        )
    assert torch.allclose(batch.advantages[valid], expected[valid], atol=1e-5)


def test_to_training_samples_is_idempotent(setup):
    """Re-preparing an already prepared batch is a no-op, not a re-freeze."""
    _, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    assert gfn.to_training_samples(batch) is batch


def test_container_getitem_extend_and_roundtrip(setup, tmp_path):
    """Slicing, extending and serializing carry the frozen tensors correctly."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)

    index = torch.tensor([0, 2, 4])
    sub = batch[index]
    assert isinstance(sub, PolicyGradientTrajectories)
    assert len(sub) == 3
    # __getitem__ truncates to the longest selected trajectory, exactly as the base
    # class does for log_probs.
    assert sub.advantages.shape == sub.actions.batch_shape
    assert torch.equal(
        sub.advantages, batch.advantages[:, index][: sub.actions.batch_shape[0]]
    )

    other = gfn.to_training_samples(sample_rollout(env, pf, n=4))
    n_before = len(sub)
    sub.extend(other)
    assert len(sub) == n_before + 4
    assert sub.advantages.shape == sub.actions.batch_shape

    path = str(tmp_path / "pg.pt")
    batch.save(path)
    loaded = PolicyGradientTrajectories.load(env, path)
    assert isinstance(loaded, PolicyGradientTrajectories)
    assert torch.allclose(loaded.advantages, batch.advantages)
    assert torch.allclose(loaded.value_targets, batch.value_targets)


def test_container_rejects_mismatched_shapes(setup):
    """The frozen tensors must align with the actions batch shape."""
    traj = setup[-1]
    with pytest.raises(ValueError, match="advantages has shape"):
        PolicyGradientTrajectories.from_trajectories(
            traj, torch.zeros(1, 1), torch.zeros(1, 1)
        )


def test_container_rejects_extending_with_plain_trajectories(setup):
    """Extending with an unprepared rollout would silently lose the frozen tensors."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    with pytest.raises(TypeError, match="PolicyGradientTrajectories"):
        batch.extend(sample_rollout(env, pf, n=2))


def test_losses_require_a_prepared_batch(setup):
    """Calling a loss on raw trajectories names the fix."""
    _, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    with pytest.raises(TypeError, match="to_training_samples"):
        gfn.policy_loss(traj)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_constructor_validation(setup):
    """The value estimator must be present exactly when the objective uses one."""
    _, pf, pb, logV, _ = setup
    with pytest.raises(ValueError, match="required"):
        PolicyGradientGFlowNet(pf=pf, pb=pb, advantage="gae")
    with pytest.raises(ValueError, match="does not use a value function"):
        PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="reward_to_go")
    with pytest.raises(ValueError, match="advantage must be one of"):
        PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ScalarEstimator"):
        PolicyGradientGFlowNet(pf=pf, pb=pb, logV=pf, advantage="gae")
    with pytest.raises(ValueError, match="kl_estimator"):
        EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, kl_estimator="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clip_eps"):
        EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, clip_eps=0.0)


# ----------------------------------------------------------------------
# Ent-PPO (Eq. 20)
# ----------------------------------------------------------------------


def test_first_inner_epoch_has_unit_ratio_and_zero_kl(setup):
    """Immediately after preparation, pi_theta IS pi_old."""
    _, pf, pb, logV, traj = setup
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, debug=True)
    batch = gfn.to_training_samples(traj)

    log_pf_new, kl = gfn._current_logprobs_and_kl(batch)
    assert kl is not None
    valid = ~batch.actions.is_dummy
    assert batch.log_probs is not None
    ratio = torch.exp(log_pf_new - batch.log_probs)
    assert torch.allclose(ratio[valid], torch.ones_like(ratio[valid]), atol=1e-6)
    assert kl[valid].abs().max() < 1e-6


def test_k1_ent_ppo_gradient_equals_vpg_gae_gradient(setup):
    """The paper states Ent-PPO with K=1 reduces to VPG(GAE); check the gradients."""
    _, pf, pb, logV, traj = setup
    ppo = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    vpg = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="gae")

    ppo.zero_grad()
    ppo.policy_loss(ppo.to_training_samples(traj)).backward()
    ppo_grads = [p.grad.clone() for p in pf.parameters()]

    vpg.zero_grad()
    vpg.policy_loss(vpg.to_training_samples(traj)).backward()
    vpg_grads = [p.grad.clone() for p in pf.parameters()]

    for a, b in zip(ppo_grads, vpg_grads):
        assert torch.isfinite(a).all()
        assert torch.allclose(a, b, atol=1e-6)


def test_ent_ppo_survives_several_inner_epochs(setup):
    """K > 1 must not error on freed graphs, and must move away from ratio 1."""
    _, pf, pb, logV, traj = setup
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    optimizer = torch.optim.Adam(pf.parameters(), lr=1e-2)

    for _ in range(4):
        optimizer.zero_grad()
        loss = gfn.policy_loss(batch)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in pf.parameters())
        optimizer.step()

    # The frozen quantities are unchanged by the inner epochs.
    log_pf_new, kl = gfn._current_logprobs_and_kl(batch)
    assert kl is not None
    valid = ~batch.actions.is_dummy
    assert kl[valid].max() > 0.0
    assert batch.log_probs is not None
    assert (torch.exp(log_pf_new - batch.log_probs)[valid] - 1).abs().max() > 0.0


def test_ablation_switches_drop_exactly_their_term(setup):
    """use_kl / use_clipping remove the KL and the clip, and nothing else."""
    _, pf, pb, logV, traj = setup
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    # Perturb the policy so the ratio leaves the clip window and the KL is positive.
    with torch.no_grad():
        for p in pf.parameters():
            p.add_(torch.randn_like(p) * 0.2)

    valid = ~batch.actions.is_dummy
    log_pf_new, kl = gfn._current_logprobs_and_kl(batch)
    assert kl is not None
    assert batch.log_probs is not None
    ratio = torch.exp(log_pf_new - batch.log_probs)
    advantages = batch.advantages

    full = gfn.policy_loss(batch)
    expected_full = (
        -((ppo_clip(ratio, advantages, gfn.clip_eps) - kl) * valid).sum(dim=0).mean()
    )
    assert torch.allclose(full, expected_full, atol=1e-5)

    no_kl = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, use_kl=False)
    expected_no_kl = (
        -(ppo_clip(ratio, advantages, gfn.clip_eps) * valid).sum(dim=0).mean()
    )
    assert torch.allclose(no_kl.policy_loss(batch), expected_no_kl, atol=1e-5)

    no_clip = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, use_clipping=False)
    expected_no_clip = -((ratio * advantages - kl) * valid).sum(dim=0).mean()
    assert torch.allclose(no_clip.policy_loss(batch), expected_no_clip, atol=1e-5)

    # Clipping is only a no-op while the ratio is inside the trust region.
    assert not torch.allclose(full, no_clip.policy_loss(batch), atol=1e-5)


def test_importance_kl_estimator_is_rho_log_rho(setup):
    """The generic fallback is the single-sample estimator E_old[rho * log rho]."""
    _, pf, pb, logV, traj = setup
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, kl_estimator="importance")
    batch = gfn.to_training_samples(traj)
    # Perturb the policy so pi_theta and pi_old actually differ.
    with torch.no_grad():
        for p in pf.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    log_pf_new, kl = gfn._current_logprobs_and_kl(batch)
    assert kl is not None
    assert batch.log_probs is not None
    log_ratio = log_pf_new - batch.log_probs
    assert torch.allclose(kl, torch.exp(log_ratio) * log_ratio, atol=1e-6)
    # It needs no estimator outputs, which is the point of offering it.
    batch.estimator_outputs = None
    assert torch.isfinite(gfn.policy_loss(batch))


def test_both_kl_estimators_vanish_at_the_rollout_policy(setup):
    """Before any inner step, pi_theta == pi_old, so every KL estimator is zero."""
    _, pf, pb, logV, traj = setup
    for estimator in ("analytic", "importance"):
        gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, kl_estimator=estimator)
        batch = gfn.to_training_samples(traj)
        _, kl = gfn._current_logprobs_and_kl(batch)
        assert kl is not None
        assert kl.abs().max() < 1e-6


def test_analytic_kl_requires_estimator_outputs(setup):
    """The analytic KL needs the rollout logits, and says so."""
    env, pf, pb, logV, _ = setup
    traj = Sampler(estimator=pf).sample_trajectories(
        env, n=4, save_logprobs=True, save_estimator_outputs=False
    )
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    with pytest.raises(ValueError, match="save_estimator_outputs"):
        gfn.policy_loss(gfn.to_training_samples(traj))


# ----------------------------------------------------------------------
# Critic and backward policy
# ----------------------------------------------------------------------


def test_value_loss_regresses_onto_the_frozen_targets(setup):
    """The MSE critic loss is the squared residual against y_t on valid steps."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="gae")
    batch = gfn.to_training_samples(traj)
    valid = ~batch.actions.is_dummy
    values = soft_state_values(logV, batch)
    expected = ((values[:-1] - batch.value_targets)[valid] ** 2).mean()
    assert torch.allclose(gfn.value_loss(env, batch), expected, atol=1e-6)


def test_gae_value_target_is_the_bootstrapped_form(setup):
    """y_t = sg[A_t + V(s_t)] under GAE, per Algorithm 2 line 4."""
    _, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="gae")
    batch = gfn.to_training_samples(traj)
    with torch.no_grad():
        values = soft_state_values(logV, traj)
    valid = ~batch.actions.is_dummy
    expected = (batch.advantages + values[:-1])[valid]
    assert torch.allclose(batch.value_targets[valid], expected, atol=1e-5)


def test_gradients_reach_the_right_parameters(setup):
    """The policy loss trains pf only; the critic loss trains logV only."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="gae")
    batch = gfn.to_training_samples(traj)

    gfn.zero_grad()
    gfn.policy_loss(batch).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pf.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in logV.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pb.parameters())

    gfn.zero_grad()
    gfn.value_loss(env, batch).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in logV.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pf.parameters())


def test_sub_eb_critic_trains_only_the_value_function(setup):
    """The Sub-EB objective is evaluated with the policies frozen (Eqs. 14-15)."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(
        pf=pf, pb=pb, logV=logV, advantage="gae", critic_loss="sub_eb"
    )
    batch = gfn.to_training_samples(traj)
    loss = gfn.value_loss(env, batch)
    assert torch.isfinite(loss)

    gfn.zero_grad()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in logV.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pf.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pb.parameters())

    # Sharing pf/pb/logV with the internal SubTB must not duplicate parameters.
    ids = [id(p) for p in gfn.parameters()]
    assert len(ids) == len(set(ids))


def test_sub_eb_requires_a_value_function(setup):
    """A balance-based critic objective with no critic is a configuration error."""
    _, pf, pb, _, _ = setup
    with pytest.raises(ValueError, match="does not use"):
        PolicyGradientGFlowNet(
            pf=pf, pb=pb, advantage="reward_to_go", critic_loss="sub_eb"
        )


def test_subtb_logprob_injection_must_be_paired(setup):
    """SubTB's new injection kwargs are all-or-nothing."""
    env, pf, pb, logV, traj = setup
    subtb = SubTBGFlowNet(pf=pf, pb=pb, logF=logV)
    log_pf = get_trajectory_pfs(pf, traj)
    with pytest.raises(ValueError, match="supplied together"):
        subtb.loss(env, traj, log_pf_trajectories=log_pf)
    with pytest.raises(ValueError, match="expected"):
        subtb.loss(
            env,
            traj,
            log_pf_trajectories=log_pf[:1],
            log_pb_trajectories=log_pf[:1],
        )


def test_subtb_injection_reproduces_the_uninjected_loss(setup):
    """Passing the same logprobs SubTB would have computed changes nothing."""
    env, pf, pb, logV, traj = setup
    subtb = SubTBGFlowNet(pf=pf, pb=pb, logF=logV)
    baseline = subtb.loss(env, traj)
    with torch.no_grad():
        log_pf = get_trajectory_pfs(pf, traj)
        log_pb = get_trajectory_pbs(pb, traj)
    injected = subtb.loss(
        env, traj, log_pf_trajectories=log_pf, log_pb_trajectories=log_pb
    )
    assert torch.allclose(baseline, injected, atol=1e-6)


def test_tlm_loss_trains_only_the_backward_policy(setup):
    """TLM (Eq. 24) fits P_B to the trajectory distribution induced by P_F."""
    _, pf, pb, _, traj = setup
    loss = tlm_loss(pb, traj)
    assert torch.isfinite(loss)
    with torch.no_grad():
        expected = -get_trajectory_pbs(pb, traj).sum(dim=0).mean()
    assert torch.allclose(loss, expected, atol=1e-6)

    pf.zero_grad()
    pb.zero_grad()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pb.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pf.parameters())


def test_tlm_reduces_the_kl_it_estimates(setup):
    """Repeated TLM updates increase the likelihood of P_F's trajectories under P_B."""
    env, pf, pb, _, _ = setup
    traj = sample_rollout(env, pf, n=64)
    optimizer = torch.optim.Adam(pb.parameters(), lr=1e-2)
    before = tlm_loss(pb, traj).item()
    for _ in range(20):
        optimizer.zero_grad()
        tlm_loss(pb, traj).backward()
        optimizer.step()
    assert tlm_loss(pb, traj).item() < before


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------


def train(
    gfn,
    env,
    n_iterations: int,
    batch_size: int = 32,
    lr: float = 1e-2,
    K: int = 1,
    seed: int = 0,
) -> float:
    """Runs a short training loop and returns the final L1 distance to the target."""
    torch.manual_seed(seed)
    sampler = Sampler(estimator=gfn.pf)
    policy_optimizer = torch.optim.Adam(gfn.pf.parameters(), lr=lr)
    value_optimizer = (
        torch.optim.Adam(gfn.logV_parameters(), lr=lr / 3)
        if gfn.logV is not None
        else None
    )
    for _ in range(n_iterations):
        traj = sampler.sample_trajectories(
            env, n=batch_size, save_logprobs=True, save_estimator_outputs=True
        )
        batch = gfn.to_training_samples(traj)
        for _ in range(K):
            policy_optimizer.zero_grad()
            gfn.policy_loss(batch).backward()
            torch.nn.utils.clip_grad_norm_(gfn.pf.parameters(), 1.0)
            policy_optimizer.step()
        if value_optimizer is not None:
            for _ in range(2):
                value_optimizer.zero_grad()
                gfn.value_loss(env, batch).backward()
                value_optimizer.step()
    return env.validate(gfn, 10000, check_sample_sufficiency=False)[0]["l1_dist"]


@pytest.mark.parametrize(
    "method, kwargs, K",
    [
        ("vpg", {"advantage": "gae"}, 1),
        ("ent_ppo", {}, 4),
    ],
)
def test_converges_on_hypergrid(method, kwargs, K):
    """Both objectives learn the HyperGrid terminating distribution."""
    env, pf, pb, logV = make_hypergrid_setup(ndim=2, height=8, seed=1)
    gfn = (
        EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, **kwargs)
        if method == "ent_ppo"
        else PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, **kwargs)
    )
    untrained = env.validate(gfn, 10000, check_sample_sufficiency=False)[0]["l1_dist"]
    trained = train(gfn, env, n_iterations=300, K=K)
    assert trained < untrained
    assert trained < 0.5


def test_ent_ppo_runs_on_a_sequence_environment():
    """The paper's other small-problem family: autoregressive sequence generation."""
    torch.manual_seed(0)
    env = BitSequence(
        word_size=2,
        seq_size=6,
        n_modes=3,
        H=torch.randint(0, 2, (3, 6), dtype=torch.long),
        device_str="cpu",
        seed=0,
    )
    state_dim = int(env.s0.shape[-1])
    preprocessor = IdentityPreprocessor(output_dim=state_dim)
    pf = DiscretePolicyEstimator(
        MLP(state_dim, env.n_actions),
        env.n_actions,
        preprocessor=preprocessor,
    )
    pb = DiscretePolicyEstimator(
        MLP(state_dim, env.n_actions - 1),
        env.n_actions,
        preprocessor=preprocessor,
        is_backward=True,
    )
    logV = ScalarEstimator(MLP(state_dim, 1), preprocessor=preprocessor)
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)

    sampler = Sampler(estimator=pf)
    optimizer = torch.optim.Adam(pf.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        traj = sampler.sample_trajectories(
            env, n=16, save_logprobs=True, save_estimator_outputs=True
        )
        batch = gfn.to_training_samples(traj)
        for _ in range(4):
            optimizer.zero_grad()
            loss = gfn.policy_loss(batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))


def test_plain_trajectories_still_work_through_loss_from_trajectories(setup):
    """The generic GFlowNet entry point prepares the batch itself."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    assert isinstance(traj, Trajectories)
    loss = gfn.loss_from_trajectories(env, traj)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pf.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in logV.parameters())


# ----------------------------------------------------------------------
# Regressions
# ----------------------------------------------------------------------


def test_gae_handles_a_zero_step_batch():
    """An empty selection has no row 0 for the reverse recursion to seed from."""
    advantages = gae_advantages(torch.zeros(0, 4), torch.zeros(1, 4), 0.7)
    assert advantages.shape == (0, 4)


@pytest.mark.parametrize("advantage", ["total", "reward_to_go", "baseline", "gae"])
def test_empty_selection_prepares_without_error(setup, advantage):
    """Mini-batch splitters and replay paths can hand over an empty selection."""
    _, pf, pb, logV, traj = setup
    kwargs = {"logV": logV} if advantage in ("baseline", "gae") else {}
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, advantage=advantage, **kwargs)
    empty = traj[torch.tensor([], dtype=torch.long)]
    batch = gfn.to_training_samples(empty)
    assert len(batch) == 0
    assert batch.advantages.shape == batch.actions.batch_shape


def test_combined_loss_rejects_unreduced_output(setup):
    """The policy term is per-trajectory and the critic term per-step; they don't add."""
    env, pf, pb, logV, traj = setup
    gfn = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV)
    with pytest.raises(ValueError, match="reduction='none' is not supported"):
        gfn.loss(env, traj, reduction="none")
    # Without a critic there is only one term, so 'none' is meaningful.
    vpg = PolicyGradientGFlowNet(pf=pf, pb=pb, advantage="reward_to_go")
    assert vpg.loss(env, traj, reduction="none").shape == (traj.batch_size,)


def test_prepared_batch_holds_no_autograd_graph(setup):
    """The sampler records estimator outputs with a graph; the batch must not keep it."""
    import copy

    _, pf, pb, logV, traj = setup
    assert traj.estimator_outputs is not None
    assert traj.estimator_outputs.requires_grad, "Test needs a graph to strip."

    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    assert batch.estimator_outputs is not None
    assert not batch.estimator_outputs.requires_grad
    assert batch.log_probs is not None
    assert not batch.log_probs.requires_grad
    copy.deepcopy(batch)  # Would raise if any tensor were still a non-leaf.


def test_sub_eb_critic_uses_the_frozen_rollout_policy(setup):
    """Eq. 15 holds theta fixed, so drifting the policy must not move the critic loss."""
    env = setup[0]
    losses = {}
    for critic in ("mse", "sub_eb"):
        torch.manual_seed(0)
        pf_i, pb_i, logV_i = make_hypergrid_setup()[1:]
        gfn = PolicyGradientGFlowNet(
            pf=pf_i, pb=pb_i, logV=logV_i, advantage="gae", critic_loss=critic
        )
        batch = gfn.to_training_samples(sample_rollout(env, pf_i))
        before = gfn.value_loss(env, batch).item()
        with torch.no_grad():
            for p in pf_i.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        losses[critic] = (before, gfn.value_loss(env, batch).item())

    for critic, (before, after) in losses.items():
        assert before == pytest.approx(
            after, rel=1e-6
        ), f"the {critic} critic loss moved when only the policy changed"


def test_sub_eb_does_not_duplicate_the_state_dict(setup):
    """The internal SubTB helper borrows pf/pb/logV and must not re-register them."""
    _, pf, pb, logV, _ = setup
    mse = PolicyGradientGFlowNet(pf=pf, pb=pb, logV=logV, advantage="gae")
    sub_eb = PolicyGradientGFlowNet(
        pf=pf, pb=pb, logV=logV, advantage="gae", critic_loss="sub_eb"
    )
    assert set(sub_eb.state_dict()) == set(mse.state_dict())
    mse.load_state_dict(sub_eb.state_dict())  # Would raise on unexpected keys.
    assert not any("_sub_eb" in k for k in sub_eb.state_dict())


def test_extreme_log_ratio_gives_finite_gradients(setup):
    """exp() of a large log-ratio overflows to inf, whose backward is NaN."""
    _, pf, pb, logV, traj = setup
    gfn = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV)
    batch = gfn.to_training_samples(traj)
    assert batch.log_probs is not None
    # Pretend pi_old assigned these actions a vanishing probability.
    batch.log_probs = batch.log_probs - 500.0

    for estimator in ("analytic", "importance"):
        gfn_i = EntPPOGFlowNet(pf=pf, pb=pb, logV=logV, kl_estimator=estimator)
        gfn_i.zero_grad()
        loss = gfn_i.policy_loss(batch)
        assert torch.isfinite(loss), estimator
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in pf.parameters()), estimator


def test_tlm_loss_is_importable_from_the_gflownet_package():
    """tlm_loss is an objective, so it lives with the other objectives."""
    from gfn.gflownet.policy_gradient import tlm_loss as direct

    assert tlm_loss is direct
