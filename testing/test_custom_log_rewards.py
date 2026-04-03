"""Tests that the optional `log_rewards` parameter correctly overrides
environment rewards in the TB/DB/SubTB/FM/LPV/RTB GFlowNet loss implementations
covered by this file.

For each loss variant we verify two properties:

1. **Identity**: passing `log_rewards=trajectories.log_rewards` (or the
   equivalent for transition/state-based losses) reproduces the default loss
   exactly.
2. **Override**: passing a *different* `log_rewards` tensor produces a
   different loss value, proving that the custom rewards are actually used.
"""

import torch

from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.detailed_balance import DBGFlowNet
from gfn.gflownet.flow_matching import FMGFlowNet
from gfn.gflownet.sub_trajectory_balance import SubTBGFlowNet
from gfn.gflownet.trajectory_balance import (
    LogPartitionVarianceGFlowNet,
    RelativeTrajectoryBalanceGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_hypergrid_estimators():
    """Build a HyperGrid env with PF, PB, and logF estimators."""
    torch.manual_seed(0)
    env = HyperGrid(ndim=2, height=4, validate_modes=False)
    preproc = KHotPreprocessor(env.height, env.ndim)
    assert isinstance(preproc.output_dim, int)

    pf_module = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions)
    pb_module = MLP(input_dim=preproc.output_dim, output_dim=env.n_actions - 1)
    logF_module = MLP(input_dim=preproc.output_dim, output_dim=1)

    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=False,
    )
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        preprocessor=preproc,
        is_backward=True,
    )
    logF = ScalarEstimator(module=logF_module, preprocessor=preproc)
    return env, pf, pb, logF


def _sample_trajectories(env, pf, n=8):
    """Sample trajectories from the environment."""
    sampler = Sampler(estimator=pf)
    return sampler.sample_trajectories(env, n=n, save_logprobs=True)


# ---------------------------------------------------------------------------
# TB
# ---------------------------------------------------------------------------


class TestTBCustomLogRewards:
    def test_identity(self):
        """Passing the trajectory's own log_rewards reproduces the default loss."""
        env, pf, pb, _ = _make_hypergrid_estimators()
        gfn = TBGFlowNet(pf=pf, pb=pb)
        trajs = _sample_trajectories(env, pf)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            explicit_loss = gfn.loss(env, trajs, log_rewards=trajs.log_rewards)

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        env, pf, pb, _ = _make_hypergrid_estimators()
        gfn = TBGFlowNet(pf=pf, pb=pb)
        trajs = _sample_trajectories(env, pf)

        custom_rewards = torch.zeros(trajs.batch_size)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            custom_loss = gfn.loss(env, trajs, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


class TestDBCustomLogRewards:
    def test_identity(self):
        """Passing the transition's own log_rewards reproduces the default loss."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = DBGFlowNet(pf=pf, pb=pb, logF=logF)
        trajs = _sample_trajectories(env, pf)
        transitions = trajs.to_transitions()

        with torch.no_grad():
            default_loss = gfn.loss(env, transitions)
            explicit_loss = gfn.loss(
                env, transitions, log_rewards=transitions.log_rewards
            )

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = DBGFlowNet(pf=pf, pb=pb, logF=logF)
        trajs = _sample_trajectories(env, pf)
        transitions = trajs.to_transitions()

        custom_rewards = torch.zeros(transitions.n_transitions)

        with torch.no_grad():
            default_loss = gfn.loss(env, transitions)
            custom_loss = gfn.loss(env, transitions, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"


# ---------------------------------------------------------------------------
# SubTB
# ---------------------------------------------------------------------------


class TestSubTBCustomLogRewards:
    def test_identity(self):
        """Passing the trajectory's own log_rewards reproduces the default loss."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting="geometric_within")
        trajs = _sample_trajectories(env, pf)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            explicit_loss = gfn.loss(env, trajs, log_rewards=trajs.log_rewards)

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting="geometric_within")
        trajs = _sample_trajectories(env, pf)

        custom_rewards = torch.zeros(trajs.batch_size)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            custom_loss = gfn.loss(env, trajs, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"

    def test_identity_tb_weighting(self):
        """TB weighting path also respects custom log_rewards identity."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting="TB")
        trajs = _sample_trajectories(env, pf)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            explicit_loss = gfn.loss(env, trajs, log_rewards=trajs.log_rewards)

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity (TB weighting) failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override_tb_weighting(self):
        """TB weighting path also uses custom log_rewards when provided."""
        env, pf, pb, logF = _make_hypergrid_estimators()
        gfn = SubTBGFlowNet(pf=pf, pb=pb, logF=logF, weighting="TB")
        trajs = _sample_trajectories(env, pf)

        custom_rewards = torch.zeros(trajs.batch_size)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            custom_loss = gfn.loss(env, trajs, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override (TB weighting) failed: custom rewards produced the same loss"


# ---------------------------------------------------------------------------
# FM
# ---------------------------------------------------------------------------


class TestFMCustomLogRewards:
    def test_identity(self):
        """Passing the container's own terminating_log_rewards reproduces the
        default loss."""
        torch.manual_seed(0)
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        preproc = KHotPreprocessor(env.height, env.ndim)
        module = MLP(
            input_dim=preproc.output_dim, output_dim=env.n_actions  # type: ignore
        )
        logF = DiscretePolicyEstimator(
            module=module, n_actions=env.n_actions, preprocessor=preproc
        )
        gfn = FMGFlowNet(logF)

        trajs = gfn.sample_trajectories(env, n=8)
        states_container = gfn.to_training_samples(trajs)

        with torch.no_grad():
            default_loss = gfn.loss(env, states_container)
            explicit_loss = gfn.loss(
                env,
                states_container,
                log_rewards=states_container.terminating_log_rewards,
            )

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        torch.manual_seed(0)
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
        preproc = KHotPreprocessor(env.height, env.ndim)
        module = MLP(
            input_dim=preproc.output_dim, output_dim=env.n_actions  # type: ignore
        )
        logF = DiscretePolicyEstimator(
            module=module, n_actions=env.n_actions, preprocessor=preproc
        )
        gfn = FMGFlowNet(logF)

        trajs = gfn.sample_trajectories(env, n=8)
        states_container = gfn.to_training_samples(trajs)

        n_terminating = len(states_container.terminating_states)
        custom_rewards = torch.zeros(n_terminating)

        with torch.no_grad():
            default_loss = gfn.loss(env, states_container)
            custom_loss = gfn.loss(env, states_container, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"


# ---------------------------------------------------------------------------
# LPV
# ---------------------------------------------------------------------------


class TestLPVCustomLogRewards:
    def test_identity(self):
        """Passing the trajectory's own log_rewards reproduces the default loss."""
        env, pf, pb, _ = _make_hypergrid_estimators()
        gfn = LogPartitionVarianceGFlowNet(pf=pf, pb=pb)
        trajs = _sample_trajectories(env, pf)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            explicit_loss = gfn.loss(env, trajs, log_rewards=trajs.log_rewards)

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        env, pf, pb, _ = _make_hypergrid_estimators()
        gfn = LogPartitionVarianceGFlowNet(pf=pf, pb=pb)
        trajs = _sample_trajectories(env, pf)

        custom_rewards = torch.zeros(trajs.batch_size)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            custom_loss = gfn.loss(env, trajs, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"


# ---------------------------------------------------------------------------
# RTB
# ---------------------------------------------------------------------------


class TestRTBCustomLogRewards:
    def _make_rtb(self):
        env, pf, _, _ = _make_hypergrid_estimators()
        preproc = KHotPreprocessor(env.height, env.ndim)
        prior_module = MLP(
            input_dim=preproc.output_dim, output_dim=env.n_actions  # type: ignore
        )
        prior_pf = DiscretePolicyEstimator(
            module=prior_module,
            n_actions=env.n_actions,
            preprocessor=preproc,
            is_backward=False,
        )
        for p in prior_pf.parameters():
            p.requires_grad_(False)

        gfn = RelativeTrajectoryBalanceGFlowNet(pf=pf, prior_pf=prior_pf, beta=1.0)
        trajs = _sample_trajectories(env, pf)
        return env, gfn, trajs

    def test_identity(self):
        """Passing the trajectory's own log_rewards reproduces the default loss."""
        env, gfn, trajs = self._make_rtb()

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            explicit_loss = gfn.loss(env, trajs, log_rewards=trajs.log_rewards)

        assert torch.allclose(default_loss, explicit_loss), (
            f"Identity failed: default={default_loss.item()}, "
            f"explicit={explicit_loss.item()}"
        )

    def test_override(self):
        """Passing different log_rewards produces a different loss."""
        env, gfn, trajs = self._make_rtb()

        custom_rewards = torch.zeros(trajs.batch_size)

        with torch.no_grad():
            default_loss = gfn.loss(env, trajs)
            custom_loss = gfn.loss(env, trajs, log_rewards=custom_rewards)

        assert not torch.allclose(
            default_loss, custom_loss
        ), "Override failed: custom rewards produced the same loss as default"
