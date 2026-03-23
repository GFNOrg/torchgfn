"""Integration tests for recurrent/non-recurrent estimators with GFlowNets.

Tests run against both BitSequence and NonAutoregressiveBitSequence environments
to verify estimator compatibility across autoregressive and non-autoregressive modes.
"""

import warnings

import pytest
import torch

from gfn.estimators import (
    DiscretePolicyEstimator,
    RecurrentDiscretePolicyEstimator,
    ScalarEstimator,
)
from gfn.gflownet import DBGFlowNet, TBGFlowNet
from gfn.gym.bitSequence import BitSequence
from gfn.gym.bitSequenceNonAutoregressive import NonAutoregressiveBitSequence
from gfn.preprocessors import IdentityPreprocessor
from gfn.utils.modules import MLP, RecurrentDiscreteSequenceModel


@pytest.fixture(params=["BitSequence", "NonAutoregressiveBitSequence"])
def bitseq_env(request):
    """Creates a BitSequence or NonAutoregressiveBitSequence environment."""
    device = torch.device("cpu")
    word_size, seq_size, n_modes = 3, 9, 5
    H = torch.randint(0, 2, (n_modes, seq_size), dtype=torch.long, device=device)
    if request.param == "BitSequence":
        return BitSequence(
            word_size=word_size,
            seq_size=seq_size,
            n_modes=n_modes,
            temperature=1.0,
            H=H,
            device_str=str(device),
            seed=0,
        )
    else:
        return NonAutoregressiveBitSequence(
            word_size=word_size,
            seq_size=seq_size,
            n_modes=n_modes,
            reward_exponent=2.0,
            H=H,
            device_str=str(device),
            seed=0,
        )


def _make_recurrent_pf(env, device):
    model = RecurrentDiscreteSequenceModel(
        vocab_size=env.n_actions,
        embedding_dim=16,
        hidden_size=32,
        num_layers=1,
        rnn_type="lstm",
        dropout=0.0,
    ).to(device)
    return RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, is_backward=False
    ).to(device)


def _make_nonrecurrent_pf_pb(env, device):
    input_dim = env.words_per_seq
    preprocessor = IdentityPreprocessor(output_dim=input_dim)
    pf_module = MLP(
        input_dim=input_dim, output_dim=env.n_actions, hidden_dim=32, n_hidden_layers=2
    ).to(device)
    pb_module = MLP(
        input_dim=input_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=2,
    ).to(device)
    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=preprocessor,
    ).to(device)
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=preprocessor,
    ).to(device)
    return pf, pb


def test_recurrent_tb_passes_with_pb_none(bitseq_env):
    device = torch.device("cpu")
    pf = _make_recurrent_pf(bitseq_env, device)
    gfn = TBGFlowNet(pf=pf, pb=None, init_logZ=0.0, constant_pb=True)

    trajectories = gfn.sample_trajectories(
        bitseq_env, n=4, save_logprobs=True, save_estimator_outputs=False
    )
    loss = gfn.loss(bitseq_env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)


def test_warn_on_recurrent_pf_with_nonrecurrent_pb(bitseq_env):
    device = torch.device("cpu")
    pf = _make_recurrent_pf(bitseq_env, device)
    pb_pf, pb = _make_nonrecurrent_pf_pb(bitseq_env, device)
    del pb_pf
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0, constant_pb=False)
        assert any("unusual" in str(x.message).lower() for x in w)


def test_error_on_recurrent_pb(bitseq_env):
    device = torch.device("cpu")
    pf_nonrec, _ = _make_nonrecurrent_pf_pb(bitseq_env, device)

    model = RecurrentDiscreteSequenceModel(
        vocab_size=bitseq_env.n_actions - 1,
        embedding_dim=16,
        hidden_size=32,
        num_layers=1,
        rnn_type="lstm",
        dropout=0.0,
    ).to(device)
    pb_recurrent = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=bitseq_env.n_actions, is_backward=True
    ).to(device)

    with pytest.raises(TypeError, match="Recurrent PB estimators are not supported"):
        _ = TBGFlowNet(pf=pf_nonrec, pb=pb_recurrent, init_logZ=0.0, constant_pb=False)


def test_db_gflownet_rejects_recurrent_pf_and_adapter(bitseq_env):
    device = torch.device("cpu")
    pf = _make_recurrent_pf(bitseq_env, device)

    logF_est = ScalarEstimator(
        module=MLP(
            input_dim=bitseq_env.words_per_seq,
            output_dim=1,
            hidden_dim=16,
            n_hidden_layers=2,
        ).to(device)
    )
    with pytest.raises(TypeError, match="does not support recurrent PF"):
        _ = DBGFlowNet(
            pf=pf,
            pb=None,
            logF=logF_est,
            constant_pb=True,
        )  # type: ignore[arg-type]

    # Non-recurrent PF should be accepted
    pf_nonrec, _ = _make_nonrecurrent_pf_pb(bitseq_env, device)
    _ = DBGFlowNet(
        pf=pf_nonrec,
        pb=None,
        logF=logF_est,
        constant_pb=True,
    )  # type: ignore[arg-type]


def test_nonrecurrent_tb_passes_with_pb_defined(bitseq_env):
    device = torch.device("cpu")
    pf, pb = _make_nonrecurrent_pf_pb(bitseq_env, device)
    gfn = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0, constant_pb=False)

    trajectories = gfn.sample_trajectories(
        bitseq_env, n=3, save_logprobs=True, save_estimator_outputs=False
    )
    loss = gfn.loss(bitseq_env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)


def test_pb_mlp_trunk_sharing_parity_on_transitions(bitseq_env):
    device = torch.device("cpu")

    pf, _ = _make_nonrecurrent_pf_pb(bitseq_env, device)

    # PB with trunk sharing from PF
    pb_shared_module = MLP(
        input_dim=bitseq_env.words_per_seq,
        output_dim=bitseq_env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=2,
        trunk=pf.module.trunk,  # type: ignore[attr-defined]
    ).to(device)
    pb_shared = DiscretePolicyEstimator(
        module=pb_shared_module,
        n_actions=bitseq_env.n_actions,
        is_backward=True,
    ).to(device)

    # PB independent with identical weights
    pb_indep_module = MLP(
        input_dim=bitseq_env.words_per_seq,
        output_dim=bitseq_env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=2,
    ).to(device)
    pb_indep_module.load_state_dict(pb_shared_module.state_dict())
    pb_indep = DiscretePolicyEstimator(
        module=pb_indep_module,
        n_actions=bitseq_env.n_actions,
        is_backward=True,
    ).to(device)

    from gfn.samplers import Sampler

    sampler = Sampler(estimator=pf)
    trajectories = sampler.sample_trajectories(
        bitseq_env, n=5, save_logprobs=False, save_estimator_outputs=False
    )
    transitions = trajectories.to_transitions()

    from gfn.utils.prob_calculations import get_transition_pbs

    lp_shared = get_transition_pbs(pb_shared, transitions)
    lp_indep = get_transition_pbs(pb_indep, transitions)

    torch.testing.assert_close(lp_shared, lp_indep)
