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
from gfn.preprocessors import IdentityPreprocessor
from gfn.utils.modules import MLP, RecurrentDiscreteSequenceModel


def _make_bitsequence_env(
    *, device: torch.device, word_size: int = 3, seq_size: int = 9, n_modes: int = 5
) -> BitSequence:
    H = torch.randint(0, 2, (n_modes, seq_size), dtype=torch.long, device=device)
    env = BitSequence(
        word_size=word_size,
        seq_size=seq_size,
        n_modes=n_modes,
        temperature=1.0,
        H=H,
        device_str=str(device),
        seed=0,
    )
    return env


def _make_recurrent_pf(
    env: BitSequence, device: torch.device
) -> RecurrentDiscretePolicyEstimator:
    model = RecurrentDiscreteSequenceModel(
        vocab_size=env.n_actions,
        embedding_dim=16,
        hidden_size=32,
        num_layers=1,
        rnn_type="lstm",
        dropout=0.0,
    ).to(device)
    pf = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, is_backward=False
    ).to(device)
    return pf


def _make_nonrecurrent_pf_pb(env: BitSequence, device: torch.device):
    # BitSequence states are integer words of length words_per_seq
    input_dim = env.words_per_seq
    preprocessor = IdentityPreprocessor(output_dim=input_dim)

    pf_module = MLP(
        input_dim=input_dim, output_dim=env.n_actions, hidden_dim=32, n_hidden_layers=1
    ).to(device)
    pb_module = MLP(
        input_dim=input_dim,
        output_dim=env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=1,
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


def test_recurrent_tb_passes_with_pb_none():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)
    pf = _make_recurrent_pf(env, device)
    gfn = TBGFlowNet(pf=pf, pb=None, init_logZ=0.0, constant_pb=True)

    # sample and compute a loss to ensure end-to-end path works
    trajectories = gfn.sample_trajectories(
        env, n=4, save_logprobs=True, save_estimator_outputs=False
    )
    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)


def test_warn_on_recurrent_pf_with_nonrecurrent_pb():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)
    pf = _make_recurrent_pf(env, device)
    pb_pf, pb = _make_nonrecurrent_pf_pb(env, device)
    del pb_pf  # unused
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0, constant_pb=False)
        assert any("unusual" in str(x.message).lower() for x in w)


def test_error_on_recurrent_pb():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)
    pf_nonrec, _ = _make_nonrecurrent_pf_pb(env, device)

    # Build a recurrent PB
    model = RecurrentDiscreteSequenceModel(
        vocab_size=env.n_actions - 1,
        embedding_dim=16,
        hidden_size=32,
        num_layers=1,
        rnn_type="lstm",
        dropout=0.0,
    ).to(device)
    pb_recurrent = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, is_backward=True
    ).to(device)

    with pytest.raises(TypeError, match="Recurrent PB estimators are not supported"):
        _ = TBGFlowNet(pf=pf_nonrec, pb=pb_recurrent, init_logZ=0.0, constant_pb=False)


def test_db_gflownet_rejects_recurrent_pf_and_adapter():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)
    pf = _make_recurrent_pf(env, device)

    # recurrent PF should be rejected
    logF_est = ScalarEstimator(
        module=MLP(
            input_dim=env.words_per_seq,
            output_dim=1,
            hidden_dim=16,
            n_hidden_layers=1,
        ).to(device)
    )
    with pytest.raises(TypeError, match="does not support recurrent PF"):
        _ = DBGFlowNet(
            pf=pf,
            pb=None,
            logF=logF_est,
            constant_pb=True,
        )  # type: ignore[arg-type]

    # Non-recurrent PF should be accepted (adapters are now part of estimators)
    pf_nonrec, _ = _make_nonrecurrent_pf_pb(env, device)
    _ = DBGFlowNet(
        pf=pf_nonrec,
        pb=None,
        logF=logF_est,
        constant_pb=True,
    )  # type: ignore[arg-type]


def test_nonrecurrent_tb_passes_with_pb_defined():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)
    pf, pb = _make_nonrecurrent_pf_pb(env, device)
    gfn = TBGFlowNet(pf=pf, pb=pb, init_logZ=0.0, constant_pb=False)

    trajectories = gfn.sample_trajectories(
        env, n=3, save_logprobs=True, save_estimator_outputs=False
    )
    loss = gfn.loss(env, trajectories, recalculate_all_logprobs=False)
    assert torch.isfinite(loss)


def test_pb_mlp_trunk_sharing_parity_on_transitions():
    device = torch.device("cpu")
    env = _make_bitsequence_env(device=device)

    # Build non-recurrent PF for sampling
    pf, _ = _make_nonrecurrent_pf_pb(env, device)

    # PB with trunk sharing from PF
    pb_shared_module = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=1,
        trunk=pf.module.trunk,  # type: ignore[attr-defined]
    ).to(device)
    pb_shared = DiscretePolicyEstimator(
        module=pb_shared_module, n_actions=env.n_actions, is_backward=True
    ).to(device)

    # PB independent with identical weights copied from shared version
    pb_indep_module = MLP(
        input_dim=env.words_per_seq,
        output_dim=env.n_actions - 1,
        hidden_dim=32,
        n_hidden_layers=1,
    ).to(device)
    pb_indep_module.load_state_dict(pb_shared_module.state_dict())
    pb_indep = DiscretePolicyEstimator(
        module=pb_indep_module, n_actions=env.n_actions, is_backward=True
    ).to(device)

    # Sample trajectories and convert to transitions
    from gfn.samplers import Sampler

    sampler = Sampler(estimator=pf)
    trajectories = sampler.sample_trajectories(
        env, n=5, save_logprobs=False, save_estimator_outputs=False
    )
    transitions = trajectories.to_transitions()

    # Compute PB log-probs using vectorized default adapters for each PB
    from gfn.utils.prob_calculations import get_transition_pbs

    lp_shared = get_transition_pbs(pb_shared, transitions)
    lp_indep = get_transition_pbs(pb_indep, transitions)

    torch.testing.assert_close(lp_shared, lp_indep)
