"""Correctness gates for the fixed-length KV-cache sampling optimization.

Covers two layers:

- The transformer module's in-place preallocated carry (PR1a): incremental decode
  equals a single full-sequence forward, the buffers are preallocated and mutated in
  place, and the decode position is load-bearing.
- The recurrent estimator's ``use_kv_cache`` toggle (PR1b): the cached fast path and
  the corrected full-prefix baseline produce identical per-step logits, both equal to
  a teacher-forced forward, and the fixed-length guard fires on ragged batches.

These are the gates the spec calls "load-bearing": they turn red on the previous
(grow-both) recurrent path and green on the corrected implementation.
"""

from typing import Literal

import pytest
import torch

from gfn.estimators import RecurrentDiscretePolicyEstimator
from gfn.gym.bitSequence import BitSequence
from gfn.samplers import Sampler
from gfn.utils.modules import TransformerDiscreteSequenceModel

ATOL = 1e-5


def _make_model(
    vocab_size: int = 7,
    embedding_dim: int = 16,
    num_heads: int = 2,
    num_layers: int = 2,
    max_position_embeddings: int = 32,
    positional_embedding: Literal["learned", "sinusoidal"] = "learned",
) -> TransformerDiscreteSequenceModel:
    model = TransformerDiscreteSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_hidden_dim=2 * embedding_dim,
        num_layers=num_layers,
        max_position_embeddings=max_position_embeddings,
        dropout=0.0,
        positional_embedding=positional_embedding,
    )
    model.eval()
    return model


# --------------------------------------------------------------------------------
# PR1a: module-level in-place preallocated carry
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch, num_heads, num_layers, seq_len",
    [(1, 1, 1, 4), (3, 2, 2, 6), (2, 4, 3, 9)],
)
@pytest.mark.parametrize("positional_embedding", ["learned", "sinusoidal"])
def test_inplace_cache_matches_full_forward(
    batch: int,
    num_heads: int,
    num_layers: int,
    seq_len: int,
    positional_embedding: Literal["learned", "sinusoidal"],
) -> None:
    """Incremental in-place decode == a single full-sequence forward (dense gate)."""
    vocab_size = 7
    model = _make_model(
        vocab_size=vocab_size,
        embedding_dim=4 * num_heads,
        num_heads=num_heads,
        num_layers=num_layers,
        max_position_embeddings=seq_len + 2,
        positional_embedding=positional_embedding,
    )
    device = torch.device("cpu")
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        carry = model.init_carry(batch, device)
        logits_full, _ = model(tokens, carry)

        carry = model.init_carry(batch, device, max_len=seq_len)
        step_logits = []
        for t in range(seq_len):
            lo, carry = model(tokens[:, t : t + 1], carry)
            step_logits.append(lo)
        logits_inplace = torch.cat(step_logits, dim=1)

    assert torch.allclose(logits_full, logits_inplace, atol=ATOL)


def test_inplace_cache_is_preallocated_and_mutated_in_place() -> None:
    """Buffers are horizon-length from step 0, the same object across steps, and the
    cursor advances one slot per decoded token."""
    vocab_size, batch, seq_len = 7, 3, 6
    model = _make_model(vocab_size=vocab_size, max_position_embeddings=seq_len + 2)
    device = torch.device("cpu")
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        carry = model.init_carry(batch, device, max_len=seq_len)
        key0_obj = carry["key_0"]
        # Preallocated to the full horizon from the very first step.
        assert carry["key_0"].size(2) == seq_len
        assert carry["value_0"].size(2) == seq_len
        assert int(carry["cache_len"].item()) == 0

        for t in range(seq_len):
            _, carry = model(tokens[:, t : t + 1], carry)
            assert carry["key_0"] is key0_obj  # same tensor object, mutated in place
            assert carry["key_0"].size(2) == seq_len  # never reallocated / grown
            assert int(carry["cache_len"].item()) == t + 1


def test_decode_position_is_load_bearing() -> None:
    """Perturbing the decode position by +1 must break equivalence (guards abs-pos)."""
    vocab_size, batch, seq_len = 7, 2, 5
    model = _make_model(vocab_size=vocab_size, max_position_embeddings=seq_len + 2)
    device = torch.device("cpu")
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        carry_ref = model.init_carry(batch, device, max_len=seq_len)
        logits_ref, _ = model(tokens[:, 0:1], carry_ref)

        carry_bad = model.init_carry(batch, device, max_len=seq_len)
        carry_bad["position"] = carry_bad["position"] + 1  # off-by-one
        logits_bad, _ = model(tokens[:, 0:1], carry_bad)

    assert not torch.allclose(logits_ref, logits_bad, atol=ATOL)


# --------------------------------------------------------------------------------
# PR1b: estimator-level use_kv_cache toggle
# --------------------------------------------------------------------------------


def _env_and_model(word_size: int = 2, seq_size: int = 12):
    env = BitSequence(
        word_size=word_size,
        seq_size=seq_size,
        n_modes=2,
        device_str="cpu",
        seed=0,
        debug=False,
    )
    model = _make_model(
        vocab_size=env.n_actions,
        num_heads=2,
        num_layers=2,
        max_position_embeddings=env.words_per_seq + 4,
    )
    return env, model


def _per_step_logits(env, est, words: torch.Tensor) -> torch.Tensor:
    """Drive the estimator over reconstructed per-step states (0..L words committed)."""
    batch = words.shape[0]
    device = words.device
    carry = est.init_carry(batch, device)
    out = []
    with torch.no_grad():
        for t in range(env.words_per_seq + 1):
            tensor = words.clone()
            tensor[:, t:] = -1
            logits, carry = est(env.States(tensor), carry)
            out.append(logits)
    return torch.stack(out, dim=0)  # (L+1, B, n_actions)


def test_estimator_cache_equivalence() -> None:
    """use_kv_cache OFF == ON == a single teacher-forced forward, at the logit level."""
    env, model = _env_and_model()
    batch, L = 4, env.words_per_seq
    words = torch.randint(0, env.n_actions - 1, (batch, L))

    est_off = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=False
    )
    est_on = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=True
    )
    logits_off = _per_step_logits(env, est_off, words)
    logits_on = _per_step_logits(env, est_on, words)

    with torch.no_grad():
        bos = torch.full((batch, 1), env.n_actions, dtype=torch.long)
        seq = torch.cat([bos, words], dim=1)
        tf_logits, _ = model(seq, model.init_carry(batch, torch.device("cpu")))
        tf_logits = tf_logits.transpose(0, 1)  # (L+1, B, n_actions)

    assert torch.allclose(logits_off, tf_logits, atol=ATOL)
    assert torch.allclose(logits_on, tf_logits, atol=ATOL)
    assert torch.allclose(logits_off, logits_on, atol=ATOL)


def test_estimator_fixed_length_guard() -> None:
    """The cached path rejects ragged (variable-length) active batches (debug on)."""
    env, model = _env_and_model()
    # The equal-length check is gated behind debug (it forces a per-step device sync).
    est = RecurrentDiscretePolicyEstimator(
        module=model, n_actions=env.n_actions, use_kv_cache=True, debug=True
    )
    carry = est.init_carry(3, torch.device("cpu"))
    ragged = torch.randint(0, env.n_actions - 1, (3, env.words_per_seq))
    ragged[0, 1:] = -1  # row 0 has a different committed length
    with pytest.raises(ValueError, match="equal-length"):
        est(env.States(ragged), carry)


def test_cache_max_len_validated_eagerly() -> None:
    """A cache_max_len beyond the model's position range fails at construction."""
    env, model = _env_and_model()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        RecurrentDiscretePolicyEstimator(
            module=model,
            n_actions=env.n_actions,
            use_kv_cache=True,
            cache_max_len=model.max_position_embeddings + 10,
        )


def test_sampling_runs_in_both_modes() -> None:
    """Both toggle settings produce valid fixed-length trajectories end-to-end."""
    env, model = _env_and_model()
    for use_cache in (False, True):
        est = RecurrentDiscretePolicyEstimator(
            module=model, n_actions=env.n_actions, use_kv_cache=use_cache
        )
        with torch.no_grad():
            trajs = Sampler(estimator=est).sample_trajectories(
                env, n=5, save_logprobs=True, save_estimator_outputs=False
            )
        # Fixed-length: every trajectory has the same number of steps.
        assert trajs.max_length == env.words_per_seq + 1
