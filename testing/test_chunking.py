from __future__ import annotations

import random
from typing import List, Sequence, cast

import torch
import torch.nn as nn

from gfn.chunking.policy import ActionEncoder, ChunkedPolicy
from gfn.containers import Trajectories
from gfn.env import ChunkedDiscreteEnvironment
from gfn.states import ChunkedStates, DiscreteStates

# from gfn.chunking.adapters import ChunkedAdapter


class SyntheticTokenEnv(ChunkedDiscreteEnvironment):
    def __init__(self, device: torch.device = torch.device("cpu")):
        # n_actions = 27: A=1,B=2,C=3,D=4,...,Z=26,EXIT=27
        n_actions = 27
        s0 = torch.tensor([0], device=device)
        # Tokenizer maps primitive ints to letters for string-based chunkers.

        def _letters_tokenizer(seq: Sequence[int]) -> str:
            alpha = {i: chr(ord("A") + i - 1) for i in range(1, 27)}  # 1->A ... 26->Z
            alpha[27] = "EXIT"

            return "".join(alpha.get(i, "") for i in seq)

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=(1,),
            action_shape=(1,),
            check_action_validity=True,
            tokenizer=_letters_tokenizer,
            exit_action=torch.tensor([27], device=device),
        )

    def update_masks(self, states: ChunkedStates) -> None:
        # Forward mask: disallow [B,A] and [C,D]
        # Backward mask: inverse constraints: if curr==A, disallow parent B; if curr==D, disallow parent C
        device = states.device
        batch_shape = states.batch_shape

        # Initialize all true for both masks with correct batch shape
        fwd = torch.ones((*batch_shape, self.n_actions), dtype=torch.bool, device=device)
        bwd = torch.ones(
            (*batch_shape, self.n_actions - 1), dtype=torch.bool, device=device
        )

        # Current token value per batch element (handles (B,1) and (T,B,1))
        last = states.tensor.squeeze(-1)  # shape == (*batch_shape,)

        # Forward constraints
        disallow_A = last == 2  # if last == B --> disallow A
        if disallow_A.any():
            # Mask primitive A (index 0)
            fwd[..., 1].masked_fill_(disallow_A, False)
        disallow_D = last == 3  # if last == C --> disallow D
        if disallow_D.any():
            # Mask primitive D (index 3)
            fwd[..., 4].masked_fill_(disallow_D, False)

        # Backward constraints (no EXIT column)
        disallow_parent_B = last == 1  # current == A -> disallow parent B
        if disallow_parent_B.any():
            bwd[..., 2].masked_fill_(disallow_parent_B, False)

        disallow_parent_C = last == 4  # current == D -> disallow parent C
        if disallow_parent_C.any():
            bwd[..., 3].masked_fill_(disallow_parent_C, False)

        states.forward_masks = fwd
        states.backward_masks = bwd

        # Overlay global disables and macro feasibility
        self.apply_soft_disabled_to_forward_masks(states)
        self.apply_macro_forward_mask(states)

    def step(self, states: DiscreteStates, actions) -> DiscreteStates:
        # Set state to the action token, unless EXIT; EXIT leads to sink (-1)
        device = states.device
        a = actions.tensor  # preserve shape (matches states.tensor)
        new = states.tensor.clone()
        # For any EXIT, set sink
        is_exit = a == (self.n_actions - 1)
        new[is_exit] = torch.tensor([-1], device=device, dtype=new.dtype)
        # For non-exit, set to the chosen primitive token id
        non_exit = ~is_exit
        if a.dtype != new.dtype:
            a = a.to(dtype=new.dtype)
        new[non_exit] = a[non_exit]
        out = self.states_from_tensor(new)
        return out

    def backward_step(self, states: DiscreteStates, actions) -> DiscreteStates:
        # For synthetic tests, just revert to s0 for simplicity when not exit
        device = states.device
        a = actions.tensor.view(-1)
        new = states.tensor.clone()
        # Non-exit moves to a dummy parent; for tests we can use 0
        new[a != (self.n_actions - 1)] = torch.tensor(
            [0], device=device, dtype=new.dtype
        )
        out = self.states_from_tensor(new)
        return out


def generate_synthetic_corpus(
    n_traj: int = 10000,
    length: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Trajectories:
    # Build N trajectories respecting forward constraints and injecting chunks
    # Tokens 1..4 only; no EXIT in the corpus
    env = SyntheticTokenEnv(device)

    actions_2d = torch.zeros(length, n_traj, dtype=torch.long, device=device)
    term = torch.full((n_traj,), length, dtype=torch.long, device=device)

    subseq = [2, 3, 1, 4, 2]  # "BCADB"
    for i in range(n_traj):
        seq: List[int] = []

        # Choose an insertion index for BCADB that fits in the sequence
        ins_start = random.randint(0, length - len(subseq))

        t = 0
        while t < length:

            # Add the subsequence here.
            if t == ins_start:
                seq.extend(subseq)
                t += len(subseq)
                continue

            last = seq[-1] if seq else 0
            candidates = list(range(1, 27))

            # Forward constraints from SyntheticTokenEnv.update_masks
            if last == 2:  # after B, disallow A
                candidates.remove(1)
            if last == 3:  # after C, disallow D
                candidates.remove(4)
            seq.append(random.choice(candidates))
            t += 1

        actions_2d[:, i] = torch.tensor(seq[:length], dtype=torch.long, device=device)

    # Wrap into env-specific containers
    actions = env.actions_from_tensor(actions_2d.unsqueeze(-1))

    # Derive states by unrolling tokens as last-observation states
    states_tensor = torch.zeros(length + 1, n_traj, 1, dtype=torch.long, device=device)
    states_tensor[0, :, 0] = 0  # s0
    states_tensor[1:, :, 0] = actions_2d
    states = cast(ChunkedStates, env.states_from_tensor(states_tensor))

    return Trajectories(
        env=env,
        states=states,
        actions=actions,
        terminating_idx=term,
        is_backward=False,
    )


class TinyStateModule(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(
            6, embed_dim
        )  # allow -1,0..4 remapped in preprocessing
        self.proj = nn.Linear(embed_dim, 32)

    def forward(self, states: DiscreteStates) -> torch.Tensor:
        x = states.tensor.view(-1)
        # Remap -1->0, keep 0..4 as is plus 1 offset to avoid negative indices
        x = torch.clamp(x + 1, min=0)
        e = self.embed(x)
        out = self.proj(e)
        return out


class _ConstState(nn.Module):
    """Return a constant state embedding for any input batch.

    This isolates the test to the interaction between `ChunkedPolicy` and
    `ActionEncoder`, avoiding any dependency on a learned state network.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)

    def forward(self, states: DiscreteStates) -> torch.Tensor:
        batch = states.batch_shape[0]
        return self.embedding.expand(batch, -1)


def test_policy_encoder_growing_action_space_with_synthetic_env():
    # Ensure deterministic encoder behavior
    torch.manual_seed(0)

    device = torch.device("cpu")
    D = 32

    # Reuse the synthetic environment with primitives A..Z and EXIT
    env = SyntheticTokenEnv(device)

    # Build a real batch of discrete states (values don't matter for constant state net)
    states = env.states_from_tensor(torch.tensor([[0], [1], [2]], device=device))

    # Encoder maps sequences of primitive ids to action embeddings in R^D
    encoder = ActionEncoder(
        n_primitive_actions=env.n_actions,  # primitives + EXIT
        action_embedding_dimension=D,
        hidden_dim=32,
        num_layers=1,
        num_head=4,
        max_len=8,
        dropout=0.0,
    )

    # Policy produces logits via scaled dot product between state and action embeddings
    policy = ChunkedPolicy(_ConstState(D), encoder, env, action_embedding_dim=D)

    # First pass: primitives only
    logits1 = policy.forward_logits(states)
    assert logits1.shape == (states.batch_shape[0], env.n_actions)
    emb1 = policy._library_embeddings.detach().clone()
    assert torch.isfinite(emb1).all()

    # Grow action space by adding a length-5 macro (BCADB)
    env.add_tokens([(2, 3, 1, 4, 2)])
    logits2 = policy.forward_logits(states)
    assert logits2.shape == (states.batch_shape[0], env.n_actions)
    emb2 = policy._library_embeddings.detach().clone()

    # Existing embeddings should be unchanged after refresh
    assert torch.allclose(emb2[: emb1.shape[0]], emb1, atol=1e-1)

    # Grow again with a different-length macro
    env.add_tokens([(3, 3, 3)])
    logits3 = policy.forward_logits(states)
    assert logits3.shape == (states.batch_shape[0], env.n_actions)
    emb3 = policy._library_embeddings.detach()
    assert emb3.shape[0] == env.n_actions
    assert torch.isfinite(emb3).all()

    # Check scaled dot-product formula: l_t = f_θ(A) q_t / sqrt(d)
    state_emb = policy.state_module(states)
    expected = torch.einsum("bd,nd->bn", state_emb, emb3) / (D**0.5)
    assert torch.allclose(logits3, expected, atol=1e-6)


def test_mining_finds_chunks():
    from gfn.chunking.chunkers import BPEChunker, WordPieceChunker

    device = torch.device("cpu")
    trajs = generate_synthetic_corpus(2000, 20, device)
    env = trajs.env  # SyntheticTokenEnv with letters tokenizer.

    # Propose with BPE and WordPiece; expect 'BCADB' among new tokens (present in all sequences)
    bpe = BPEChunker(unk_token="[UNK]", delimiter="")
    wp = WordPieceChunker(unk_token="[UNK]", delimiter="")

    proposed_bpe = set(
        bpe.propose_tokens(env, trajs, n_tokens_to_add=50, remove_old=False)
    )
    proposed_wp = set(
        wp.propose_tokens(env, trajs, n_tokens_to_add=50, remove_old=False)
    )

    assert "BCADB" in proposed_bpe
    assert "BCADB" in proposed_wp


def test_macro_masking():
    device = torch.device("cpu")
    trajs = generate_synthetic_corpus(2000, 20, device)
    env = trajs.env  # SyntheticTokenEnv with letters tokenizer.

    # Add macro keys to env vocab (tuple form for executable macros)
    new_ids = env.add_tokens([(2, 3, 1, 4, 2)])
    assert len(new_ids) == 1
    assert env.id_to_token_key[-1] == (2, 3, 1, 4, 2)  # BCADB, the new macro-action.
    assert env.id_to_token_key[-2] == "<EXIT>"
    assert env.id_to_token_key[:26] == list(range(26))  # The alphabet.

    # Macro should be feasible from a generic state (start at s0)
    state_0 = env.states_from_tensor(torch.tensor([[0]], device=device))
    env.update_masks(state_0)
    macro_id = new_ids[0]  # BCADB, allowed.
    assert state_0.forward_masks[0, macro_id].item()

    # Macro should be infeasable from a generic state.
    new_ids = env.add_tokens([(2, 3, 4, 4, 2)])  # BCDDB, disallowed (no C->D allowed).
    state_0 = env.states_from_tensor(torch.tensor([[0]], device=device))
    env.update_masks(state_0)
    macro_id = new_ids[0]  # BCDDB, disallowed.
    assert not state_0.forward_masks[0, macro_id].item()


def test_macro_mask_guard_no_recursion_batch_only():
    device = torch.device("cpu")
    env = SyntheticTokenEnv(device)
    # Simple (B,) batch
    B = 2
    states_tensor = torch.zeros((B, 1), dtype=torch.long, device=device)
    states = cast(ChunkedStates, env.states_from_tensor(states_tensor))

    # Should not recurse
    env.update_masks(states)
    mask = env.compute_strict_macro_forward_mask(states)
    assert mask.shape == (states.batch_shape[0], env.n_actions)
    assert mask.dtype == torch.bool
    assert getattr(env, "_macro_overlay_depth", 0) == 0


def test_macro_mask_guard_no_recursion_trajectories():
    device = torch.device("cpu")
    env = SyntheticTokenEnv(device)
    # (T,B) batch
    T, B = 3, 2
    states_tensor = torch.zeros((T, B, 1), dtype=torch.long, device=device)
    states = cast(ChunkedStates, env.states_from_tensor(states_tensor))

    # Should not recurse
    env.update_masks(states)
    mask = env.compute_strict_macro_forward_mask(states)
    assert mask.shape == (T, B, env.n_actions)
    assert mask.dtype == torch.bool
    assert getattr(env, "_macro_overlay_depth", 0) == 0


def test_horizon_mask_blocks_oversized_macro():
    device = torch.device("cpu")
    env = SyntheticTokenEnv(device)

    # Register a length-3 macro
    macro_ids = env.add_tokens([(2, 2, 2)])
    macro_id = macro_ids[0]

    # Build (T=3, B=2) states: remaining steps = 3,2,1 at t=0,1,2
    T, B = 3, 2
    states_tensor = torch.zeros((T, B, 1), dtype=torch.long, device=device)
    states = cast(ChunkedStates, env.states_from_tensor(states_tensor))
    env.update_masks(states)
    mask = env.compute_strict_macro_forward_mask(states)

    # At t=0: remaining=3 -> macro allowed by horizon check
    assert mask[0, :, macro_id].all().item()

    # At t>=1: remaining < 3 -> macro disallowed
    assert (~mask[1:, :, macro_id]).all().item()


def test_apply_macro_forward_mask_noop_under_guard():
    device = torch.device("cpu")
    env = SyntheticTokenEnv(device)
    B = 2
    states_tensor = torch.zeros((B, 1), dtype=torch.long, device=device)
    states = cast(ChunkedStates, env.states_from_tensor(states_tensor))
    env.update_masks(states)
    fwd_before = states.forward_masks.clone()

    # Manually enter guard
    setattr(env, "_macro_overlay_depth", getattr(env, "_macro_overlay_depth", 0) + 1)
    try:
        env.apply_macro_forward_mask(states)
    finally:
        setattr(env, "_macro_overlay_depth", getattr(env, "_macro_overlay_depth", 1) - 1)
    assert torch.equal(states.forward_masks, fwd_before)
