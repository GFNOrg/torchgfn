from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

import torch
from torch import nn

from gfn.states import DiscreteStates

if TYPE_CHECKING:
    from gfn.env import ChunkedDiscreteEnvironment


class ChunkedPolicy(nn.Module):
    """Compute logits over a macro library via state and macro embeddings.

    The `state_module` maps preprocessed states to a fixed-size embedding. The
    `action_encoder` maps action sequences (macros) to the same embedding space.
    Logits are the scaled dot products between state embeddings and macro embeddings.
    """

    def __init__(
        self,
        state_module: nn.Module,
        action_encoder: ActionEncoder,
        env: "ChunkedDiscreteEnvironment",
        action_embedding_dim: int,
        primitive_id_mapper: Callable[[int], int] | None = None,
    ) -> None:
        super().__init__()
        self.state_module = state_module
        self.action_encoder = action_encoder
        self.env = env
        self.action_embedding_dim = int(action_embedding_dim)
        self.primitive_id_mapper: Callable[[int], int] = (
            primitive_id_mapper if primitive_id_mapper is not None else (lambda x: x)
        )
        self.register_buffer("_library_embeddings", torch.empty(0))

    @torch.no_grad()
    def refresh_library_embeddings(self, device: torch.device) -> None:
        # Build sequences from env vocab decoded to primitive ids
        vocab = self.env.vocab
        seqs: List[List[int]] = []
        max_len = 0

        # TODO: This should rely on the env tokenizer instead of primitive_id_mapper.
        for key in vocab:
            decoded = list(self.env.decode_key_to_actions(key))
            mapped = [self.primitive_id_mapper(i) for i in decoded]
            max_len = max(max_len, len(mapped))

            # Ensure at least length 1 for encoder stability.
            seqs.append(mapped if len(mapped) > 0 else [0])  # 0 is a placeholder FIXME.

        if len(seqs) == 0:
            self._library_embeddings = torch.empty(
                0, self.action_embedding_dim, device=device
            )
            return

        x = torch.full(
            (len(seqs), max_len),
            fill_value=0,
            dtype=torch.long,
            device=device,
        )

        # TODO: Vectorize this.
        for i, s in enumerate(seqs):
            if len(s) > 0:
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        self._library_embeddings = self.action_encoder(x)  # (N, D)

    def forward_logits(self, states: DiscreteStates) -> torch.Tensor:
        state_emb = self.state_module(states)  # (*B, D)
        if (
            self._library_embeddings.numel() == 0
            or self._library_embeddings.shape[0] != self.env.n_actions
        ):
            self.refresh_library_embeddings(device=state_emb.device)

        logits = torch.einsum("bd,nd->bn", state_emb, self._library_embeddings)
        logits = logits / (state_emb.shape[-1] ** 0.5)

        return logits


class ActionModel(nn.Module):
    def __init__(
        self,
        n_primitive_actions: int,
        hidden_dim: int = 256,
        action_embedding_dimension: int = 128,
    ) -> None:
        super().__init__()
        self.primitive_embedding = nn.Embedding(n_primitive_actions, hidden_dim)
        self.rnn_encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_embedding_dimension)
        )

    def forward(self, x):  # x: (B, L)
        emb = self.primitive_embedding(x)
        s, _ = self.rnn_encoder(emb)
        out = s[:, -1]
        out = self.out_layer(out)
        return out


class PositionalEncoding(nn.Module):
    """Minimal sinusoidal positional encoding for completeness.

    If a richer implementation exists elsewhere, prefer importing it.
    """

    pe: torch.Tensor  # registered buffer

    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.get_default_dtype()).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.get_default_dtype(),
            )
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : dim // 2]

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:L]
        return self.dropout(x)


class ActionEncoder(nn.Module):
    def __init__(
        self,
        n_primitive_actions: int,
        action_embedding_dimension: int,
        hidden_dim: int,
        num_layers: int,
        num_head: int,
        max_len: int = 60,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len + 1)
        self.embedding = nn.Embedding(n_primitive_actions, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim, num_head, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # TODO: For the action encoder to work properly with macros of variable length,
        # do we need the embedding layer to be recurrent. Or can we just use a simple
        # embedding layer?
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dimension)

    def forward(self, x_ids):  # (B, L) with 0 = PAD
        pad = x_ids == 0  # (B, L) bool
        x = self.embedding(x_ids)  # (B, L, D)
        x = self.pos(x)  # (B, L, D)
        x = self.encoder(x, src_key_padding_mask=pad)  # mask pads in attention

        mask = (~pad).unsqueeze(-1)  # (B, L, 1)
        denom = mask.sum(dim=1).clamp_min(1)  # (B, 1)
        pooled = (x * mask).sum(dim=1) / denom  # (B, D)

        return self.action_embedding_layer(pooled)
