from typing import Literal

import pytest
import torch

from gfn.utils.modules import (
    RecurrentDiscreteSequenceModel,
    TransformerDiscreteSequenceModel,
)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_recurrent_smoke(rnn_type: Literal["lstm", "gru"], device: torch.device) -> None:
    batch_size = 2
    vocab_size = 11
    total_steps = 4
    model = RecurrentDiscreteSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=5,
        hidden_size=7,
        num_layers=2,
        rnn_type=rnn_type,
        dropout=0.0,
    ).to(device)
    model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, total_steps), device=device)

    def collect_logits(
        chunk_sizes: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        carry = model.init_carry(batch_size, device)
        outputs: list[torch.Tensor] = []
        start = 0
        with torch.no_grad():
            for chunk in chunk_sizes:
                end = start + chunk
                logits, carry = model(tokens[:, start:end], carry)
                outputs.append(logits)
                start = end
        if start != total_steps:
            raise ValueError("Chunk sizes must cover the entire sequence length.")
        return torch.cat(outputs, dim=1), carry

    logits_all, carry_all = collect_logits([total_steps])
    logits_single, carry_single = collect_logits([1] * total_steps)
    logits_double, carry_double = collect_logits([2, 2])

    scripted = torch.jit.script(model)
    carry_script = model.init_carry(batch_size, device)
    with torch.no_grad():
        logits_script, carry_script = scripted(tokens, carry_script)

    assert torch.allclose(logits_all, logits_single, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_all, logits_double, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_all, logits_script, atol=1e-6, rtol=1e-5)

    assert torch.allclose(
        carry_all["hidden"], carry_single["hidden"], atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        carry_all["hidden"], carry_double["hidden"], atol=1e-6, rtol=1e-5
    )

    if rnn_type == "lstm":
        assert torch.allclose(
            carry_all["cell"], carry_single["cell"], atol=1e-6, rtol=1e-5
        )
        assert torch.allclose(
            carry_all["cell"], carry_double["cell"], atol=1e-6, rtol=1e-5
        )


@pytest.mark.parametrize("positional_embedding", ["learned", "sinusoidal"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_transformer_smoke(
    positional_embedding: Literal["learned", "sinusoidal"],
    device: torch.device,
) -> None:
    batch_size = 3
    vocab_size = 13
    total_steps = 4
    model = TransformerDiscreteSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=12,
        num_heads=3,
        ff_hidden_dim=24,
        num_layers=2,
        max_position_embeddings=32,
        dropout=0.0,
        positional_embedding=positional_embedding,
    ).to(device)
    model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size, total_steps), device=device)

    def collect_logits(
        chunk_sizes: list[int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        carry = model.init_carry(batch_size, device)
        outputs: list[torch.Tensor] = []
        start = 0
        with torch.no_grad():
            for chunk in chunk_sizes:
                end = start + chunk
                logits, carry = model(tokens[:, start:end], carry)
                outputs.append(logits)
                start = end
        if start != total_steps:
            raise ValueError("Chunk sizes must cover the entire sequence length.")
        return torch.cat(outputs, dim=1), carry

    logits_all, carry_all = collect_logits([total_steps])
    logits_single, carry_single = collect_logits([1] * total_steps)
    logits_double, carry_double = collect_logits([2, 2])

    scripted = torch.jit.script(model)
    carry_script = model.init_carry(batch_size, device)

    with torch.no_grad():
        logits_script, carry_script = scripted(tokens, carry_script)

    assert torch.allclose(logits_all, logits_single, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_all, logits_double, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logits_all, logits_script, atol=1e-6, rtol=1e-5)
    assert torch.equal(carry_all["position"], carry_single["position"])
    assert torch.equal(carry_all["position"], carry_double["position"])

    def carry_matches(
        ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]
    ) -> bool:
        for idx in range(model.num_layers):
            key_name = model.key_names[idx]
            value_name = model.value_names[idx]
            if not torch.allclose(ref[key_name], other[key_name], atol=1e-6, rtol=1e-5):
                return False
            if not torch.allclose(
                ref[value_name], other[value_name], atol=1e-6, rtol=1e-5
            ):
                return False
        return True

    assert carry_matches(carry_all, carry_single)
    assert carry_matches(carry_all, carry_double)

    for idx in range(model.num_layers):
        assert (
            carry_all[f"key_{idx}"].size(2)
            == carry_all[f"value_{idx}"].size(2)
            == total_steps
        )
