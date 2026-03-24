from typing import Literal

import pytest
import torch

from gfn.actions import GraphActions
from gfn.estimators import DiscreteGraphPolicyEstimator
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.utils.modules import (
    GraphActionGNN,
    GraphEdgeActionMLP,
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


# ---------------------------------------------------------------------------
# Graph module tests
# ---------------------------------------------------------------------------

N_NODES = 6
NUM_NODE_CLASSES = 6
NUM_EDGE_CLASSES = 1
EMBEDDING_DIM = 32


def _make_graph_env(directed: bool = False) -> GraphBuildingOnEdges:
    return GraphBuildingOnEdges(
        n_nodes=N_NODES,
        state_evaluator=lambda s: torch.ones(len(s)),
        directed=directed,
        device=torch.device("cpu"),
    )


def _make_module(module_cls, is_backward: bool, directed: bool):
    """Instantiate either GraphActionGNN or GraphEdgeActionMLP."""
    if module_cls is GraphActionGNN:
        return module_cls(
            num_node_classes=NUM_NODE_CLASSES,
            directed=directed,
            num_edge_classes=NUM_EDGE_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            is_backward=is_backward,
        )
    else:
        return module_cls(
            n_nodes=N_NODES,
            directed=directed,
            num_node_classes=NUM_NODE_CLASSES,
            num_edge_classes=NUM_EDGE_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            is_backward=is_backward,
        )


def _sample_valid_states(module_cls, directed, n=4):
    """Sample trajectories and return non-sink terminating states."""
    env = _make_graph_env(directed)
    pf = DiscreteGraphPolicyEstimator(module=_make_module(module_cls, False, directed))
    pb = DiscreteGraphPolicyEstimator(
        module=_make_module(module_cls, True, directed),
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb)
    traj = gflownet.sample_trajectories(env, n=n, save_logprobs=True)
    return env, gflownet, traj


@pytest.mark.parametrize("module_cls", [GraphActionGNN, GraphEdgeActionMLP])
@pytest.mark.parametrize("is_backward", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_graph_module_output_shapes(module_cls, is_backward, directed):
    """Forward pass returns TensorDict with correct keys and 2D shapes."""
    env, _, traj = _sample_valid_states(module_cls, directed)
    module = _make_module(module_cls, is_backward, directed)

    # Use terminating states — guaranteed to be valid (non-sink) graphs.
    states = traj.terminating_states
    states_tensor = states.tensor

    with torch.no_grad():
        out = module(states_tensor)

    expected_keys = {
        GraphActions.ACTION_TYPE_KEY,
        GraphActions.NODE_CLASS_KEY,
        GraphActions.NODE_INDEX_KEY,
        GraphActions.EDGE_CLASS_KEY,
        GraphActions.EDGE_INDEX_KEY,
    }
    assert set(out.keys()) == expected_keys

    B = len(states)
    for key in expected_keys:
        assert out[key].ndim == 2, f"{key} has ndim={out[key].ndim}, expected 2"
        assert (
            out[key].shape[0] == B
        ), f"{key} batch dim is {out[key].shape[0]}, expected {B}"


@pytest.mark.parametrize("module_cls", [GraphActionGNN, GraphEdgeActionMLP])
@pytest.mark.parametrize("directed", [False, True])
def test_graph_module_output_shapes_on_empty_graphs(module_cls, directed):
    """Forward pass works on initial (empty) states."""
    env = _make_graph_env(directed)
    module = _make_module(module_cls, False, directed)

    states = env.states_from_batch_shape((4,))
    with torch.no_grad():
        out = module(states.tensor)

    B = 4
    for key in out.keys():
        assert out[key].ndim == 2, f"{key} has ndim={out[key].ndim}, expected 2"
        assert out[key].shape[0] == B


@pytest.mark.parametrize("module_cls", [GraphActionGNN, GraphEdgeActionMLP])
@pytest.mark.parametrize("directed", [False, True])
def test_graph_tb_pipeline(module_cls, directed):
    """Full TB training step works with both GNN and MLP modules."""
    env = _make_graph_env(directed)
    pf = DiscreteGraphPolicyEstimator(module=_make_module(module_cls, False, directed))
    pb = DiscreteGraphPolicyEstimator(
        module=_make_module(module_cls, True, directed),
        is_backward=True,
    )
    gflownet = TBGFlowNet(pf, pb)
    optimizer = torch.optim.Adam(gflownet.parameters(), lr=1e-3)

    traj = gflownet.sample_trajectories(env, n=8, save_logprobs=True)
    samples = gflownet.to_training_samples(traj)
    optimizer.zero_grad()
    loss = gflownet.loss(env, samples, recalculate_all_logprobs=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
