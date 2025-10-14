from typing import Literal, Tuple, cast

import pytest
import torch
from torch.distributions import Categorical

from gfn.containers import Trajectories, Transitions
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.estimators import PolicyMixin  # Use policy mixin directly instead of adapters
from gfn.estimators import (
    RecurrentPolicyMixin,  # Use recurrent policy mixin instead of adapters
)
from gfn.estimators import RolloutContext  # New rollout context used by PolicyMixin
from gfn.estimators import (
    DiscreteGraphPolicyEstimator,
    DiscretePolicyEstimator,
    Estimator,
)
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.graph_building import GraphBuildingOnEdges
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.preprocessors import (
    EnumPreprocessor,
    IdentityPreprocessor,
    KHotPreprocessor,
    OneHotPreprocessor,
)
from gfn.samplers import (
    LocalSearchSampler,
    Sampler,
)
from gfn.states import States
from gfn.utils.modules import (
    MLP,
    GraphActionGNN,
    RecurrentDiscreteSequenceModel,
    TransformerDiscreteSequenceModel,
)
from gfn.utils.prob_calculations import get_trajectory_pfs
from gfn.utils.training import states_actions_tns_to_traj


def trajectory_sampling_with_return(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"],
    preprocessor_name: Literal["Identity", "KHot", "OneHot", "Enum"],
    delta: float,
    n_components: int,
    n_components_s0: int,
) -> Tuple[Trajectories, Trajectories, Estimator, Estimator]:
    if preprocessor_name != "Identity" and env_name != "HyperGrid":
        pytest.skip("Useless tests")
    if (delta != 0.1 or n_components != 1 or n_components_s0 != 1) and env_name != "Box":
        pytest.skip("Useless tests")

    if env_name in ["HyperGrid", "DiscreteEBM"]:
        if env_name == "HyperGrid":
            env = HyperGrid(ndim=2, height=8, validate_modes=False)
            if preprocessor_name == "KHot":
                preprocessor = KHotPreprocessor(env.height, env.ndim)
            elif preprocessor_name == "OneHot":
                preprocessor = OneHotPreprocessor(
                    n_states=env.n_states, get_states_indices=env.get_states_indices
                )
            elif preprocessor_name == "Identity":
                preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
            elif preprocessor_name == "Enum":
                preprocessor = EnumPreprocessor(env.get_states_indices)
            else:
                raise ValueError("Invalid preprocessor name")

        elif env_name == "DiscreteEBM":
            env = DiscreteEBM(ndim=8)
            preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])

        assert isinstance(preprocessor.output_dim, int)

        pf_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
        pb_module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions - 1)
        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=preprocessor,
        )

    elif env_name == "Box":
        env = Box(delta=delta)
        pf_module = BoxPFMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=n_components,
            n_components_s0=n_components_s0,
        )
        pb_module = BoxPBMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=n_components,
            trunk=pf_module.trunk,
        )
        pf_estimator = BoxPFEstimator(
            env=env,
            module=pf_module,
            n_components=n_components,
            n_components_s0=n_components_s0,
        )
        pb_estimator = BoxPBEstimator(
            env=env, module=pb_module, n_components=n_components
        )
    elif env_name == "GraphBuildingOnEdges":
        env = GraphBuildingOnEdges(
            n_nodes=10,
            state_evaluator=lambda s: torch.zeros(s.batch_shape),
            directed=False,
            device=torch.device("cpu"),
        )
        pf_module = GraphActionGNN(
            num_node_classes=env.n_nodes,
            directed=env.is_directed,
            num_edge_classes=env.num_edge_classes,
            embedding_dim=128,
            is_backward=False,
        )
        pb_module = GraphActionGNN(
            num_node_classes=env.n_nodes,
            directed=env.is_directed,
            num_edge_classes=env.num_edge_classes,
            num_conv_layers=1,
            embedding_dim=128,
            is_backward=True,
        )
        pf_estimator = DiscreteGraphPolicyEstimator(
            module=pf_module,
            is_backward=False,
        )
        pb_estimator = DiscreteGraphPolicyEstimator(
            module=pb_module,
            is_backward=True,
        )
    else:
        raise ValueError("Unknown environment name")

    sampler = Sampler(estimator=pf_estimator)
    # Test mode collects log_probs and estimator_ouputs, not encountered in the wild.
    trajectories = sampler.sample_trajectories(
        env,
        n=5,
        save_logprobs=True,
        save_estimator_outputs=False,  # FIXME: This fails on GraphBuildingOnEdges if True
    )

    states = env.reset(batch_shape=5, random=True)
    bw_sampler = Sampler(estimator=pb_estimator)
    bw_trajectories = bw_sampler.sample_trajectories(
        env, save_logprobs=True, states=states, save_estimator_outputs=False
    )

    return trajectories, bw_trajectories, pf_estimator, pb_estimator


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
@pytest.mark.parametrize("preprocessor_name", ["KHot", "OneHot", "Identity", "Enum"])
@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8])
@pytest.mark.parametrize("n_components_s0", [1, 2, 5])
@pytest.mark.parametrize("n_components", [1, 2, 5])
def test_trajectory_sampling(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"],
    preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"],
    delta: float,
    n_components_s0: int,
    n_components: int,
):
    _ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,
        delta,
        n_components_s0,
        n_components,
    )


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
def test_trajectories_getitem(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
):
    try:
        _ = trajectory_sampling_with_return(
            env_name,
            preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
            delta=0.1,
            n_components=1,
            n_components_s0=1,
        )
    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
def test_trajectories_extend(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
):
    trajectories, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    try:
        trajectories.extend(trajectories[[1, 0]])
    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
def test_sub_sampling(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
):
    trajectories, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    try:
        _ = trajectories.sample(n_samples=2)
    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_reverse_backward_trajectories(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box"]
):
    """
    Ensures that the vectorized `Trajectories.reverse_backward_trajectories`
    matches the for-loop approach by toggling `debug=True`.
    """
    _, backward_trajectories, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )

    reversed_traj = backward_trajectories.reverse_backward_trajectories()

    for i in range(len(backward_trajectories)):
        terminating_idx = backward_trajectories.terminating_idx[i]
        for j in range(terminating_idx):
            assert torch.all(
                reversed_traj.actions.tensor[j, i]
                == backward_trajectories.actions.tensor[terminating_idx - j - 1, i]
            )
            assert torch.all(
                reversed_traj.states.tensor[j, i]
                == backward_trajectories.states.tensor[terminating_idx - j, i]
            )

        assert torch.all(reversed_traj.actions[terminating_idx, i].is_exit)
        assert torch.all(reversed_traj.states[terminating_idx + 1, i].is_sink_state)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_local_search_for_loop_equivalence(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box"]
):
    """
    Ensures that the vectorized `LocalSearchSampler.local_search` matches
    the for-loop approach by toggling `debug=True`.
    """
    # Build environment
    is_discrete = env_name in ["HyperGrid", "DiscreteEBM"]
    if is_discrete:
        if env_name == "HyperGrid":
            env = HyperGrid(ndim=2, height=5, validate_modes=False)
            preprocessor = KHotPreprocessor(env.height, env.ndim)
        elif env_name == "DiscreteEBM":
            env = DiscreteEBM(ndim=5)
            preprocessor = IdentityPreprocessor(output_dim=env.state_shape[-1])
        else:
            raise ValueError("Unknown environment name")

        assert isinstance(preprocessor.output_dim, int)
        # Build pf & pb
        pf_module = MLP(preprocessor.output_dim, env.n_actions)
        pb_module = MLP(preprocessor.output_dim, env.n_actions - 1)
        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=preprocessor,
        )

    else:
        env = Box(delta=0.1)

        # Build pf & pb
        pf_module = BoxPFMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=1,
            n_components_s0=1,
        )
        pb_module = BoxPBMLP(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=1,
            trunk=pf_module.trunk,
        )
        pf_estimator = BoxPFEstimator(
            env=env,
            module=pf_module,
            n_components=1,
            n_components_s0=1,
        )
        pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    # Build sampler
    sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)

    # Initial forward-sampler call
    trajectories = sampler.sample_trajectories(env, n=3, save_logprobs=True)

    # Now run local_search in debug mode so that for-loop logic is compared
    # to the vectorized logic.
    # If there's any mismatch, local_search() will raise AssertionError
    try:
        _ = sampler.local_search(
            env,
            trajectories,
            save_logprobs=True,
            back_ratio=0.5,
            use_metropolis_hastings=True,
            debug=True,  # <--- TRIGGER THE COMPARISON
        )
    except Exception as e:
        raise ValueError(
            f"Error while testing LocalSearchSampler.local_search in {env_name}"
        ) from e


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
def test_to_transition(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
):
    """
    Ensures that the `Trajectories.to_transitions` method works as expected.
    """
    trajectories, bwd_trajectories, pf_estimator, _ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )

    _ = trajectories.to_transitions()

    bwd_trajectories = Trajectories.reverse_backward_trajectories(bwd_trajectories)
    # evaluate with pf_estimator
    backward_traj_pfs = get_trajectory_pfs(
        pf=pf_estimator,
        trajectories=bwd_trajectories,
        recalculate_all_logprobs=False,
    )
    bwd_trajectories.log_probs = backward_traj_pfs
    _ = bwd_trajectories.to_transitions()


@pytest.mark.parametrize(
    "env_name", ["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"]
)
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(
    env_name: Literal["HyperGrid", "DiscreteEBM", "Box", "GraphBuildingOnEdges"],
    objects: Literal["trajectories", "transitions"],
):
    """Test that the replay buffer works correctly with different types of objects."""
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=4, validate_modes=False)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=8)
    elif env_name == "Box":
        env = Box(delta=0.1)
    elif env_name == "GraphBuildingOnEdges":
        env = GraphBuildingOnEdges(
            n_nodes=10,
            state_evaluator=lambda s: torch.zeros(s.batch_shape),
            directed=False,
            device=torch.device("cpu"),
        )
    else:
        raise ValueError("Unknown environment name")
    replay_buffer = ReplayBuffer(env, capacity=10)
    trajectories, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    try:
        if objects == "trajectories":
            # Filter out trajectories that are at max length
            training_objects = trajectories
            training_objects_2 = trajectories[
                trajectories.terminating_idx != trajectories.max_length
            ]
            replay_buffer.add(training_objects_2)

        else:
            training_objects = trajectories.to_transitions()

        # Add objects multiple times to test buffer behavior
        replay_buffer.add(training_objects)
        replay_buffer.add(training_objects)
        replay_buffer.add(training_objects)
        replay_buffer.add(training_objects)

        # Test that we can sample from the buffer
        sampled = replay_buffer.sample(5)
        assert len(sampled) == 5
        if objects == "trajectories":
            assert isinstance(sampled, Trajectories)
        else:
            assert isinstance(sampled, Transitions)

    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e


def test_states_actions_tns_to_traj():
    env = HyperGrid(2, 4, validate_modes=False)
    states = torch.tensor([[0, 0], [0, 1], [0, 2], [-1, -1]])
    actions = torch.tensor([1, 1, 2])
    trajs = states_actions_tns_to_traj(states, actions, env)

    # Test that we can add the trajectories to a replay buffer
    replay_buffer = ReplayBuffer(env, capacity=10)
    replay_buffer.add(trajs)


# ---------------------- Adapters: unit-level smoke tests ----------------------


class _FakeStates:
    def __init__(self, n: int, device: torch.device):
        self.tensor = torch.zeros((n, 1), device=device)

    @property
    def batch_shape(self):
        return (self.tensor.shape[0],)


class _DummyPolicy(PolicyMixin):
    is_backward = False

    # Minimal callable module that matches the `PolicyMixin` expectation of `self.module`
    class _Module:
        def __call__(
            self, states: _FakeStates, conditioning: torch.Tensor | None = None
        ):
            n = states.batch_shape[0]
            return torch.zeros((n, 3), device=states.tensor.device)

    def __init__(self):
        # The mixin calls `self.module(...)`; we provide a tiny callable to produce logits
        self.module = self._Module()

    def to_probability_distribution(
        self, states: _FakeStates, est_out: torch.Tensor, **_: dict
    ):
        # Build a simple categorical policy directly from the provided logits
        return Categorical(logits=est_out)

    def __call__(self, states: _FakeStates, conditioning: torch.Tensor | None = None):
        return self.module(states, conditioning)


class _DummyRecurrentPolicy(RecurrentPolicyMixin):
    is_backward = False

    def init_carry(self, batch_size: int, device: torch.device):
        # Provide a simple hidden state that increments each step
        return {"hidden": torch.zeros((batch_size, 2), device=device)}

    def __call__(self, states: _FakeStates, carry: dict[str, torch.Tensor]):
        # Produce trivial logits and update the carry
        n = states.batch_shape[0]
        logits = torch.zeros((n, 3), device=states.tensor.device)
        new_carry = {"hidden": carry["hidden"] + 1}
        return logits, new_carry

    def to_probability_distribution(
        self, states: _FakeStates, est_out: torch.Tensor, **_: dict
    ):
        return Categorical(logits=est_out)


def test_rollout_context_basic():
    ctx = RolloutContext(batch_size=4, device=torch.device("cpu"), conditioning=None)
    assert ctx.batch_size == 4
    assert ctx.device.type == "cpu"
    # extras supports arbitrary entries
    ctx.extras["foo"] = 123
    assert ctx.extras["foo"] == 123


def test_default_adapter_compute_record():
    # Adapted to directly use a policy implementing `PolicyMixin`
    policy = _DummyPolicy()
    device = torch.device("cpu")
    n = 5
    states = _FakeStates(n, device)
    ctx = policy.init_context(n, device, conditioning=None)

    step_mask = torch.ones(n, dtype=torch.bool, device=device)
    dist, ctx = policy.compute_dist(
        cast(States, states), ctx, step_mask, save_estimator_outputs=True
    )
    actions = dist.sample()
    _, ctx = policy.log_probs(
        actions, dist, ctx, step_mask, vectorized=False, save_logprobs=True
    )
    stacked_logprobs = (
        torch.stack(ctx.trajectory_log_probs, dim=0)
        if ctx.trajectory_log_probs
        else None
    )
    stacked_estimator_outputs = (
        torch.stack(ctx.trajectory_estimator_outputs, dim=0)
        if ctx.trajectory_estimator_outputs
        else None
    )
    assert stacked_logprobs is not None
    assert stacked_logprobs.shape == (1, n)
    assert stacked_estimator_outputs is not None
    assert stacked_estimator_outputs.shape[:2] == (1, n)


def test_recurrent_adapter_requires_init_carry():
    # Recurrent policies must implement `init_carry`; verify error when missing
    class _BadRecurrentPolicy(RecurrentPolicyMixin):
        is_backward = False

    with pytest.raises(TypeError, match="requires.*init_carry"):
        _ = _BadRecurrentPolicy().init_context(2, torch.device("cpu"), conditioning=None)


def test_recurrent_adapter_flow():
    # Adapted to directly use a policy implementing `RecurrentPolicyMixin`
    policy = _DummyRecurrentPolicy()
    device = torch.device("cpu")
    n = 3
    states = _FakeStates(n, device)
    ctx = policy.init_context(n, device, conditioning=None)

    step_mask = torch.ones(n, dtype=torch.bool, device=device)
    dist, ctx = policy.compute_dist(
        cast(States, states), ctx, step_mask, save_estimator_outputs=True
    )
    actions = dist.sample()
    # carry should update when we record multiple steps
    h0 = ctx.carry["hidden"].clone()
    _, ctx = policy.log_probs(
        actions, dist, ctx, step_mask, vectorized=False, save_logprobs=True
    )
    # second step
    dist, ctx = policy.compute_dist(
        cast(States, states), ctx, step_mask, save_estimator_outputs=True
    )
    actions = dist.sample()
    _, ctx = policy.log_probs(
        actions, dist, ctx, step_mask, vectorized=False, save_logprobs=True
    )
    h1 = ctx.carry["hidden"].clone()
    assert torch.all(h1 == h0 + 1)
    stacked_logprobs = (
        torch.stack(ctx.trajectory_log_probs, dim=0)
        if ctx.trajectory_log_probs
        else None
    )
    stacked_estimator_outputs = (
        torch.stack(ctx.trajectory_estimator_outputs, dim=0)
        if ctx.trajectory_estimator_outputs
        else None
    )
    assert stacked_logprobs is not None
    assert stacked_logprobs.shape == (2, n)
    assert stacked_estimator_outputs is not None
    assert stacked_estimator_outputs.shape[:2] == (2, n)


# ---------------------- Integration with real recurrent modules ----------------------


class _SeqStates:
    def __init__(self, tokens: torch.Tensor, n_actions: int):
        self.tensor = tokens  # (batch, seq_len)
        b = tokens.shape[0]
        device = tokens.device
        self.forward_masks = torch.ones((b, n_actions), dtype=torch.bool, device=device)
        self.backward_masks = torch.ones(
            (b, max(n_actions - 1, 1)), dtype=torch.bool, device=device
        )

    @property
    def batch_shape(self):
        return (self.tensor.shape[0],)

    @property
    def device(self):
        return self.tensor.device


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_integration_recurrent_sequence_model_with_adapter(
    rnn_type: Literal["lstm", "gru"]
) -> None:
    device = torch.device("cpu")
    batch_size = 3
    vocab_size = 11
    seq_len = 4

    model = RecurrentDiscreteSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=8,
        hidden_size=16,
        num_layers=1,
        rnn_type=rnn_type,
        dropout=0.0,
    ).to(device)

    from gfn.estimators import RecurrentDiscretePolicyEstimator

    estimator = RecurrentDiscretePolicyEstimator(
        module=model,
        n_actions=vocab_size,
        is_backward=False,
    )

    # Use the estimator directly via `RecurrentPolicyMixin`
    ctx = estimator.init_context(batch_size, device, conditioning=None)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    states = _SeqStates(tokens, vocab_size)

    # Run two steps and verify carry and artifact shapes
    step_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    for _ in range(2):
        dist, ctx = estimator.compute_dist(
            cast(States, states), ctx, step_mask, save_estimator_outputs=True
        )
        actions = dist.sample()
        _, ctx = estimator.log_probs(
            actions, dist, ctx, step_mask, vectorized=False, save_logprobs=True
        )

    stacked_logprobs = (
        torch.stack(ctx.trajectory_log_probs, dim=0)
        if ctx.trajectory_log_probs
        else None
    )
    stacked_estimator_outputs = (
        torch.stack(ctx.trajectory_estimator_outputs, dim=0)
        if ctx.trajectory_estimator_outputs
        else None
    )

    assert stacked_logprobs is not None
    assert stacked_logprobs.shape[0] == 2
    assert stacked_estimator_outputs is not None
    assert stacked_estimator_outputs.shape[0] == 2


@pytest.mark.parametrize("positional_embedding", ["learned", "sinusoidal"])
def test_integration_transformer_sequence_model_with_adapter(
    positional_embedding: Literal["learned", "sinusoidal"]
) -> None:
    device = torch.device("cpu")
    batch_size = 2
    vocab_size = 9
    seq_len = 5

    model = TransformerDiscreteSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=12,
        num_heads=3,
        ff_hidden_dim=24,
        num_layers=1,
        max_position_embeddings=32,
        dropout=0.0,
        positional_embedding=positional_embedding,
    ).to(device)

    from gfn.estimators import RecurrentDiscretePolicyEstimator

    estimator = RecurrentDiscretePolicyEstimator(
        module=model,
        n_actions=vocab_size,
        is_backward=False,
    )

    # Use the estimator directly via `RecurrentPolicyMixin`
    ctx = estimator.init_context(batch_size, device, conditioning=None)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    states = _SeqStates(tokens, vocab_size)

    step_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    dist, ctx = estimator.compute_dist(
        cast(States, states), ctx, step_mask, save_estimator_outputs=True
    )
    actions = dist.sample()
    _, ctx = estimator.log_probs(
        actions, dist, ctx, step_mask, vectorized=False, save_logprobs=True
    )

    stacked_logprobs = (
        torch.stack(ctx.trajectory_log_probs, dim=0)
        if ctx.trajectory_log_probs
        else None
    )
    stacked_estimator_outputs = (
        torch.stack(ctx.trajectory_estimator_outputs, dim=0)
        if ctx.trajectory_estimator_outputs
        else None
    )
    assert stacked_logprobs is not None
    assert stacked_logprobs.shape[0] == 1
    assert stacked_estimator_outputs is not None
    assert stacked_estimator_outputs.shape[0] == 1
