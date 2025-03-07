from typing import Literal, Tuple

import pytest
import torch

from gfn.containers import Trajectories, Transitions
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.modules import DiscretePolicyEstimator, GFNModule
from gfn.samplers import LocalSearchSampler, Sampler
from gfn.utils.modules import MLP
from gfn.utils.prob_calculations import get_trajectory_pfs
from gfn.utils.training import states_actions_tns_to_traj


def trajectory_sampling_with_return(
    env_name: str,
    preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"],
    delta: float,
    n_components_s0: int,
    n_components: int,
) -> Tuple[Trajectories, Trajectories, GFNModule, GFNModule]:
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=8, preprocessor_name=preprocessor_name)
    elif env_name == "DiscreteEBM":
        if preprocessor_name != "Identity" or delta != 0.1:
            pytest.skip("Useless tests")
        env = DiscreteEBM(ndim=8)
    elif env_name == "Box":
        if preprocessor_name != "Identity":
            pytest.skip("Useless tests")
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
    else:
        raise ValueError("Unknown environment name")

    if env_name != "Box":
        assert not isinstance(env, Box)
        pf_module = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
        pb_module = MLP(
            input_dim=env.preprocessor.output_dim, output_dim=env.n_actions - 1
        )
        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=env.preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
        )

    sampler = Sampler(estimator=pf_estimator)
    # Test mode collects log_probs and estimator_ouputs, not encountered in the wild.
    trajectories = sampler.sample_trajectories(
        env,
        n=5,
        save_logprobs=True,
        save_estimator_outputs=True,
    )
    #  trajectories = sampler.sample_trajectories(env, n_trajectories=10)  # TODO - why is this duplicated?

    states = env.reset(batch_shape=5, random=True)
    bw_sampler = Sampler(estimator=pb_estimator)
    bw_trajectories = bw_sampler.sample_trajectories(
        env, save_logprobs=True, states=states
    )

    return trajectories, bw_trajectories, pf_estimator, pb_estimator


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("preprocessor_name", ["KHot", "OneHot", "Identity"])
@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8])
@pytest.mark.parametrize("n_components_s0", [1, 2, 5])
@pytest.mark.parametrize("n_components", [1, 2, 5])
def test_trajectory_sampling(
    env_name: str,
    preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"],
    delta: float,
    n_components_s0: int,
    n_components: int,
):
    if env_name == "HyperGrid":
        if delta != 0.1 or n_components_s0 != 1 or n_components != 1:
            pytest.skip("Useless tests")
    elif env_name == "DiscreteEBM":
        if (
            preprocessor_name != "Identity"
            or delta != 0.1
            or n_components_s0 != 1
            or n_components != 1
        ):
            pytest.skip("Useless tests")
    elif env_name == "Box":
        if preprocessor_name != "Identity":
            pytest.skip("Useless tests")
    else:
        raise ValueError("Unknown environment name")

    _ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,
        delta,
        n_components_s0,
        n_components,
    )


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_trajectories_getitem(env_name: str):
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


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_trajectories_extend(env_name: str):
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


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_sub_sampling(env_name: str):
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
def test_reverse_backward_trajectories(env_name: str):
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
    try:
        _ = backward_trajectories.reverse_backward_trajectories(
            debug=True  # <--- TRIGGER THE COMPARISON
        )
    except Exception as e:
        raise ValueError(
            f"Error while testing Trajectories.reverse_backward_trajectories in {env_name}"
        ) from e


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_local_search_for_loop_equivalence(env_name):
    """
    Ensures that the vectorized `LocalSearchSampler.local_search` matches
    the for-loop approach by toggling `debug=True`.
    """
    # Build environment
    is_discrete = env_name in ["HyperGrid", "DiscreteEBM"]
    if is_discrete:
        if env_name == "HyperGrid":
            env = HyperGrid(ndim=2, height=5, preprocessor_name="KHot")
        elif env_name == "DiscreteEBM":
            env = DiscreteEBM(ndim=5)
        else:
            raise ValueError("Unknown environment name")

        # Build pf & pb
        pf_module = MLP(env.preprocessor.output_dim, env.n_actions)
        pb_module = MLP(env.preprocessor.output_dim, env.n_actions - 1)
        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=env.preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
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


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_to_transition(env_name: str):
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

    try:
        _ = trajectories.to_transitions()
        bwd_trajectories = Trajectories.reverse_backward_trajectories(bwd_trajectories)
        # evaluate with pf_estimator
        backward_traj_pfs = get_trajectory_pfs(
            pf=pf_estimator, trajectories=bwd_trajectories
        )
        bwd_trajectories.log_probs = backward_traj_pfs
        _ = bwd_trajectories.to_transitions()
    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(
    env_name: str,
    objects: Literal["trajectories", "transitions"],
):
    """Test that the replay buffer works correctly with different types of objects."""
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=4)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=8)
    elif env_name == "Box":
        env = Box(delta=0.1)
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
                trajectories.when_is_done != trajectories.max_length
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
    env = HyperGrid(2, 4)
    states = torch.tensor([[0, 0], [0, 1], [0, 2], [-1, -1]])
    actions = torch.tensor([1, 1, 2])
    trajs = states_actions_tns_to_traj(states, actions, env)

    # Test that we can add the trajectories to a replay buffer
    replay_buffer = ReplayBuffer(env, capacity=10)
    replay_buffer.add(trajs)


# ------ GRAPH TESTS ------

# TODO: This test fails randomly. it should not rely on a custom GraphActionNet.
# def test_graph_building():
#     feature_dim = 8
#     env = GraphBuilding(
#         feature_dim=feature_dim, state_evaluator=lambda s: torch.zeros(s.batch_shape)
#     )

#     module = GraphActionNet(feature_dim)
#     pf_estimator = GraphActionPolicyEstimator(module=module)

#     sampler = Sampler(estimator=pf_estimator)
#     trajectories = sampler.sample_trajectories(
#         env,
#         n=7,
#         save_logprobs=True,
#         save_estimator_outputs=False,
#     )

#     assert len(trajectories) == 7


# class GraphActionNet(nn.Module):
#     def __init__(self, feature_dim: int):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.action_type_conv = GCNConv(feature_dim, 3)
#         self.features_conv = GCNConv(feature_dim, feature_dim)
#         self.edge_index_conv = GCNConv(feature_dim, 8)

#     def forward(self, states: GraphStates) -> TensorDict:
#         node_feature = states.tensor.x.reshape(-1, self.feature_dim)

#         if states.tensor.x.shape[0] == 0:
#             action_type = torch.zeros((len(states), 3))
#             action_type[:, GraphActionType.ADD_NODE] = 1
#             features = torch.zeros((len(states), self.feature_dim))
#         else:
#             action_type = self.action_type_conv(node_feature, states.tensor.edge_index)
#             action_type = action_type.reshape(
#                 len(states), -1, action_type.shape[-1]
#             ).mean(dim=1)
#             features = self.features_conv(node_feature, states.tensor.edge_index)
#             features = features.reshape(len(states), -1, features.shape[-1]).mean(dim=1)

#         edge_index = self.edge_index_conv(node_feature, states.tensor.edge_index)
#         edge_index = torch.einsum("nf,mf->nm", edge_index, edge_index)
#         edge_index = edge_index[None].repeat(len(states), 1, 1)

#         return TensorDict(
#             {
#                 "action_type": action_type,
#                 "features": features,
#                 "edge_index": edge_index.reshape(
#                     states.batch_shape + edge_index.shape[1:]
#                 ),
#             },
#             batch_size=states.batch_shape,
#         )
