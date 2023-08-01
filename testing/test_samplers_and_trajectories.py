from typing import Literal

import pytest

from gfn.containers import Trajectories
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.helpers.box_utils import (
    BoxPBEstimator,
    BoxPBNeuralNet,
    BoxPFEstimator,
    BoxPFNeuralNet,
)
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils import NeuralNet


def trajectory_sampling_with_return(
    env_name: str,
    preprocessor_name: str,
    delta: float,
    n_components_s0: int,
    n_components: int,
) -> Trajectories:
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
    else:
        raise ValueError("Unknown environment name")

    if env_name == "Box":
        pf_module = BoxPFNeuralNet(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=n_components,
            n_components_s0=n_components_s0,
        )
        pb_module = BoxPBNeuralNet(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=n_components,
            torso=pf_module.torso,
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
        pf_module = NeuralNet(
            input_dim=env.preprocessor.output_dim, output_dim=env.n_actions
        )
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim, output_dim=env.n_actions - 1
        )
        pf_estimator = DiscretePolicyEstimator(env=env, module=pf_module, forward=True)
        pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)

    sampler = Sampler(estimator=pf_estimator)
    trajectories = sampler.sample_trajectories(n_trajectories=5)
    trajectories = sampler.sample_trajectories(n_trajectories=10)

    states = env.reset(batch_shape=5, random=True)
    bw_sampler = Sampler(estimator=pb_estimator, is_backward=True)
    bw_trajectories = bw_sampler.sample_trajectories(states)

    return trajectories, bw_trajectories, pf_estimator, pb_estimator


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("preprocessor_name", ["KHot", "OneHot", "Identity"])
@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8])
@pytest.mark.parametrize("n_components_s0", [1, 2, 5])
@pytest.mark.parametrize("n_components", [1, 2, 5])
def test_trajectory_sampling(
    env_name: str,
    preprocessor_name: str,
    delta: float,
    n_components_s0: int,
    n_components: int,
) -> Trajectories:
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

    (
        trajectories,
        bw_trajectories,
        pf_estimator,
        pb_estimator,
    ) = trajectory_sampling_with_return(
        env_name,
        preprocessor_name,
        delta,
        n_components_s0,
        n_components,
    )


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_trajectories_getitem(env_name: str):
    try:
        trajectories, *_ = trajectory_sampling_with_return(
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
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(
    env_name: str,
    objects: Literal["trajectories", "transitions"],
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=4)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBM(ndim=8)
    elif env_name == "Box":
        env = Box(delta=0.1)
    else:
        raise ValueError("Unknown environment name")
    replay_buffer = ReplayBuffer(env, capacity=10, objects_type=objects)
    training_objects, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    try:
        if objects == "trajectories":
            replay_buffer.add(
                training_objects[
                    training_objects.when_is_done != training_objects.max_length
                ]
            )
        else:
            training_objects = training_objects.to_transitions()
            replay_buffer.add(training_objects)

        replay_buffer.add(training_objects)
        replay_buffer.add(training_objects)
    except Exception as e:
        raise ValueError(f"Error while testing {env_name}") from e
