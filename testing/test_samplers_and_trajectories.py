from typing import Literal

import pytest
import torch

from gfn.containers import Trajectories
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.envs import DiscreteEBMEnv, HyperGrid, BoxEnv
from gfn.samplers import ActionsSampler, TrajectoriesSampler
from gfn.utils import DiscretePBEstimator, DiscretePFEstimator, NeuralNet
from gfn.examples.box_utils import (
    BoxPFEStimator,
    BoxPFNeuralNet,
    BoxPBEstimator,
    BoxPBNeuralNet,
)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("preprocessor_name", ["KHot", "OneHot", "Identity"])
@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8])
def test_trajectory_sampling(
    env_name: str, preprocessor_name: str, delta: float
) -> Trajectories:
    if env_name == "HyperGrid":
        if delta != 0.1:
            pytest.skip("Useless tests")
        env = HyperGrid(ndim=2, height=8, preprocessor_name=preprocessor_name)
    elif env_name == "DiscreteEBM":
        if preprocessor_name != "Identity" or delta != 0.1:
            pytest.skip("Useless tests")
        env = DiscreteEBMEnv(ndim=8)
    elif env_name == "Box":
        if preprocessor_name != "Identity":
            pytest.skip("Useless tests")
        env = BoxEnv(delta=delta)
    else:
        raise ValueError("Unknown environment name")

    if env_name == "Box":
        pf_module = BoxPFNeuralNet(
            hidden_dim=32, n_hidden_layers=2, n_components=3, n_components_s0=2
        )
        pb_module = BoxPBNeuralNet(
            hidden_dim=32,
            n_hidden_layers=2,
            n_components=3,
            torso=pf_module.torso,
        )
        pf_estimator = BoxPFEStimator(
            env=env, module=pf_module, n_components=3, n_components_s0=2
        )
        pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=3)
    else:
        logit_pf_module = NeuralNet(
            input_dim=env.preprocessor.output_shape[0], output_dim=env.n_actions
        )
        logit_pb_module = NeuralNet(
            input_dim=env.preprocessor.output_shape[0], output_dim=env.n_actions - 1
        )
        pf_estimator = DiscretePFEstimator(env=env, module=logit_pf_module)
        pb_estimator = DiscretePBEstimator(env=env, module=logit_pb_module)

    actions_sampler = ActionsSampler(estimator=pf_estimator)

    trajectories_sampler = TrajectoriesSampler(actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)

    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=10)

    bw_actions_sampler = ActionsSampler(estimator=pb_estimator)

    bw_trajectories_sampler = TrajectoriesSampler(bw_actions_sampler, is_backward=True)

    states = env.reset(batch_shape=5, random=True)
    bw_trajectories = bw_trajectories_sampler.sample_trajectories(states)

    return trajectories, bw_trajectories


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_trajectories_getitem(env_name: str):
    trajectories, _ = test_trajectory_sampling(
        env_name,
        preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
        delta=0.1,
    )
    print(f"There are {trajectories.n_trajectories} original trajectories")
    print(trajectories)
    print(trajectories[0])
    print(trajectories[[1, 0]])
    print(trajectories[torch.tensor([1, 2], dtype=torch.long)])


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_trajectories_extend(env_name: str):
    trajectories, _ = test_trajectory_sampling(
        env_name,
        preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
        delta=0.1,
    )
    print(
        f"There are {trajectories.n_trajectories} original trajectories. To which we will add the two first trajectories"
    )
    trajectories.extend(trajectories[[1, 0]])
    print(trajectories)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
def test_sub_sampling(env_name: str):
    trajectories, _ = test_trajectory_sampling(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
    )
    print(
        f"There are {trajectories.n_trajectories} original trajectories, from which we will sample 2"
    )
    print(trajectories)
    sampled_trajectories = trajectories.sample(n_samples=2)
    print("The two sampled trajectories are:")
    print(sampled_trajectories)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM", "Box"])
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(
    env_name: str,
    objects: Literal["trajectories", "transitions"],
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=4)
    elif env_name == "DiscreteEBM":
        env = DiscreteEBMEnv(ndim=8)
    elif env_name == "Box":
        env = BoxEnv(delta=0.1)
    else:
        raise ValueError("Unknown environment name")
    replay_buffer = ReplayBuffer(env, capacity=10, objects_type=objects)
    print(f"After initialization, the replay buffer is {replay_buffer} ")
    training_objects, _ = test_trajectory_sampling(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
    )
    if objects == "trajectories":
        replay_buffer.add(
            training_objects[
                training_objects.when_is_done != training_objects.max_length
            ]
        )
    else:
        training_objects = training_objects.to_transitions()
        replay_buffer.add(training_objects)

    print(
        f"After adding {len(training_objects)} trajectories, the replay buffer is {replay_buffer} \n {replay_buffer.training_objects} "
    )
    replay_buffer.add(training_objects)
    print(
        f"After adding {len(training_objects)} trajectories, the replay buffer is {replay_buffer} \n {replay_buffer.training_objects}  "
    )
    replay_buffer.add(training_objects)
    print(
        f"After adding {len(training_objects)} trajectories, the replay buffer is {replay_buffer} \n {replay_buffer.training_objects}  "
    )
