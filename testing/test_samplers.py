from typing import Literal

import pytest
import torch

from gfn.containers import Trajectories
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.envs import DiscreteEBMEnv, HyperGrid
from gfn.estimators import LogitPBEstimator, LogitPFEstimator
from gfn.samplers import TrajectoriesSampler
from gfn.samplers.actions_samplers import (
    BackwardDiscreteActionsSampler,
    DiscreteActionsSampler,
)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("height", [4, 5])
@pytest.mark.parametrize("preprocessor_name", ["KHot", "OneHot", "Identity"])
def test_trajectory_sampling(
    env_name: str,
    height: int,
    preprocessor_name: str,
    human_print=False,
) -> Trajectories:
    if human_print:
        print("---Trying Forward sampling of trajectories---")
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=height, preprocessor_name=preprocessor_name)
    elif env_name == "DiscreteEBM":
        if preprocessor_name != "Identity" or height != 4:
            pytest.skip("Useless tests")
        env = DiscreteEBMEnv(ndim=8)
    else:
        raise ValueError("Unknown environment name")

    actions_sampler = DiscreteActionsSampler(
        LogitPFEstimator(env=env, module_name="NeuralNet")
    )

    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    if human_print:
        print(trajectories)

    if human_print:
        print("\nTrying the LogitPFActionSampler: ")

    logit_pf_estimator = LogitPFEstimator(env, module_name="NeuralNet")

    logit_pf_actions_sampler = DiscreteActionsSampler(estimator=logit_pf_estimator)

    trajectories_sampler = TrajectoriesSampler(
        env,
        actions_sampler=logit_pf_actions_sampler,
    )

    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=10)
    if human_print:
        print(trajectories)

    if human_print:
        print("\n\n---Trying Backward sampling of trajectories---")

    states = env.reset(batch_shape=20, random=True)

    logit_pb_estimator = LogitPBEstimator(env=env, module_name="NeuralNet")

    logit_pb_actions_sampler = BackwardDiscreteActionsSampler(
        estimator=logit_pb_estimator
    )

    bw_trajectories_sampler = TrajectoriesSampler(env, logit_pb_actions_sampler)

    states = env.reset(batch_shape=5, random=True)
    bw_trajectories = bw_trajectories_sampler.sample_trajectories(states)
    if human_print:
        print(bw_trajectories)

    if human_print:
        print("\n\n---Making Sure Last states are computed correctly---")

    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    if human_print:
        print(trajectories)
    return trajectories


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("height", [4, 5])
def test_trajectories_getitem(env_name: str, height: int):
    trajectories = test_trajectory_sampling(
        env_name,
        height,
        preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
    )
    print(f"There are {trajectories.n_trajectories} original trajectories")
    print(trajectories)
    print(trajectories[0])
    print(trajectories[[1, 0]])
    print(trajectories[torch.tensor([1, 2], dtype=torch.long)])


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("height", [4, 5])
def test_trajectories_extend(env_name: str, height: int):
    trajectories = test_trajectory_sampling(
        env_name,
        height,
        preprocessor_name="KHot" if env_name == "HyperGrid" else "Identity",
    )
    print(
        f"There are {trajectories.n_trajectories} original trajectories. To which we will add the two first trajectories"
    )
    trajectories.extend(trajectories[[1, 0]])
    print(trajectories)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("height", [4, 5])
def test_sub_sampling(env_name: str, height: int):
    trajectories = test_trajectory_sampling(
        env_name,
        height,
        preprocessor_name="Identity",
    )
    print(
        f"There are {trajectories.n_trajectories} original trajectories, from which we will sample 2"
    )
    print(trajectories)
    sampled_trajectories = trajectories.sample(n_samples=2)
    print("The two sampled trajectories are:")
    print(sampled_trajectories)


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
@pytest.mark.parametrize("height", [4, 5])
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(
    env_name: str,
    height: int,
    objects: Literal["trajectories", "transitions"],
):
    if env_name == "HyperGrid":
        env = HyperGrid(ndim=2, height=height)
    elif env_name == "DiscreteEBM":
        if height != 4:
            pytest.skip("Useless tests")
        env = DiscreteEBMEnv(ndim=8)
    else:
        raise ValueError("Unknown environment name")
    replay_buffer = ReplayBuffer(env, capacity=10, objects_type=objects)
    print(f"After initialization, the replay buffer is {replay_buffer} ")
    training_objects = test_trajectory_sampling(
        env_name,
        height,
        preprocessor_name="Identity",
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


def test_extend_trajectories_on_cuda():
    import os
    import sys

    sys.path.insert(0, os.path.abspath("__file__" + "/../"))

    from src.gfn.containers.trajectories import Trajectories as Traj

    torch.manual_seed(0)

    env = HyperGrid(ndim=4, height=8, R0=0.01, device_str="cuda")
    sampler = TrajectoriesSampler(
        env=env,
        actions_sampler=DiscreteActionsSampler(
            estimator=LogitPFEstimator(env=env, module_name="NeuralNet"),
        ),
    )

    trajectories_1 = sampler.sample(n_trajectories=10)
    trajectories_2 = sampler.sample(n_trajectories=10)

    trajectories_1 = Traj(
        env=sampler.env,
        states=trajectories_1.states,
        actions=trajectories_1.actions,
        when_is_done=trajectories_1.when_is_done,
        is_backward=sampler.is_backward,
        log_rewards=trajectories_1.log_rewards,
        log_probs=trajectories_1.log_probs,
    )
    trajectories_2 = Traj(
        env=sampler.env,
        states=trajectories_2.states,
        actions=trajectories_2.actions,
        when_is_done=trajectories_2.when_is_done,
        is_backward=sampler.is_backward,
        log_rewards=trajectories_2.log_rewards,
        log_probs=trajectories_2.log_probs,
    )

    trajectories_1.extend(trajectories_2)
