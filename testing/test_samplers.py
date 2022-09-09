from typing import Literal

import pytest
import torch

from gfn.containers import ReplayBuffer, Trajectories
from gfn.envs import HyperGrid
from gfn.estimators import LogitPFEstimator
from gfn.modules import NeuralNet
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers import TrajectoriesSampler, TransitionsSampler
from gfn.samplers.actions_samplers import (
    FixedActionsSampler,
    LogitPFActionsSampler,
    UniformActionsSampler,
)


@pytest.mark.parametrize("height", [4, 5])
def test_hypergrid_trajectory_sampling(height: int, human_print=False) -> Trajectories:
    if human_print:
        print("---Trying Forward sampling of trajectories---")
    env = HyperGrid(ndim=2, height=height)

    if human_print:
        print("Trying the Uniform Action Sample with sf_temperature")
    actions_sampler = UniformActionsSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    if human_print:
        print(trajectories)

    if human_print:
        print("\nTrying the Fixed Actions Sampler: ")
    actions_sampler = FixedActionsSampler(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    if human_print:
        print(trajectories)

    if human_print:
        print("\nTrying the LogitPFActionSampler: ")
    preprocessors = [
        IdentityPreprocessor(env=env),
        OneHotPreprocessor(env=env),
        KHotPreprocessor(env=env),
    ]
    modules = [
        NeuralNet(
            input_dim=preprocessor.output_dim, hidden_dim=12, output_dim=env.n_actions
        )
        for preprocessor in preprocessors
    ]
    logit_pf_estimators = [
        LogitPFEstimator(preprocessor=preprocessor, module=module)
        for (preprocessor, module) in zip(preprocessors, modules)
    ]

    logit_pf_actions_samplers = [
        LogitPFActionsSampler(estimator=logit_pf_estimator)
        for logit_pf_estimator in logit_pf_estimators
    ]

    trajectories_samplers = [
        TrajectoriesSampler(env, logit_pf_actions_sampler)
        for logit_pf_actions_sampler in logit_pf_actions_samplers
    ]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        if human_print:
            print(
                "\n",
                i,
                ": Trying the LogitPFActionSampler with preprocessor {}".format(
                    preprocessors[i]
                ),
            )
        trajectories = trajectories_sampler.sample_trajectories(n_trajectories=10)
        if human_print:
            print(trajectories)

    if human_print:
        print("\n\n---Trying Backward sampling of trajectories---")

    if human_print:
        print("\n\n---Making Sure Last states are computed correctly---")
    actions_sampler = UniformActionsSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    if human_print:
        print(trajectories)
    return trajectories


@pytest.mark.parametrize("height", [4, 5])
def test_trajectories_getitem(height: int):
    trajectories = test_hypergrid_trajectory_sampling(height)
    print(f"There are {trajectories.n_trajectories} original trajectories")
    print(trajectories)
    print(trajectories[0])
    print(trajectories[[1, 0]])
    print(trajectories[torch.tensor([1, 2], dtype=torch.long)])


@pytest.mark.parametrize("height", [4, 5])
def test_trajectories_extend(height: int):
    trajectories = test_hypergrid_trajectory_sampling(height)
    print(
        f"There are {trajectories.n_trajectories} original trajectories. To which we will add the two first trajectories"
    )
    trajectories.extend(trajectories[[1, 0]])
    print(trajectories)


@pytest.mark.parametrize("height", [4, 5])
def test_sub_sampling(height: int):
    trajectories = test_hypergrid_trajectory_sampling(height)
    print(
        f"There are {trajectories.n_trajectories} original trajectories, from which we will sample 2"
    )
    print(trajectories)
    sampled_trajectories = trajectories.sample(n_trajectories=2)
    print("The two sampled trajectories are:")
    print(sampled_trajectories)


@pytest.mark.parametrize("height", [4, 5])
def test_hypergrid_transition_sampling(height: int):
    env = HyperGrid(ndim=2, height=height)

    print("---Trying Forward sampling of trajectories---")

    print("Trying the Uniform Action Sampler")
    actions_sampler = UniformActionsSampler()
    transitions_sampler = TransitionsSampler(env, actions_sampler)
    transitions = transitions_sampler.sample_transitions(n_transitions=5)
    print(transitions)
    transitions = transitions_sampler.sample_transitions(states=transitions.next_states)
    print(transitions)

    print("Trying the Fixed Actions Sampler")
    actions_sampler = FixedActionsSampler(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    transitions_sampler = TransitionsSampler(env, actions_sampler)
    transitions = transitions_sampler.sample_transitions(n_transitions=5)
    print(transitions)

    transitions = transitions_sampler.sample_transitions(states=transitions.next_states)
    print(transitions)
    return transitions


@pytest.mark.parametrize("height", [4, 5])
@pytest.mark.parametrize("objects", ["trajectories", "transitions"])
def test_replay_buffer(height: int, objects: Literal["trajectories", "transitions"]):
    env = HyperGrid(ndim=2, height=height)
    replay_buffer = ReplayBuffer(env, capacity=10, objects=objects)
    print(f"After initialization, the replay buffer is {replay_buffer} ")
    if objects == "trajectories":
        training_objects = test_hypergrid_trajectory_sampling(height)
        replay_buffer.add(
            training_objects[
                training_objects.when_is_done != training_objects.max_length
            ]
        )
    else:
        training_objects = test_hypergrid_transition_sampling(height)
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


test_hypergrid_trajectory_sampling(4)
