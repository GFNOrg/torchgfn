import pytest
import torch

from gfn.containers.trajectories import Trajectories
from gfn.envs import HyperGrid
from gfn.estimators import LogitPBEstimator, LogitPFEstimator
from gfn.modules import NeuralNet
from gfn.preprocessors import IdentityPreprocessor, KHotPreprocessor, OneHotPreprocessor
from gfn.samplers import TrajectoriesSampler, TransitionsSampler
from gfn.samplers.actions_samplers import (
    FixedActionsSampler,
    LogitPBActionsSampler,
    LogitPFActionsSampler,
    UniformActionsSampler,
    UniformBackwardActionsSampler,
)


@pytest.mark.parametrize("height", [4, 5])
def test_hypergrid_trajectory_sampling(height: int) -> Trajectories:
    # TODO: make this deterministic by introducing a seed - PyTest fails sometimes here for no reason
    print("---Trying Forward sampling of trajectories---")
    env = HyperGrid(ndim=2, height=height)

    print("Trying the Uniform Action Sample with sf_temperature")
    actions_sampler = UniformActionsSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)

    print("\nTrying the Fixed Actions Sampler: ")
    actions_sampler = FixedActionsSampler(
        torch.tensor(
            [[0, 1, 2, 0], [1, 1, 1, 2], [2, 2, 2, 2], [1, 0, 1, 2], [1, 0, 2, 1]]
        )
    )
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)

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
        print(
            "\n",
            i,
            ": Trying the LogitPFActionSampler with preprocessor {}".format(
                preprocessors[i]
            ),
        )
        trajectories = trajectories_sampler.sample_trajectories(n_trajectories=10)
        print(trajectories)

    print("\n\n---Trying Backward sampling of trajectories---")
    states = env.reset(batch_shape=20, random_init=True)

    print(
        "\nTrying the Uniform Backward Action Sampler with one of the initial states being s_0"
    )
    actions_sampler = UniformBackwardActionsSampler()
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(states)
    print(trajectories)

    modules = [
        NeuralNet(
            input_dim=preprocessor.output_dim,
            hidden_dim=12,
            output_dim=env.n_actions - 1,
        )
        for preprocessor in preprocessors
    ]
    logit_pb_estimators = [
        LogitPBEstimator(preprocessor=preprocessor, module=module)
        for (preprocessor, module) in zip(preprocessors, modules)
    ]

    logit_pb_actions_samplers = [
        LogitPBActionsSampler(estimator=logit_pb_estimator)
        for logit_pb_estimator in logit_pb_estimators
    ]

    trajectories_samplers = [
        TrajectoriesSampler(env, logit_pb_actions_sampler)
        for logit_pb_actions_sampler in logit_pb_actions_samplers
    ]

    for i, trajectories_sampler in enumerate(trajectories_samplers):
        print(
            "\n",
            i,
            ": Trying the LogitPBActionSampler with preprocessor {}".format(
                preprocessors[i]
            ),
        )
        states = env.reset(batch_shape=5, random_init=True)
        trajectories = trajectories_samplers[i].sample_trajectories(states)
        print(trajectories)

    print("\n\n---Making Sure Last states are computed correctly---")
    actions_sampler = UniformActionsSampler(sf_temperature=2.0)
    trajectories_sampler = TrajectoriesSampler(env, actions_sampler)
    trajectories = trajectories_sampler.sample_trajectories(n_trajectories=5)
    print(trajectories)
    return trajectories


@pytest.mark.parametrize("height", [4, 5])
def test_trajectories_getitem_setitem(height: int):
    trajectories = test_hypergrid_trajectory_sampling(height)
    print(f"There are {trajectories.n_trajectories} original trajectories")
    print(trajectories)
    print(trajectories[0])
    print(trajectories[[1, 0]])
    print(trajectories[torch.tensor([1, 2], dtype=torch.long)])
    trajectories[2] = trajectories[1]
    print(trajectories)
    print(trajectories[[2, 0, 1]])


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


test_trajectories_extend(5)
