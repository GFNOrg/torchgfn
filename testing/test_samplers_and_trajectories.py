from typing import Literal, Tuple

import pytest

from gfn.containers import Trajectories
from gfn.containers.replay_buffer import ReplayBuffer
from gfn.gym import Box, DiscreteEBM, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.modules import DiscretePolicyEstimator, GFNModule
from gfn.samplers import LocalSearchSampler, Sampler
from gfn.utils.modules import MLP


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


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_reverse_backward_trajectories(env_name: str):
    """
    Ensures that the vectorized `Trajectories.reverse_backward_trajectories`
    matches the for-loop approach by toggling `debug=True`.

    Note that `Trajectories.reverse_backward_trajectories` is not compatible with
    environment with continuous states (e.g., Box).
    """
    _, backward_trajectories, *_ = trajectory_sampling_with_return(
        env_name,
        preprocessor_name="Identity",
        delta=0.1,
        n_components=1,
        n_components_s0=1,
    )
    try:
        _ = Trajectories.reverse_backward_trajectories(
            backward_trajectories, debug=True  # <--- TRIGGER THE COMPARISON
        )
    except Exception as e:
        raise ValueError(
            f"Error while testing Trajectories.reverse_backward_trajectories in {env_name}"
        ) from e


@pytest.mark.parametrize("env_name", ["HyperGrid", "DiscreteEBM"])
def test_local_search_for_loop_equivalence(env_name):
    """
    Ensures that the vectorized `LocalSearchSampler.local_search` matches
    the for-loop approach by toggling `debug=True`.

    Note that this is not supported for environment with continuous state
    space (e.g., Box), since `Trajectories.reverse_backward_trajectories`
    is not compatible with continuous states.
    """
    # Build environment
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

    sampler = LocalSearchSampler(pf_estimator=pf_estimator, pb_estimator=pb_estimator)

    # Initial forward-sampler call
    trajectories = sampler.sample_trajectories(env, n=3, save_logprobs=True)

    # Now run local_search in debug mode so that for-loop logic is compared
    # to the vectorized logic.
    # If there’s any mismatch, local_search() will raise AssertionError
    try:
        new_trajectories, is_updated = sampler.local_search(
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
