from gfn.containers import Trajectories
from gfn.containers.base import Container
from gfn.gflownet import FMGFlowNet, TBGFlowNet
from gfn.gym import Box
from gfn.gym.helpers.box_utils import (
    BoxPBEstimator,
    BoxPBNeuralNet,
    BoxPFEstimator,
    BoxPFNeuralNet,
)
from gfn.modules import DiscretePolicyEstimator
from gfn.states import States


def test_trajectory_based_gflownet_generic():
    pf_module = BoxPFNeuralNet(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    pb_module = BoxPBNeuralNet(
        hidden_dim=32, n_hidden_layers=2, n_components=1, torso=pf_module.torso
    )

    env = Box()

    pf_estimator = BoxPFEstimator(
        env=env, module=pf_module, n_components=1, n_components_s0=1
    )
    pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    mock_trajectories = Trajectories(env)

    result = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is of type `Trajectories`
    assert isinstance(
        result, Container
    ), f"Expected type Container, but got {type(result)}"
    assert isinstance(
        result, Trajectories
    ), f"Expected type Trajectories, but got {type(result)}"


def test_flow_matching_gflownet_generic():
    env = Box()
    module = BoxPFNeuralNet(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    estimator = DiscretePolicyEstimator(env, module, True)
    gflownet = FMGFlowNet(estimator)
    mock_trajectories = Trajectories(env)
    states_tuple = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is a tuple of `States`
    assert isinstance(
        states_tuple, tuple
    ), f"Expected type tuple, but got {type(states_tuple)}"

    assert isinstance(
        states_tuple[0], States
    ), f"Expected type States for first element, but got {type(states_tuple[0])}"
    assert isinstance(
        states_tuple[1], States
    ), f"Expected type States for second element, but got {type(states_tuple[1])}"
