from gfn.containers import StatePairs, Trajectories
from gfn.containers.base import Container
from gfn.gflownet import FMGFlowNet, TBGFlowNet
from gfn.gym import Box, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.modules import DiscretePolicyEstimator
from gfn.states import DiscreteStates
from gfn.utils.modules import MLP


def test_trajectory_based_gflownet_generic():
    pf_module = BoxPFMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    pb_module = BoxPBMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, trunk=pf_module.trunk
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
    env = HyperGrid(ndim=2, preprocessor_name="KHot")
    module = MLP(input_dim=env.preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module, n_actions=env.n_actions, preprocessor=env.preprocessor
    )
    gflownet = FMGFlowNet(estimator)
    mock_trajectories = Trajectories(env)
    states_pairs = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is a StatePairs[DiscreteStates]
    assert isinstance(states_pairs, StatePairs)
    assert isinstance(states_pairs.intermediary_states, DiscreteStates)
    assert isinstance(states_pairs.terminating_states, DiscreteStates)


def test_pytorch_inheritance():
    pf_module = BoxPFMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, n_components_s0=1
    )
    pb_module = BoxPBMLP(
        hidden_dim=32, n_hidden_layers=2, n_components=1, trunk=pf_module.trunk
    )

    env = Box()

    pf_estimator = BoxPFEstimator(
        env=env, module=pf_module, n_components=1, n_components_s0=1
    )
    pb_estimator = BoxPBEstimator(env=env, module=pb_module, n_components=1)

    tbgflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator)
    assert hasattr(
        tbgflownet.parameters(), "__iter__"
    ), "Expected gflownet to have iterable parameters() method inherited from nn.Module"
    assert hasattr(
        tbgflownet.state_dict(), "__dict__"
    ), "Expected gflownet to have indexable state_dict() method inherited from nn.Module"

    estimator = DiscretePolicyEstimator(
        pf_module, n_actions=2, preprocessor=env.preprocessor
    )
    fmgflownet = FMGFlowNet(estimator)
    assert hasattr(
        fmgflownet.parameters(), "__iter__"
    ), "Expected gflownet to have iterable parameters() method inherited from nn.Module"
    assert hasattr(
        fmgflownet.state_dict(), "__dict__"
    ), "Expected gflownet to have indexable state_dict() method inherited from nn.Module"
