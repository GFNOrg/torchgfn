import pytest
import torch

from gfn.containers import StatesContainer, Trajectories
from gfn.containers.base import Container
from gfn.estimators import DiscretePolicyEstimator
from gfn.gflownet import FMGFlowNet, TBGFlowNet
from gfn.gflownet.base import loss_reduce
from gfn.gym import Box, HyperGrid
from gfn.gym.helpers.box_utils import BoxPBEstimator, BoxPBMLP, BoxPFEstimator, BoxPFMLP
from gfn.preprocessors import KHotPreprocessor
from gfn.states import DiscreteStates
from gfn.utils.handlers import (
    has_conditions_exception_handler,
    no_conditions_exception_handler,
)
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
    env = HyperGrid(ndim=2)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module, n_actions=env.n_actions, preprocessor=preprocessor
    )
    gflownet = FMGFlowNet(estimator)
    mock_trajectories = Trajectories(env)
    states_pairs = gflownet.to_training_samples(mock_trajectories)

    # Assert that the result is a StatesContainer[DiscreteStates]
    assert isinstance(states_pairs, StatesContainer)
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

    estimator = DiscretePolicyEstimator(pf_module, n_actions=2)
    fmgflownet = FMGFlowNet(estimator)
    assert hasattr(
        fmgflownet.parameters(), "__iter__"
    ), "Expected gflownet to have iterable parameters() method inherited from nn.Module"
    assert hasattr(
        fmgflownet.state_dict(), "__dict__"
    ), "Expected gflownet to have indexable state_dict() method inherited from nn.Module"


@pytest.mark.parametrize("seed", [0, 12, 47, 67])
def test_flow_matching_vectorized_matches_original(seed):
    torch.manual_seed(seed)
    env = HyperGrid(ndim=2)
    preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)
    module = MLP(input_dim=preprocessor.output_dim, output_dim=env.n_actions)
    estimator = DiscretePolicyEstimator(
        module, n_actions=env.n_actions, preprocessor=preprocessor
    )
    gflownet = FMGFlowNet(estimator)

    trajectories = gflownet.sample_trajectories(env, n=6)
    states_container = gflownet.to_training_samples(trajectories)
    states = states_container.intermediary_states
    conditions = states_container.intermediary_conditions

    if len(states) == 0:
        # If the sample produced only terminal states, resample with more trajectories.
        trajectories = gflownet.sample_trajectories(env, n=12)
        states_container = gflownet.to_training_samples(trajectories)
        states = states_container.intermediary_states
        conditions = states_container.intermediary_conditions

    assert len(states) > 0

    def flow_matching_loss_original(
        self, env, states, conditions, reduction: str = "mean"
    ):
        if len(states) == 0:
            return torch.tensor(0.0, device=states.device)
        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)
        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.get_default_dtype()
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.get_default_dtype()
        )
        for action_idx in range(env.n_actions - 1):
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]
            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            ).unsqueeze(-1)
            backward_actions = env.actions_from_tensor(backward_actions)
            valid_backward_states_parents = env._backward_step(
                valid_backward_states, backward_actions
            )
            if conditions is not None:
                valid_backward_conditions = conditions[valid_backward_mask]
                valid_forward_conditions = conditions[valid_forward_mask]
                with has_conditions_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                        valid_backward_conditions,
                    )[:, action_idx]
                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                        valid_forward_conditions,
                    )[:, action_idx]
            else:
                with no_conditions_exception_handler("logF", self.logF):
                    incoming_log_flows[valid_backward_mask, action_idx] = self.logF(
                        valid_backward_states_parents,
                    )[:, action_idx]
                    outgoing_log_flows[valid_forward_mask, action_idx] = self.logF(
                        valid_forward_states,
                    )[:, action_idx]
        valid_forward_mask = states.forward_masks[:, -1]
        if conditions is not None:
            with has_conditions_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                    conditions[valid_forward_mask],
                )[:, -1]
        else:
            with no_conditions_exception_handler("logF", self.logF):
                outgoing_log_flows[valid_forward_mask, -1] = self.logF(
                    states[valid_forward_mask],
                )[:, -1]
        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)
        scores = (log_incoming_flows - log_outgoing_flows).pow(2)
        return loss_reduce(scores, reduction)

    loss_original = flow_matching_loss_original(
        gflownet, env, states, conditions, reduction="mean"
    )
    loss_vectorized = gflownet.flow_matching_loss(
        env, states, conditions, reduction="mean"
    )

    torch.testing.assert_close(loss_vectorized, loss_original)
