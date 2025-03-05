from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from gfn.actions import Actions, GraphActions


class ContinuousActions(Actions):
    action_shape = (10,)
    dummy_action = torch.zeros(10)
    exit_action = torch.ones(10)


class TestGraphActions(GraphActions):
    features_dim = 10


@pytest.fixture
def continuous_action():
    return ContinuousActions(tensor=torch.arange(0, 10))


@pytest.fixture
def graph_action():
    return TestGraphActions(
        tensor=TensorDict(
            {
                "action_type": torch.zeros((1,), dtype=torch.float32),
                "features": torch.zeros((1, 10), dtype=torch.float32),
            },
            device="cpu",
        )
    )


def test_continuous_action(continuous_action):
    BATCH = 5

    exit_actions = continuous_action.make_exit_actions((BATCH,))
    assert torch.all(
        exit_actions.tensor == continuous_action.exit_action.repeat(BATCH, 1)
    )
    assert torch.all(exit_actions.is_exit == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(exit_actions.is_dummy == torch.zeros(BATCH, dtype=torch.bool))

    dummy_actions = continuous_action.make_dummy_actions((BATCH,))
    assert torch.all(
        dummy_actions.tensor == continuous_action.dummy_action.repeat(BATCH, 1)
    )
    assert torch.all(dummy_actions.is_dummy == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(dummy_actions.is_exit == torch.zeros(BATCH, dtype=torch.bool))

    # Test stack
    stacked_actions = continuous_action.stack([exit_actions, dummy_actions])
    assert stacked_actions.batch_shape == (2, BATCH)
    assert torch.all(
        stacked_actions.tensor
        == torch.stack([exit_actions.tensor, dummy_actions.tensor], dim=0)
    )
    is_exit_stacked = torch.stack([exit_actions.is_exit, dummy_actions.is_exit], dim=0)
    assert torch.all(stacked_actions.is_exit == is_exit_stacked)
    assert stacked_actions[0, 1].is_exit
    stacked_actions[0, 1] = stacked_actions[1, 1]
    is_exit_stacked[0, 1] = False
    assert torch.all(stacked_actions.is_exit == is_exit_stacked)

    # Test extend
    extended_actions = deepcopy(exit_actions)
    extended_actions.extend(dummy_actions)
    assert extended_actions.batch_shape == (BATCH * 2,)
    assert torch.all(
        extended_actions.tensor
        == torch.cat([exit_actions.tensor, dummy_actions.tensor], dim=0)
    )
    is_exit_extended = torch.cat([exit_actions.is_exit, dummy_actions.is_exit], dim=0)
    assert torch.all(extended_actions.is_exit == is_exit_extended)
    assert extended_actions[0].is_exit and extended_actions[BATCH].is_dummy
    extended_actions[0] = extended_actions[BATCH]
    is_exit_extended[0] = False
    assert torch.all(extended_actions.is_exit == is_exit_extended)


def test_graph_action(graph_action):
    BATCH = 5

    exit_actions = graph_action.make_exit_actions((BATCH,))
    assert torch.all(exit_actions.is_exit == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(exit_actions.is_dummy == torch.zeros(BATCH, dtype=torch.bool))
    dummy_actions = graph_action.make_dummy_actions((BATCH,))
    assert torch.all(dummy_actions.is_dummy == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(dummy_actions.is_exit == torch.zeros(BATCH, dtype=torch.bool))

    # Test stack
    stacked_actions = graph_action.stack([exit_actions, dummy_actions])
    assert stacked_actions.batch_shape == (2, BATCH)
    manually_stacked_tensor = torch.stack(
        [exit_actions.tensor, dummy_actions.tensor], dim=0
    )
    assert torch.all(
        stacked_actions.tensor["action_type"]
        == manually_stacked_tensor["action_type"]  # pyright: ignore
    )
    assert torch.all(
        stacked_actions.tensor["features"]
        == manually_stacked_tensor["features"]  # pyright: ignore
    )
    assert torch.all(
        stacked_actions.tensor["edge_index"]
        == manually_stacked_tensor["edge_index"]  # pyright: ignore
    )
    is_exit_stacked = torch.stack([exit_actions.is_exit, dummy_actions.is_exit], dim=0)
    assert torch.all(stacked_actions.is_exit == is_exit_stacked)
    assert stacked_actions[0, 1].is_exit
    stacked_actions[0, 1] = stacked_actions[1, 1]
    is_exit_stacked[0, 1] = False
    assert torch.all(stacked_actions.is_exit == is_exit_stacked)

    # Test extend
    extended_actions = deepcopy(exit_actions)
    extended_actions.extend(dummy_actions)
    assert extended_actions.batch_shape == (BATCH * 2,)
    manually_extended_tensor = torch.cat(
        [exit_actions.tensor, dummy_actions.tensor], dim=0
    )
    assert torch.all(
        extended_actions.tensor["action_type"]
        == manually_extended_tensor["action_type"]  # pyright: ignore
    )
    assert torch.all(
        extended_actions.tensor["features"]
        == manually_extended_tensor["features"]  # pyright: ignore
    )
    assert torch.all(
        extended_actions.tensor["edge_index"]
        == manually_extended_tensor["edge_index"]  # pyright: ignore
    )
    is_exit_extended = torch.cat([exit_actions.is_exit, dummy_actions.is_exit], dim=0)
    assert torch.all(extended_actions.is_exit == is_exit_extended)
    assert extended_actions[0].is_exit and extended_actions[BATCH].is_dummy
    extended_actions[0] = extended_actions[BATCH]
    is_exit_extended[0] = False
    assert torch.all(extended_actions.is_exit == is_exit_extended)
