from copy import deepcopy

import pytest
import torch

from gfn.actions import Actions, GraphActions


class ContinuousActions(Actions):
    action_shape = (10,)
    dummy_action = torch.zeros(10)
    exit_action = torch.ones(10)


@pytest.fixture
def continuous_action():
    return ContinuousActions(tensor=torch.arange(0, 10))


@pytest.fixture
def graph_action():
    return GraphActions(torch.zeros((1, 4)))


@pytest.mark.parametrize("action_fixture", ["continuous_action", "graph_action"])
def test_continuous_action(action_fixture, request):
    action = request.getfixturevalue(action_fixture)
    BATCH = 5

    exit_actions = action.make_exit_actions((BATCH,), action.device)
    assert torch.all(exit_actions.is_exit == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(exit_actions.is_dummy == torch.zeros(BATCH, dtype=torch.bool))

    dummy_actions = action.make_dummy_actions((BATCH,), action.device)
    assert torch.all(dummy_actions.is_dummy == torch.ones(BATCH, dtype=torch.bool))
    assert torch.all(dummy_actions.is_exit == torch.zeros(BATCH, dtype=torch.bool))

    # Test stack
    stacked_actions = action.stack([exit_actions, dummy_actions])
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
