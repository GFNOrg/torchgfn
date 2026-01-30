import numpy as np
import pytest
import torch
from torch_geometric.data import Data as GeometricData

from gfn.actions import GraphActions, GraphActionType
from gfn.states import DiscreteStates, GraphStates, States


class MyGraphStates(GraphStates):
    num_node_classes = 10
    max_nodes = 5
    num_edge_classes = 10
    is_directed = True

    # Initial state: a graph with 2 nodes and 1 edge
    s0 = GeometricData(
        x=torch.tensor([[-1.0], [-2.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[0.5]]),
    )

    # Sink state: a graph with 2 nodes and 1 edge (different from s0)
    sf = GeometricData(
        x=torch.tensor([[-1.0]]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 1)),
    )


class SimpleDiscreteStates(DiscreteStates):
    state_shape = (2,)  # 2-dimensional state
    n_actions = 3  # 3 possible actions
    s0 = torch.tensor([0.0, 0.0])
    sf = torch.tensor([1.0, 1.0])

    def _compute_forward_masks(self) -> torch.Tensor:
        """All forward actions allowed."""
        return torch.ones(
            (*self.batch_shape, self.n_actions),
            dtype=torch.bool,
            device=self.device,
        )

    def _compute_backward_masks(self) -> torch.Tensor:
        """All backward actions allowed."""
        return torch.ones(
            (*self.batch_shape, self.n_actions - 1),
            dtype=torch.bool,
            device=self.device,
        )


class SimpleTensorStates(States):
    state_shape = (2,)  # 2-dimensional state
    s0 = torch.tensor([0.0, 0.0])
    sf = torch.tensor([1.0, 1.0])


@pytest.fixture
def datas():
    """Creates a list of 10 GeometricData objects"""
    datas = np.empty(10, dtype=object)
    for i in range(10):
        datas[i] = GeometricData(
            x=torch.tensor([[i], [i + 1]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[i]]),
        )
    return datas


@pytest.fixture
def simple_graph_state(datas):
    """Creates a simple graph state with 2 nodes and 1 edge"""
    return MyGraphStates(datas[:1])


@pytest.fixture
def empty_graph_state():
    """Creates an empty GraphStates object"""
    return MyGraphStates(np.empty(0, dtype=object))


@pytest.fixture
def conditional_graph_state(simple_graph_state):
    """Creates a simple graph state with conditions"""
    state = simple_graph_state.clone()
    state.conditions = torch.tensor([[1.0]])
    return state


@pytest.fixture
def simple_discrete_state():
    """Creates a simple discrete state with 3 possible actions"""
    # Create a single state tensor
    tensor = torch.tensor([[0.5, 0.5]])
    return SimpleDiscreteStates(tensor)


@pytest.fixture
def empty_discrete_state():
    """Creates an empty discrete state"""
    # Create an empty state tensor
    tensor = torch.zeros((0, 2))
    return SimpleDiscreteStates(tensor)


@pytest.fixture
def conditional_discrete_state(simple_discrete_state):
    """Creates a simple discrete state with conditions"""
    state = simple_discrete_state.clone()
    state.conditions = torch.tensor([[1.0]])
    return state


@pytest.fixture
def simple_tensor_state():
    """Creates a simple tensor state"""
    # Create a single state tensor
    tensor = torch.tensor([[0.5, 0.5]])
    return SimpleTensorStates(tensor)


@pytest.fixture
def empty_tensor_state():
    """Creates an empty tensor state"""
    # Create an empty state tensor
    tensor = torch.zeros((0, 2))
    return SimpleTensorStates(tensor)


@pytest.fixture
def conditional_tensor_state(simple_tensor_state):
    """Creates a simple tensor state with conditions"""
    state = simple_tensor_state.clone()
    state.conditions = torch.tensor([[1.0]])
    return state


def test_getitem_1d(datas):
    """Test indexing into GraphStates

    Make sure the behavior is consistent with that of a Tensor.__getitem__.
    """
    tsr = torch.tensor([1, 2, 3])
    states = MyGraphStates(datas[:3])

    # Get a single graph
    single_tsr = tsr[1]
    single_state = states[1]
    assert tuple(single_tsr.shape) == single_state.batch_shape == ()
    assert single_state.tensor.x.size(0) == 2
    assert torch.allclose(single_state.tensor.x, datas[1].x)

    # Get multiple graphs
    multi_tsr = tsr[[0, 2]]
    multi_state = states[[0, 2]]
    assert tuple(multi_tsr.shape) == multi_state.batch_shape == (2,)
    assert multi_state.tensor.x.size(0) == 4
    assert torch.allclose(multi_state.tensor.get_example(0).x, datas[0].x)
    assert torch.allclose(multi_state.tensor.get_example(1).x, datas[2].x)


def test_getitem_2d(datas):
    """Test indexing into GraphStates with 2D batch shape

    Make sure the behavior is consistent with that of a Tensor.__getitem__.
    """
    # Create a tensor with 4 elements for comparison
    tsr = torch.tensor([[1, 2], [3, 4]])

    # Create a batch with 2x2 graphs
    states = MyGraphStates(datas[:4].reshape(2, 2))

    # Get a single row
    tsr_row = tsr[0]
    batch_row = states[0]
    assert tuple(tsr_row.shape) == batch_row.batch_shape == (2,)
    assert batch_row.tensor.x.size(0) == 4  # 2 graphs * 2 nodes
    assert torch.allclose(batch_row.tensor.get_example(0).x, datas[0].x)
    assert torch.allclose(batch_row.tensor.get_example(1).x, datas[1].x)

    # Try again with slicing
    tsr_row2 = tsr[0, :]
    batch_row2 = states[0, :]
    assert tuple(tsr_row2.shape) == batch_row2.batch_shape == (2,)
    assert torch.equal(batch_row.tensor.x, batch_row2.tensor.x)

    # Get a single graph with 2D indexing
    single_tsr = tsr[1, 1]
    single_state = states[1, 1]
    assert tuple(single_tsr.shape) == single_state.batch_shape == ()
    assert single_state.tensor.x.size(0) == 2  # 1 graph * 2 nodes
    assert torch.allclose(single_state.tensor.x, datas[3].x)

    with pytest.raises(IndexError):
        states[2, 2]

    # We can't index on a Batch with 0-dimensional batch shape
    with pytest.raises(IndexError):
        single_state[0]


def test_setitem_1d(datas):
    """Test setting values in States"""
    states = MyGraphStates(datas[:3])
    new_states = MyGraphStates(datas[3:5])

    # Set the new graph in the first position
    states[0] = new_states[0]

    # Check that the first graph is now the new graph
    first_graph = states[0].tensor
    assert torch.equal(first_graph.x, datas[3].x)
    assert torch.equal(first_graph.edge_attr, datas[3].edge_attr)
    assert torch.equal(first_graph.edge_index, datas[3].edge_index)
    assert states.batch_shape == (3,)  # Batch shape should not change

    # Set the new graph in the second and third positions
    states[1:] = new_states

    # Check that the second and third graphs are now the new graph
    second_graph = states[1].tensor
    assert torch.equal(second_graph.x, datas[3].x)
    assert torch.equal(second_graph.edge_attr, datas[3].edge_attr)
    assert torch.equal(second_graph.edge_index, datas[3].edge_index)

    third_graph = states[2].tensor
    assert torch.equal(third_graph.x, datas[4].x)
    assert torch.equal(third_graph.edge_attr, datas[4].edge_attr)
    assert torch.equal(third_graph.edge_index, datas[4].edge_index)
    assert states.batch_shape == (3,)  # Batch shape should not change

    # Cannot set a graph with a wrong length
    with pytest.raises(AssertionError):
        states[0] = new_states
    with pytest.raises(AssertionError):
        states[1:] = new_states[0]


def test_setitem_2d(datas):
    """Test setting values in GraphStates with 2D batch shape"""
    states = MyGraphStates(datas[:4].reshape(2, 2))
    new_states_row = MyGraphStates(
        datas[4:6].reshape(
            2,
        )
    )
    states[0] = new_states_row

    assert torch.equal(states[0, 0].tensor.x, datas[4].x)
    assert torch.equal(states[0, 0].tensor.edge_attr, datas[4].edge_attr)
    assert torch.equal(states[0, 0].tensor.edge_index, datas[4].edge_index)
    assert states.batch_shape == (2, 2)  # Batch shape should not change

    # Set the new graphs in the first column
    new_states_col = MyGraphStates(
        datas[6:8].reshape(
            2,
        )
    )
    states[:, 1] = new_states_col
    assert torch.equal(states[1, 1].tensor.x, datas[7].x)
    assert torch.equal(states[1, 1].tensor.edge_attr, datas[7].edge_attr)
    assert torch.equal(states[1, 1].tensor.edge_index, datas[7].edge_index)
    assert states.batch_shape == (2, 2)  # Batch shape should not change


@pytest.mark.parametrize(
    "state_fixture",
    [
        "simple_graph_state",
        "simple_discrete_state",
        "simple_tensor_state",
        "conditional_graph_state",
        "conditional_discrete_state",
        "conditional_tensor_state",
    ],
)
def test_clone(state_fixture, request):
    """Test cloning different types of States objects"""
    state = request.getfixturevalue(state_fixture)
    cloned = state.clone()

    # Check that the clone has the same content
    assert cloned.batch_shape == state.batch_shape

    # Check conditions
    if state.conditions is not None:
        assert cloned.conditions is not None
        assert torch.equal(cloned.conditions, state.conditions)
    else:
        assert cloned.conditions is None

    # For tensor-based states
    if hasattr(state.tensor, "shape"):
        assert torch.equal(cloned.tensor, state.tensor)
    # For graph-based states
    else:
        assert torch.equal(cloned.tensor.x, state.tensor.x)
        assert torch.equal(cloned.tensor.edge_index, state.tensor.edge_index)
        assert torch.equal(cloned.tensor.edge_attr, state.tensor.edge_attr)

    # Modify the clone and check that the original is unchanged
    cloned[0] = cloned.make_initial_states((1,))
    assert state[0] != cloned[0]


@pytest.mark.parametrize(
    "state_fixture",
    [
        "simple_graph_state",
        "simple_discrete_state",
        "simple_tensor_state",
        "conditional_graph_state",
        "conditional_discrete_state",
        "conditional_tensor_state",
    ],
)
def test_is_initial_state(state_fixture, request):
    """Test is_initial_state property for different state types"""
    state = request.getfixturevalue(state_fixture)

    # Get is_initial_state
    is_initial = state.is_initial_state

    # Check shape matches batch shape
    assert is_initial.shape == state.batch_shape

    # Check type
    assert isinstance(is_initial, torch.Tensor)
    assert is_initial.dtype == torch.bool

    initial_states = state.make_initial_states(state.batch_shape, device=state.device)
    assert torch.all(initial_states.is_initial_state)


@pytest.mark.parametrize(
    "state_fixture",
    [
        "simple_graph_state",
        "simple_discrete_state",
        "simple_tensor_state",
        "conditional_graph_state",
        "conditional_discrete_state",
        "conditional_tensor_state",
    ],
)
def test_is_sink_state(state_fixture, request):
    """Test is_sink_state property for different state types"""
    state = request.getfixturevalue(state_fixture)

    # Get is_sink_state
    is_sink = state.is_sink_state

    # Check shape matches batch shape
    assert is_sink.shape == state.batch_shape

    # Check type
    assert isinstance(is_sink, torch.Tensor)
    assert is_sink.dtype == torch.bool

    sink_states = state.make_sink_states(state.batch_shape, device=state.device)
    assert torch.all(sink_states.is_sink_state)


@pytest.mark.parametrize(
    "state_fixture",
    ["simple_graph_state", "simple_discrete_state", "simple_tensor_state"],
)
def test_from_batch_shape(state_fixture, request):
    """Test creating states from batch shape for different state types"""
    StateClass = request.getfixturevalue(state_fixture).__class__

    # Create states with initial state
    states = StateClass.from_batch_shape((3,))  # device will be automatically set
    assert states.batch_shape == (3,)

    # Check all states are initial states
    is_initial = states.is_initial_state
    assert torch.all(is_initial)

    # Create states with sink state
    sink_states = StateClass.from_batch_shape((2,), sink=True)
    assert sink_states.batch_shape == (2,)

    # Check all states are sink states
    is_sink = sink_states.is_sink_state
    assert torch.all(is_sink)


@pytest.mark.parametrize(
    "empty_state_fixture",
    ["empty_graph_state", "empty_discrete_state", "empty_tensor_state"],
)
@pytest.mark.parametrize(
    "simple_state_fixture",
    [
        "simple_graph_state",
        "simple_discrete_state",
        "simple_tensor_state",
        "conditional_graph_state",
        "conditional_discrete_state",
        "conditional_tensor_state",
    ],
)
def test_extend_empty_state(empty_state_fixture, simple_state_fixture, request):
    """Test extending an empty state with a non-empty state"""
    empty_state = request.getfixturevalue(empty_state_fixture)
    simple_state = request.getfixturevalue(simple_state_fixture)

    # Skip if states are of different types
    if not isinstance(empty_state, simple_state.__class__):
        pytest.skip("States must be of same type")

    pre_extend_shape = simple_state.batch_shape
    empty_state.extend(simple_state)
    assert simple_state.batch_shape == pre_extend_shape

    # Check that the empty state now has the same content as the simple state
    assert empty_state.batch_shape == simple_state.batch_shape

    # For tensor-based states
    if hasattr(simple_state.tensor, "shape"):
        assert torch.equal(empty_state.tensor, simple_state.tensor)
    # For graph-based states
    else:
        assert torch.equal(empty_state.tensor.x, simple_state.tensor.x)
        assert torch.equal(empty_state.tensor.edge_index, simple_state.tensor.edge_index)
        assert torch.equal(empty_state.tensor.edge_attr, simple_state.tensor.edge_attr)
        assert torch.equal(empty_state.tensor.batch, simple_state.tensor.batch)


def test_extend_1d(simple_graph_state):
    """Test extending two 1D batch states"""
    other_state = simple_graph_state.clone()

    # Store original number of nodes and edges
    original_num_nodes = simple_graph_state.tensor.x.size(0)
    original_num_edges = simple_graph_state.tensor.num_edges

    other_state_batch_shape = other_state.batch_shape
    simple_graph_state.extend(other_state)
    assert other_state.batch_shape == other_state_batch_shape

    # Check batch shape is updated
    assert simple_graph_state.batch_shape[0] == 2

    # Check number of nodes and edges doubled
    assert simple_graph_state.tensor.x.size(0) == 2 * original_num_nodes
    assert simple_graph_state.tensor.num_edges == 2 * original_num_edges

    # Check that batch indices are properly updated
    batch_indices = simple_graph_state.tensor.batch
    assert torch.equal(
        batch_indices[:original_num_nodes],
        torch.zeros(original_num_nodes, dtype=torch.long),
    )
    assert torch.equal(
        batch_indices[original_num_nodes:],
        torch.ones(original_num_nodes, dtype=torch.long),
    )

    assert (
        simple_graph_state[0].tensor.edge_index == other_state[0].tensor.edge_index
    ).all()
    assert (
        simple_graph_state[1].tensor.edge_index == other_state[0].tensor.edge_index
    ).all()


def test_extend_2d(datas):
    """Test extending two 2D batch states"""
    state1 = MyGraphStates(datas[:4].reshape(2, 2))
    state2 = MyGraphStates(datas[4:].reshape(3, 2))

    # Check that we have the correct number of nodes and edges
    # Each graph has 2 nodes and 1 edge
    # For 3 time steps and 2 batches, we should have:
    expected_nodes = state1.tensor.x.size(0) + state2.tensor.x.size(0)
    assert isinstance(MyGraphStates.sf.x.size(0), int)  # type: ignore
    expected_nodes += 2 * MyGraphStates.sf.x.size(0)  # type: ignore
    expected_edges = state1.tensor.num_edges + state2.tensor.num_edges
    expected_edges += 2 * MyGraphStates.sf.num_edges

    # Extend state1 with state2
    pre_extend_shape = state2.batch_shape
    state1.extend(state2)
    assert state2.batch_shape == pre_extend_shape
    # Check final shape should be (max_len=3, B=4)
    assert state1.batch_shape == (3, 4)

    # The actual count might be higher due to padding with sink states
    assert state1.tensor.x.size(0) == expected_nodes
    assert state1.tensor.num_edges == expected_edges

    # Check if states are extended as expected
    assert (state1[0, 0].tensor.x == datas[0].x).all()
    assert (state1[0, 1].tensor.x == datas[1].x).all()
    assert (state1[0, 2].tensor.x == datas[4].x).all()
    assert (state1[0, 3].tensor.x == datas[5].x).all()
    assert (state1[1, 0].tensor.x == datas[2].x).all()
    assert (state1[1, 1].tensor.x == datas[3].x).all()
    assert (state1[1, 2].tensor.x == datas[6].x).all()
    assert (state1[1, 3].tensor.x == datas[7].x).all()
    assert (state1[2, 0].tensor.x == MyGraphStates.sf.x).all()
    assert (state1[2, 1].tensor.x == MyGraphStates.sf.x).all()
    assert (state1[2, 2].tensor.x == datas[8].x).all()
    assert (state1[2, 3].tensor.x == datas[9].x).all()

    is_sink_state = torch.zeros(state1.batch_shape, dtype=torch.bool)
    is_sink_state[2, 0] = True
    is_sink_state[2, 1] = True
    assert torch.all(state1.is_sink_state == is_sink_state), state1.is_sink_state

    assert (state1[0, 0].tensor.edge_index == datas[0].edge_index).all()
    assert (state1[0, 1].tensor.edge_index == datas[1].edge_index).all()
    assert (state1[0, 2].tensor.edge_index == datas[4].edge_index).all()
    assert (state1[0, 3].tensor.edge_index == datas[5].edge_index).all()
    assert (state1[1, 0].tensor.edge_index == datas[2].edge_index).all()
    assert (state1[1, 1].tensor.edge_index == datas[3].edge_index).all()
    assert (state1[1, 2].tensor.edge_index == datas[6].edge_index).all()
    assert (state1[1, 3].tensor.edge_index == datas[7].edge_index).all()
    assert (state1[2, 0].tensor.edge_index == MyGraphStates.sf.edge_index).all()
    assert (state1[2, 1].tensor.edge_index == MyGraphStates.sf.edge_index).all()
    assert (state1[2, 2].tensor.edge_index == datas[8].edge_index).all()
    assert (state1[2, 3].tensor.edge_index == datas[9].edge_index).all()


def test_forward_masks(datas):
    """Test forward_masks property"""
    # Create a graph with 2 nodes and 1 edge
    states = MyGraphStates(datas[:1])

    # Get forward masks
    masks = states.forward_masks

    # Check action type mask
    assert masks[GraphActions.ACTION_TYPE_KEY].shape == (1, 3)
    assert masks[GraphActions.ACTION_TYPE_KEY][
        0, GraphActionType.ADD_NODE
    ].item()  # Can add node
    assert (
        masks[GraphActions.ACTION_TYPE_KEY][0, GraphActionType.ADD_EDGE]
    ).item()  # Can add edge (2 nodes)
    assert masks[GraphActions.ACTION_TYPE_KEY][
        0, GraphActionType.EXIT
    ].item()  # Can exit

    # Check node class mask
    assert masks[GraphActions.NODE_CLASS_KEY].shape == (1, states.num_node_classes)
    assert torch.all(masks[GraphActions.NODE_CLASS_KEY])

    # Check node index mask
    assert masks[GraphActions.NODE_INDEX_KEY].shape == (1, states.tensor.x.size(0) + 1)
    assert torch.all(torch.sum(masks[GraphActions.NODE_INDEX_KEY], dim=-1) == 1)

    # Check edge_class mask
    assert masks[GraphActions.EDGE_CLASS_KEY].shape == (1, states.num_edge_classes)
    assert torch.all(masks[GraphActions.EDGE_CLASS_KEY])

    # Check edge_index masks
    assert len(masks[GraphActions.EDGE_INDEX_KEY]) == 1  # 1 graph
    assert torch.all(
        masks[GraphActions.EDGE_INDEX_KEY][0] == torch.tensor([[False, True]])
    )


def test_backward_masks(datas):
    """Test backward_masks property"""
    # Create a graph with 2 nodes and 1 edge
    states = MyGraphStates(datas[:1])

    # Get backward masks
    masks = states.backward_masks

    # Check action type mask
    assert masks[GraphActions.ACTION_TYPE_KEY].shape == (1, 3)
    assert not masks[GraphActions.ACTION_TYPE_KEY][
        0, GraphActionType.ADD_NODE
    ].item()  # Can't remove node as it has an edge
    assert masks[GraphActions.ACTION_TYPE_KEY][
        0, GraphActionType.ADD_EDGE
    ].item()  # Can remove edge
    assert not masks[GraphActions.ACTION_TYPE_KEY][
        0, GraphActionType.EXIT
    ].item()  # Can exit

    # Check node_class mask
    assert masks[GraphActions.NODE_CLASS_KEY].shape == (1, states.num_node_classes)
    assert not torch.any(masks[GraphActions.NODE_CLASS_KEY])

    # Check node index mask
    assert masks[GraphActions.NODE_INDEX_KEY].shape == (1, states.tensor.x.size(0))
    assert not torch.any(masks[GraphActions.NODE_INDEX_KEY])

    # Check edge_class mask
    assert masks[GraphActions.EDGE_CLASS_KEY].shape == (1, states.num_edge_classes)
    assert torch.all(masks[GraphActions.EDGE_CLASS_KEY])

    # Check edge_index masks
    assert len(masks[GraphActions.EDGE_INDEX_KEY]) == 1  # 1 graph
    assert torch.all(
        masks[GraphActions.EDGE_INDEX_KEY][0] == torch.tensor([[True, False]])
    )


def test_stack_1d(datas):
    """Test stacking GraphStates objects"""
    # Create two states
    state1 = MyGraphStates(datas[0:2])
    state2 = MyGraphStates(datas[2:4])

    # Stack the states
    stacked = MyGraphStates.stack([state1, state2])

    # Check the batch shape
    assert stacked.batch_shape == (2, 2)

    # Check the number of nodes and edges
    assert stacked.tensor.x.size(0) == 8  # 4 states * 2 nodes
    assert stacked.tensor.num_edges == 4  # 4 states * 1 edge

    # Check the batch indices
    assert torch.equal(stacked.tensor.batch[:4], state1.tensor.batch)
    assert torch.equal(stacked.tensor.batch[4:], state2.tensor.batch + 2)

    assert torch.all(stacked[0, 0].tensor.x == datas[0].x)
    assert torch.all(stacked[0, 1].tensor.x == datas[1].x)
    assert torch.all(stacked[1, 0].tensor.x == datas[2].x)
    assert torch.all(stacked[1, 1].tensor.x == datas[3].x)

    assert (stacked[0, 0].tensor.edge_index == datas[0].edge_index).all()
    assert (stacked[0, 1].tensor.edge_index == datas[1].edge_index).all()
    assert (stacked[1, 0].tensor.edge_index == datas[2].edge_index).all()
    assert (stacked[1, 1].tensor.edge_index == datas[3].edge_index).all()


def test_stack_2d(datas):
    """Test stacking GraphStates objects with 2D batch shape"""
    # Create two states
    state1 = MyGraphStates(datas[:4].reshape(2, 2))
    state2 = MyGraphStates(datas[4:8].reshape(2, 2))

    # Stack the states
    stacked = MyGraphStates.stack([state1, state2])

    # Check the batch shape
    assert stacked.batch_shape == (2, 2, 2)

    # Check the number of nodes and edges
    assert stacked.tensor.x.size(0) == 16  # 8 states * 2 nodes
    assert stacked.tensor.num_edges == 8  # 8 states * 1 edge

    # Check the batch indices
    assert torch.equal(stacked.tensor.batch[:8], state1.tensor.batch)
    assert torch.equal(stacked.tensor.batch[8:], state2.tensor.batch + 4)

    assert torch.all(stacked[0, 0, 0].tensor.x == datas[0].x)
    assert torch.all(stacked[0, 0, 1].tensor.x == datas[1].x)
    assert torch.all(stacked[0, 1, 0].tensor.x == datas[2].x)
    assert torch.all(stacked[0, 1, 1].tensor.x == datas[3].x)
    assert torch.all(stacked[1, 0, 0].tensor.x == datas[4].x)
    assert torch.all(stacked[1, 0, 1].tensor.x == datas[5].x)
    assert torch.all(stacked[1, 1, 0].tensor.x == datas[6].x)
    assert torch.all(stacked[1, 1, 1].tensor.x == datas[7].x)

    assert (stacked[0, 0, 0].tensor.edge_index == datas[0].edge_index).all()
    assert (stacked[0, 0, 1].tensor.edge_index == datas[1].edge_index).all()
    assert (stacked[0, 1, 0].tensor.edge_index == datas[2].edge_index).all()
    assert (stacked[0, 1, 1].tensor.edge_index == datas[3].edge_index).all()
    assert (stacked[1, 0, 0].tensor.edge_index == datas[4].edge_index).all()
    assert (stacked[1, 0, 1].tensor.edge_index == datas[5].edge_index).all()
    assert (stacked[1, 1, 0].tensor.edge_index == datas[6].edge_index).all()
    assert (stacked[1, 1, 1].tensor.edge_index == datas[7].edge_index).all()


def test_discrete_masks_device_consistency_construct(simple_discrete_state):
    state = simple_discrete_state
    assert state.forward_masks.device == state.device
    assert state.backward_masks.device == state.device


def test_discrete_masks_device_consistency_index_clone_flatten_extend(
    simple_discrete_state,
):
    state = simple_discrete_state

    # Indexing
    sub = state[0]
    assert sub.forward_masks.device == sub.device
    assert sub.backward_masks.device == sub.device

    # Clone
    cloned = state.clone()
    assert cloned.forward_masks.device == cloned.device
    assert cloned.backward_masks.device == cloned.device

    # Flatten
    flat = state.flatten()
    assert flat.forward_masks.device == flat.device
    assert flat.backward_masks.device == flat.device

    # Extend (devices must already match by contract)
    other = state.clone()
    pre_extend_shape = other.batch_shape
    state.extend(other)
    assert other.batch_shape == pre_extend_shape
    assert state.forward_masks.device == state.device
    assert state.backward_masks.device == state.device


def test_discrete_masks_device_consistency_after_pad_and_stack(simple_discrete_state):
    # Build a 2D batch via stacking
    s1 = simple_discrete_state
    s2 = s1.clone()
    stacked = s1.__class__.stack([s1, s2])

    # Pad first dimension and ensure masks stay on the same device
    stacked.pad_dim0_with_sf(required_first_dim=3)
    assert stacked.forward_masks.device == stacked.device
    assert stacked.backward_masks.device == stacked.device

    # Stacked masks should already be on the correct device
    assert stacked.forward_masks.device == stacked.device
    assert stacked.backward_masks.device == stacked.device


def test_discrete_masks_device_consistency_after_mask_ops(simple_discrete_state):
    state = simple_discrete_state

    # set_nonexit_action_masks should not move devices
    cond = torch.zeros(
        state.batch_shape + (state.n_actions - 1,), dtype=torch.bool, device=state.device
    )
    state.set_nonexit_action_masks(cond=cond, allow_exit=True)
    assert state.forward_masks.device == state.device

    # set_exit_masks should not move devices
    batch_idx = torch.ones(state.batch_shape, dtype=torch.bool, device=state.device)
    state.set_exit_masks(batch_idx)
    assert state.forward_masks.device == state.device

    # init_forward_masks should keep device
    state.init_forward_masks(set_ones=True)
    assert state.forward_masks.device == state.device


def _make_discrete(batch_shape: tuple[int, ...]) -> DiscreteStates:
    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 4
        s0 = torch.zeros(2)
        sf = torch.ones(2)

    tensor = torch.zeros(batch_shape + SimpleDiscreteStates.state_shape)
    return SimpleDiscreteStates(tensor, debug=True)


def test_set_nonexit_action_masks_resets_each_call_1d():
    state = _make_discrete((2,))

    cond1 = torch.tensor([[False, True, False], [True, False, False]], dtype=torch.bool)
    state.set_nonexit_action_masks(cond=cond1, allow_exit=True)
    expected1 = torch.tensor(
        [[True, False, True, True], [False, True, True, True]], dtype=torch.bool
    )
    assert torch.equal(state.forward_masks, expected1)

    # Second call should start from all True because set_nonexit_action_masks resets.
    cond2 = torch.zeros_like(cond1, dtype=torch.bool)
    state.set_nonexit_action_masks(cond=cond2, allow_exit=True)
    expected2 = torch.ones_like(expected1)
    assert torch.equal(state.forward_masks, expected2)


def test_set_nonexit_action_masks_resets_each_call_2d():
    state = _make_discrete((2, 2))

    cond1 = torch.tensor(
        [
            [[False, True, False], [True, False, False]],
            [[False, False, True], [False, True, True]],
        ],
        dtype=torch.bool,
    )
    state.set_nonexit_action_masks(cond=cond1, allow_exit=False)
    # When allow_exit is False, exit column is also masked to False.
    expected1 = torch.tensor(
        [
            [[True, False, True, False], [False, True, True, False]],
            [[True, True, False, False], [True, False, False, False]],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected1)

    # Second call should reset masks before applying the new condition.
    cond2 = torch.tensor(
        [
            [[True, False, True], [False, False, False]],
            [[False, True, False], [True, True, False]],
        ],
        dtype=torch.bool,
    )
    state.set_nonexit_action_masks(cond=cond2, allow_exit=True)
    expected2 = torch.tensor(
        [
            [[False, True, False, True], [True, True, True, True]],
            [[True, False, True, True], [False, False, True, True]],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected2)


def test_set_exit_masks_exit_only_1d():
    state = _make_discrete((3,))
    state.init_forward_masks(set_ones=True)

    batch_idx = torch.tensor([True, False, True], dtype=torch.bool)
    state.set_exit_masks(batch_idx)

    expected = torch.tensor(
        [
            [False, False, False, True],
            [True, True, True, True],
            [False, False, False, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected)

    # Running again with a different mask after re-init should not leak previous masks.
    state.init_forward_masks(set_ones=True)
    batch_idx2 = torch.tensor([False, True, False], dtype=torch.bool)
    state.set_exit_masks(batch_idx2)
    expected2 = torch.tensor(
        [
            [True, True, True, True],
            [False, False, False, True],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected2)


def test_set_exit_masks_exit_only_2d():
    state = _make_discrete((2, 2))
    state.init_forward_masks(set_ones=True)

    batch_idx = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    state.set_exit_masks(batch_idx)

    expected = torch.tensor(
        [
            [[False, False, False, True], [True, True, True, True]],
            [[True, True, True, True], [False, False, False, True]],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected)

    # Re-init and apply a different mask to ensure previous updates don't leak.
    state.init_forward_masks(set_ones=True)
    batch_idx2 = torch.tensor([[False, False], [True, False]], dtype=torch.bool)
    state.set_exit_masks(batch_idx2)
    expected2 = torch.tensor(
        [
            [[True, True, True, True], [True, True, True, True]],
            [[False, False, False, True], [True, True, True, True]],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(state.forward_masks, expected2)


def test_states_factory_requires_debug():
    class NoDebugStates(States):
        state_shape = (1,)
        s0 = torch.tensor([0.0])
        sf = torch.tensor([1.0])

        @classmethod
        def make_random_states(cls, batch_shape, device=None):
            return cls(torch.zeros(batch_shape + cls.state_shape, device=device))

    with pytest.raises(TypeError, match="must accept a `debug`"):
        NoDebugStates.from_batch_shape((2,), random=True, debug=True)


def test_discrete_states_factory_requires_debug():
    class NoDebugDiscreteStates(DiscreteStates):
        state_shape = (1,)
        s0 = torch.tensor([0.0])
        sf = torch.tensor([1.0])
        n_actions = 2

        @classmethod
        def make_random_states(cls, batch_shape, device=None):
            t = torch.zeros(batch_shape + cls.state_shape, device=device)
            return cls(t)

    with pytest.raises(TypeError, match="must accept a `debug`"):
        NoDebugDiscreteStates.from_batch_shape((2,), random=True, debug=True)


def test_graph_states_factory_requires_debug():
    class NoDebugGraphStates(GraphStates):
        num_node_classes = 1
        num_edge_classes = 1
        is_directed = True
        max_nodes = 2

        s0 = GeometricData(
            x=torch.zeros((1, 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
        )
        sf = s0.clone()

        @classmethod
        def make_random_states(cls, batch_shape, device=None):
            batch_shape = (
                batch_shape if isinstance(batch_shape, tuple) else (batch_shape,)
            )
            data_array = np.empty(batch_shape, dtype=object)
            for i in range(np.prod(batch_shape)):
                data_array.flat[i] = cls.s0.clone()

            if device is not None:
                dev = device
            else:
                dev = cls.s0.x.device  # pyright: ignore[reportOptionalMemberAccess]

            return cls(
                data_array,
                categorical_node_features=True,
                categorical_edge_features=True,
                device=dev,
            )

    with pytest.raises(TypeError, match="must accept a `debug`"):
        NoDebugGraphStates.from_batch_shape((2,), random=True, debug=True)


def normalize_device(device):
    """Normalize device to use index form (cuda:0 instead of cuda)"""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def _assert_tensordict_on_device(tensordict, device):
    device = normalize_device(device)
    for k, v in tensordict.items():
        v_device = normalize_device(v.device)
        assert v_device == device, f"Tensor {k} on {v.device} != {device}"


def test_graph_masks_device_consistency(simple_graph_state):
    state = simple_graph_state
    fm = state.forward_masks
    bm = state.backward_masks
    _assert_tensordict_on_device(fm, state.device)
    _assert_tensordict_on_device(bm, state.device)


def test_graph_masks_device_after_stack(datas):
    s1 = MyGraphStates(datas[:1])
    s2 = MyGraphStates(datas[1:2])
    stacked = MyGraphStates.stack([s1, s2])
    fm = stacked.forward_masks
    bm = stacked.backward_masks
    _assert_tensordict_on_device(fm, stacked.device)
    _assert_tensordict_on_device(bm, stacked.device)


def test_graph_masks_device_after_to_noop(simple_graph_state):
    # to() with the same device should be a no-op; masks should still be created on that device
    state = simple_graph_state
    state.to(state.device)
    fm = state.forward_masks
    bm = state.backward_masks
    _assert_tensordict_on_device(fm, state.device)
    _assert_tensordict_on_device(bm, state.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_discrete_masks_device_on_cuda():
    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 3
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    tensor = torch.tensor([[0.5, 0.5]], device=torch.device("cuda"))

    state = SimpleDiscreteStates(tensor)
    assert state.device.type == "cuda"
    assert state.forward_masks.device.type == "cuda"
    assert state.backward_masks.device.type == "cuda"

    # Mask ops keep CUDA
    cond = torch.zeros(
        state.batch_shape + (state.n_actions - 1,), dtype=torch.bool, device=state.device
    )
    state.set_nonexit_action_masks(cond=cond, allow_exit=True)
    assert state.forward_masks.device.type == "cuda"

    batch_idx = torch.ones(state.batch_shape, dtype=torch.bool, device=state.device)
    state.set_exit_masks(batch_idx)
    assert state.forward_masks.device.type == "cuda"

    state.init_forward_masks(set_ones=False)
    assert state.forward_masks.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graph_masks_device_on_cuda(datas):

    # Build two graphs on CUDA
    s1 = MyGraphStates(datas[:1]).to(torch.device("cuda"))
    s2 = MyGraphStates(datas[1:2]).to(torch.device("cuda"))

    # Sanity: device is CUDA
    assert s1.device.type == "cuda"
    assert s2.device.type == "cuda"

    # Masks on single states are on CUDA
    print("comparing {} : {}".format(s1.forward_masks.device, s1.device))
    _assert_tensordict_on_device(s1.forward_masks, s1.device)
    _assert_tensordict_on_device(s1.backward_masks, s1.device)

    # Stacked
    stacked = MyGraphStates.stack([s1, s2])
    assert stacked.device.type == "cuda"
    _assert_tensordict_on_device(stacked.forward_masks, stacked.device)
    _assert_tensordict_on_device(stacked.backward_masks, stacked.device)


def test_states_cross_device_extend_raises_meta():
    dev1, dev2 = torch.device("cpu"), torch.device("meta")

    class SimpleTensorStates(States):
        state_shape = (2,)
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    a = SimpleTensorStates(torch.tensor([[0.5, 0.5]], device=dev1))
    b = SimpleTensorStates(torch.tensor([[0.1, 0.2]], device=dev2))
    with pytest.raises(RuntimeError):
        a.extend(b)


def test_discrete_cross_device_extend_raises_meta():
    dev1, dev2 = torch.device("cpu"), torch.device("meta")

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 3
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    a = SimpleDiscreteStates(torch.tensor([[0.5, 0.5]], device=dev1))
    b = SimpleDiscreteStates(torch.tensor([[0.1, 0.2]], device=dev2))
    with pytest.raises(AssertionError):
        a.extend(b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_states_instance_to_cuda_roundtrip():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    class SimpleTensorStates(States):
        state_shape = (2,)
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    st = SimpleTensorStates(torch.tensor([[0.5, 0.5]], device=cpu))
    assert st.device.type == "cpu"
    st.to(cuda)
    assert st.device.type == "cuda"
    st.to(cpu)
    assert st.device.type == "cpu"


def test_states_instance_device_cpu_noop():
    dev1, dev2 = torch.device("cpu"), torch.device("cpu")

    class SimpleTensorStates(States):
        state_shape = (2,)
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    st = SimpleTensorStates(torch.tensor([[0.5, 0.5]], device=dev1))
    assert st.device == dev1
    st.to(dev2)
    assert st.device == dev2


def test_discrete_instance_device_cpu_noop():
    dev1, dev2 = torch.device("cpu"), torch.device("cpu")

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 3
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    tensor = torch.tensor([[0.5, 0.5]], device=dev1)
    ds = SimpleDiscreteStates(tensor)
    assert ds.device == dev1
    assert ds.forward_masks.device == dev1
    assert ds.backward_masks.device == dev1
    ds.to(dev2)
    assert ds.device == dev2
    assert ds.forward_masks.device == dev2
    assert ds.backward_masks.device == dev2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_states_cross_device_extend_raises_cuda():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    class SimpleTensorStates(States):
        state_shape = (2,)
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    a = SimpleTensorStates(torch.tensor([[0.5, 0.5]], device=cpu))
    b = SimpleTensorStates(torch.tensor([[0.1, 0.2]], device=cuda))
    with pytest.raises(RuntimeError):
        a.extend(b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_discrete_cross_device_extend_raises_cuda():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 3
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    a = SimpleDiscreteStates(torch.tensor([[0.5, 0.5]], device=cpu))
    b = SimpleDiscreteStates(torch.tensor([[0.1, 0.2]], device=cuda))
    with pytest.raises(AssertionError):
        a.extend(b)


def test_graphstates_instance_device_and_masks_cpu(datas):
    dev1, dev2 = torch.device("cpu"), torch.device("cpu")
    s1 = MyGraphStates(datas[:1])
    assert s1.device == dev1
    fm = s1.forward_masks
    bm = s1.backward_masks
    _assert_tensordict_on_device(fm, s1.device)
    _assert_tensordict_on_device(bm, s1.device)
    # to() with same cpu device is a no-op but should preserve consistency
    s1.to(dev2)
    assert s1.device == dev2
    fm2 = s1.forward_masks
    bm2 = s1.backward_masks
    _assert_tensordict_on_device(fm2, s1.device)
    _assert_tensordict_on_device(bm2, s1.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_states_two_instances_different_devices_cuda():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    class SimpleTensorStates(States):
        state_shape = (2,)
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    A = SimpleTensorStates(torch.tensor([[0.1, 0.2]], device=cpu))
    B = SimpleTensorStates(torch.tensor([[0.3, 0.4]], device=cuda))
    assert A.device.type == "cpu"
    assert B.device.type == "cuda"
    assert torch.all(A.is_initial_state == torch.tensor([False], device=A.device))
    assert torch.all(B.is_initial_state == torch.tensor([False], device=B.device))
    # Move B back to cpu; A should be unaffected
    B.to(cpu)
    assert B.device.type == "cpu"
    assert A.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_discrete_states_two_instances_different_devices_cuda():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)
        n_actions = 3
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    A = SimpleDiscreteStates(torch.tensor([[0.5, 0.5]], device=cpu))
    B = SimpleDiscreteStates(
        torch.tensor([[0.1, 0.2]], device=cuda),
        torch.ones((1, 3), dtype=torch.bool, device=cuda),
        torch.ones((1, 2), dtype=torch.bool, device=cuda),
    )
    assert A.device.type == "cpu"
    assert B.device.type == "cuda"
    # Mask devices are consistent with instance devices
    assert A.forward_masks.device.type == "cpu"
    assert B.forward_masks.device.type == "cuda"
    # Move B back to cpu; A should be unaffected
    B.to(cpu)
    assert B.device.type == "cpu"
    assert A.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graph_states_two_instances_different_devices_cuda(datas):
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    s_cpu = MyGraphStates(datas[:1])
    s_cuda = MyGraphStates(datas[1:2]).to(cuda)
    assert s_cpu.device.type == "cpu"
    assert s_cuda.device.type == "cuda"
    # is_initial_state / is_sink_state should work without mutating class templates
    _ = s_cpu.is_initial_state
    _ = s_cpu.is_sink_state
    _ = s_cuda.is_initial_state
    _ = s_cuda.is_sink_state
    # Move back and ensure consistency
    s_cuda.to(cpu)
    assert s_cuda.device.type == "cpu"
