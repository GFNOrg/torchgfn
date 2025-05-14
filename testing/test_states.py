import pytest
import torch
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from gfn.actions import GraphActionType
from gfn.states import DiscreteStates, GraphStates, States


class MyGraphStates(GraphStates):
    num_node_classes = 10
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


@pytest.fixture
def datas():
    """Creates a list of 10 GeometricData objects"""
    return [
        GeometricData(
            x=torch.tensor([[i], [i + 1]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[i]]),
        )
        for i in range(10)
    ]


@pytest.fixture
def simple_graph_state(datas):
    """Creates a simple graph state with 2 nodes and 1 edge"""
    data = datas[0]
    batch = GeometricBatch.from_data_list([data])
    batch.batch_shape = (1,)
    return MyGraphStates(batch)


@pytest.fixture
def empty_graph_state():
    """Creates an empty GraphStates object"""
    # Create an empty batch
    batch = GeometricBatch()
    batch.x = torch.zeros((0, 1))
    batch.edge_index = torch.zeros((2, 0), dtype=torch.long)
    batch.edge_attr = torch.zeros((0, 1))
    batch.batch = torch.zeros((0,), dtype=torch.long)
    batch.batch_shape = (0,)
    return MyGraphStates(batch)


@pytest.fixture
def simple_discrete_state():
    """Creates a simple discrete state with 3 possible actions"""

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)  # 2-dimensional state
        n_actions = 3  # 3 possible actions
        device = torch.device("cpu")
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    # Create a single state tensor
    tensor = torch.tensor([[0.5, 0.5]])
    forward_masks = torch.tensor([[True, True, True]])  # All actions allowed
    backward_masks = torch.tensor([[True, True]])  # All backward actions allowed

    return SimpleDiscreteStates(tensor, forward_masks, backward_masks)


@pytest.fixture
def empty_discrete_state():
    """Creates an empty discrete state"""

    class SimpleDiscreteStates(DiscreteStates):
        state_shape = (2,)  # 2-dimensional state
        n_actions = 3  # 3 possible actions
        device = torch.device("cpu")
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    # Create an empty state tensor
    tensor = torch.zeros((0, 2))
    forward_masks = torch.zeros((0, 3), dtype=torch.bool)
    backward_masks = torch.zeros((0, 2), dtype=torch.bool)

    return SimpleDiscreteStates(tensor, forward_masks, backward_masks)


@pytest.fixture
def simple_tensor_state():
    """Creates a simple tensor state"""

    class SimpleTensorStates(States):
        state_shape = (2,)  # 2-dimensional state
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    # Create a single state tensor
    tensor = torch.tensor([[0.5, 0.5]])
    return SimpleTensorStates(tensor)


@pytest.fixture
def empty_tensor_state():
    """Creates an empty tensor state"""

    class SimpleTensorStates(States):
        state_shape = (2,)  # 2-dimensional state
        s0 = torch.tensor([0.0, 0.0])
        sf = torch.tensor([1.0, 1.0])

    # Create an empty state tensor
    tensor = torch.zeros((0, 2))
    return SimpleTensorStates(tensor)


def test_getitem_1d(datas):
    """Test indexing into GraphStates

    Make sure the behavior is consistent with that of a Tensor.__getitem__.
    """
    # Create a tensor with 3 elements for comparison
    tsr = torch.tensor([1, 2, 3])

    # Create a batch with 3 graphs
    batch = GeometricBatch.from_data_list(datas[:3])
    batch.batch_shape = (3,)
    assert tuple(tsr.shape) == batch.batch_shape == (3,)
    states = MyGraphStates(batch)
    states.log_rewards = tsr.clone()

    # Get a single graph
    single_tsr = tsr[1]
    single_state = states[1]
    assert tuple(single_tsr.shape) == single_state.tensor.batch_shape == ()
    assert single_state.log_rewards is not None and single_state.log_rewards.shape == ()
    assert single_state.tensor.num_nodes == 2
    assert torch.allclose(single_state.tensor.x, datas[1].x)
    assert torch.allclose(single_state.log_rewards, tsr[1])

    # Get multiple graphs
    multi_tsr = tsr[[0, 2]]
    multi_state = states[[0, 2]]
    assert tuple(multi_tsr.shape) == multi_state.tensor.batch_shape == (2,)
    assert multi_state.log_rewards is not None and multi_state.log_rewards.shape == (2,)
    assert multi_state.tensor.num_nodes == 4
    assert torch.allclose(multi_state.tensor.get_example(0).x, datas[0].x)
    assert torch.allclose(multi_state.tensor.get_example(1).x, datas[2].x)
    assert torch.allclose(multi_state.log_rewards, tsr[[0, 2]])


def test_getitem_2d(datas):
    """Test indexing into GraphStates with 2D batch shape

    Make sure the behavior is consistent with that of a Tensor.__getitem__.
    """
    # Create a tensor with 4 elements for comparison
    tsr = torch.tensor([[1, 2], [3, 4]])

    # Create a batch with 2x2 graphs
    batch = GeometricBatch.from_data_list(datas[:4])
    batch.batch_shape = (2, 2)
    assert tuple(tsr.shape) == batch.batch_shape == (2, 2)
    states = MyGraphStates(batch)
    states.log_rewards = tsr.clone()

    # Get a single row
    tsr_row = tsr[0]
    batch_row = states[0]
    assert tuple(tsr_row.shape) == batch_row.tensor.batch_shape == (2,)
    assert batch_row.log_rewards is not None and batch_row.log_rewards.shape == (2,)
    assert batch_row.tensor.num_nodes == 4  # 2 graphs * 2 nodes
    assert torch.allclose(batch_row.tensor.get_example(0).x, datas[0].x)
    assert torch.allclose(batch_row.tensor.get_example(1).x, datas[1].x)
    assert torch.allclose(batch_row.log_rewards, tsr[0])

    # Try again with slicing
    tsr_row2 = tsr[0, :]
    batch_row2 = states[0, :]
    assert tuple(tsr_row2.shape) == batch_row2.tensor.batch_shape == (2,)
    assert torch.equal(batch_row.tensor.x, batch_row2.tensor.x)

    # Get a single graph with 2D indexing
    single_tsr = tsr[1, 1]
    single_state = states[1, 1]
    assert tuple(single_tsr.shape) == single_state.tensor.batch_shape == ()
    assert single_state.log_rewards is not None and single_state.log_rewards.shape == ()
    assert single_state.tensor.num_nodes == 2  # 1 graph * 2 nodes
    assert torch.allclose(single_state.tensor.x, datas[3].x)
    assert torch.allclose(single_state.log_rewards, tsr[1, 1])

    with pytest.raises(IndexError):
        states[2, 2]

    # We can't index on a Batch with 0-dimensional batch shape
    with pytest.raises(AssertionError):
        single_state[0]


def test_setitem_1d(datas):
    """Test setting values in States"""
    # Create a graph state with 3 graphs
    batch = GeometricBatch.from_data_list(datas[:3])
    batch.batch_shape = (3,)
    states = MyGraphStates(batch)

    # Create a new graph state
    new_batch = GeometricBatch.from_data_list(datas[3:5])
    new_batch.batch_shape = (2,)
    new_states = MyGraphStates(new_batch)

    # Set the new graph in the first position
    states[0] = new_states[0]

    # Check that the first graph is now the new graph
    first_graph = states[0].tensor
    assert torch.equal(first_graph.x, datas[3].x)
    assert torch.equal(first_graph.edge_attr, datas[3].edge_attr)
    assert torch.equal(first_graph.edge_index, datas[3].edge_index)
    assert states.tensor.batch_shape == (3,)  # Batch shape should not change

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
    assert states.tensor.batch_shape == (3,)  # Batch shape should not change

    # Cannot set a graph with a wrong length
    with pytest.raises(AssertionError):
        states[0] = new_states
    with pytest.raises(AssertionError):
        states[1:] = new_states[0]


def test_setitem_2d(datas):
    """Test setting values in GraphStates with 2D batch shape"""
    # Create a graph state with 2x2 graphs
    batch = GeometricBatch.from_data_list(datas[:4])
    batch.batch_shape = (2, 2)
    states = MyGraphStates(batch)

    # Set the new graphs in the first row
    new_batch_row = GeometricBatch.from_data_list(datas[4:6])
    new_batch_row.batch_shape = (2,)
    new_states_row = MyGraphStates(new_batch_row)
    states[0] = new_states_row
    assert torch.equal(states[0, 0].tensor.x, datas[4].x)
    assert torch.equal(states[0, 0].tensor.edge_attr, datas[4].edge_attr)
    assert torch.equal(states[0, 0].tensor.edge_index, datas[4].edge_index)
    assert states.tensor.batch_shape == (2, 2)  # Batch shape should not change

    # Set the new graphs in the first column
    new_batch_col = GeometricBatch.from_data_list(datas[6:8])
    new_batch_col.batch_shape = (2,)
    new_states_col = MyGraphStates(new_batch_col)
    states[:, 1] = new_states_col
    assert torch.equal(states[1, 1].tensor.x, datas[7].x)
    assert torch.equal(states[1, 1].tensor.edge_attr, datas[7].edge_attr)
    assert torch.equal(states[1, 1].tensor.edge_index, datas[7].edge_index)
    assert states.tensor.batch_shape == (2, 2)  # Batch shape should not change


@pytest.mark.parametrize(
    "state_fixture",
    ["simple_graph_state", "simple_discrete_state", "simple_tensor_state"],
)
def test_clone(state_fixture, request):
    """Test cloning different types of States objects"""
    state = request.getfixturevalue(state_fixture)
    cloned = state.clone()

    # Check that the clone has the same content
    assert cloned.batch_shape == state.batch_shape

    # For tensor-based states
    if hasattr(state.tensor, "shape"):
        assert torch.equal(cloned.tensor, state.tensor)
    # For graph-based states
    else:
        assert torch.equal(cloned.tensor.x, state.tensor.x)
        assert torch.equal(cloned.tensor.edge_index, state.tensor.edge_index)
        assert torch.equal(cloned.tensor.edge_attr, state.tensor.edge_attr)

    # Modify the clone and check that the original is unchanged
    if hasattr(state.tensor, "shape"):
        cloned.tensor[..., 0] = 99.0
        assert cloned.tensor[..., 0] == 99.0
        assert state.tensor[..., 0] != 99.0
    else:
        cloned.tensor.x[..., 0] = 99.0
        assert cloned.tensor.x[0] == 99.0
        assert state.tensor.x[0] != 99.0


@pytest.mark.parametrize(
    "state_fixture",
    ["simple_graph_state", "simple_discrete_state", "simple_tensor_state"],
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

    initial_states = state.make_initial_states_tensor(state.batch_shape, state.device)
    assert torch.all(state.__class__(initial_states).is_initial_state)


@pytest.mark.parametrize(
    "state_fixture",
    ["simple_graph_state", "simple_discrete_state", "simple_tensor_state"],
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

    sink_states = state.make_sink_states_tensor(state.batch_shape, state.device)
    assert torch.all(state.__class__(sink_states).is_sink_state)


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
    ["simple_graph_state", "simple_discrete_state", "simple_tensor_state"],
)
def test_extend_empty_state(empty_state_fixture, simple_state_fixture, request):
    """Test extending an empty state with a non-empty state"""
    empty_state = request.getfixturevalue(empty_state_fixture)
    simple_state = request.getfixturevalue(simple_state_fixture)

    # Skip if states are of different types
    if not isinstance(empty_state, simple_state.__class__):
        pytest.skip("States must be of same type")

    empty_state.extend(simple_state)

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
    original_num_nodes = simple_graph_state.tensor.num_nodes
    original_num_edges = simple_graph_state.tensor.num_edges

    simple_graph_state.extend(other_state)

    # Check batch shape is updated
    assert simple_graph_state.tensor.batch_shape[0] == 2

    # Check number of nodes and edges doubled
    assert simple_graph_state.tensor.num_nodes == 2 * original_num_nodes
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
    batch1 = GeometricBatch.from_data_list(datas[:4])
    batch1.batch_shape = (2, 2)
    state1 = MyGraphStates(batch1)

    batch2 = GeometricBatch.from_data_list(datas[4:])
    batch2.batch_shape = (3, 2)
    state2 = MyGraphStates(batch2)

    # Extend state1 with state2
    state1.extend(state2)

    # Check final shape should be (max_len=3, B=4)
    assert state1.tensor.batch_shape == (3, 4)

    # Check that we have the correct number of nodes and edges
    # Each graph has 2 nodes and 1 edge
    # For 3 time steps and 2 batches, we should have:
    expected_nodes = batch1.num_nodes + batch2.num_nodes
    assert isinstance(MyGraphStates.sf.num_nodes, int)
    expected_nodes += 2 * MyGraphStates.sf.num_nodes
    expected_edges = batch1.num_edges + batch2.num_edges
    expected_edges += 2 * MyGraphStates.sf.num_edges

    # The actual count might be higher due to padding with sink states
    assert state1.tensor.num_nodes == expected_nodes
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
    data = datas[0]
    batch = GeometricBatch.from_data_list([data])
    batch.batch_shape = (1,)
    states = MyGraphStates(batch)

    # Get forward masks
    masks = states.forward_masks

    # Check action type mask
    assert masks["action_type"].shape == (1, 3)
    assert masks["action_type"][0, GraphActionType.ADD_NODE].item()  # Can add node
    assert (
        masks["action_type"][0, GraphActionType.ADD_EDGE]
    ).item()  # Can add edge (2 nodes)
    assert masks["action_type"][0, GraphActionType.EXIT].item()  # Can exit

    # Check features mask
    assert masks["node_class"].shape == (1, states.num_node_classes)
    assert torch.all(masks["node_class"])

    # Check edge_class mask
    assert masks["edge_class"].shape == (1, states.num_edge_classes)
    assert torch.all(masks["edge_class"])

    # Check edge_index masks
    assert len(masks["edge_index"]) == 1  # 1 graph
    assert torch.all(masks["edge_index"][0] == torch.tensor([[False, True]]))


def test_backward_masks(datas):
    """Test backward_masks property"""
    # Create a graph with 2 nodes and 1 edge
    data = datas[0]
    batch = GeometricBatch.from_data_list([data])
    batch.batch_shape = (1,)
    states = MyGraphStates(batch)

    # Get backward masks
    masks = states.backward_masks

    # Check action type mask
    assert masks["action_type"].shape == (1, 3)
    assert masks["action_type"][0, GraphActionType.ADD_NODE].item()  # Can remove node
    assert masks["action_type"][0, GraphActionType.ADD_EDGE].item()  # Can remove edge
    assert not masks["action_type"][0, GraphActionType.EXIT].item()  # Can exit

    # Check node_class mask
    assert masks["node_class"].shape == (1, states.num_node_classes)
    assert torch.all(masks["node_class"])

    # Check edge_class mask
    assert masks["edge_class"].shape == (1, states.num_edge_classes)
    assert torch.all(masks["edge_class"])

    # Check edge_index masks
    assert len(masks["edge_index"]) == 1  # 1 graph
    assert torch.all(masks["edge_index"][0] == torch.tensor([[True, False]]))


def test_stack_1d(datas):
    """Test stacking GraphStates objects"""
    # Create two states
    batch1 = GeometricBatch.from_data_list(datas[0:2])
    batch1.batch_shape = (2,)
    state1 = MyGraphStates(batch1)

    batch2 = GeometricBatch.from_data_list(datas[2:4])
    batch2.batch_shape = (2,)
    state2 = MyGraphStates(batch2)

    # Stack the states
    stacked = MyGraphStates.stack([state1, state2])

    # Check the batch shape
    assert stacked.tensor.batch_shape == (2, 2)

    # Check the number of nodes and edges
    assert stacked.tensor.num_nodes == 8  # 4 states * 2 nodes
    assert stacked.tensor.num_edges == 4  # 4 states * 1 edge

    # Check the batch indices
    assert torch.equal(stacked.tensor.batch[:4], batch1.batch)
    assert torch.equal(stacked.tensor.batch[4:], batch2.batch + 2)

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
    batch1 = GeometricBatch.from_data_list(datas[:4])
    batch1.batch_shape = (2, 2)
    state1 = MyGraphStates(batch1)

    batch2 = GeometricBatch.from_data_list(datas[4:8])
    batch2.batch_shape = (2, 2)
    state2 = MyGraphStates(batch2)

    # Stack the states
    stacked = MyGraphStates.stack([state1, state2])

    # Check the batch shape
    assert stacked.tensor.batch_shape == (2, 2, 2)

    # Check the number of nodes and edges
    assert stacked.tensor.num_nodes == 16  # 8 states * 2 nodes
    assert stacked.tensor.num_edges == 8  # 8 states * 1 edge

    # Check the batch indices
    assert torch.equal(stacked.tensor.batch[:8], batch1.batch)
    assert torch.equal(stacked.tensor.batch[8:], batch2.batch + 4)

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
