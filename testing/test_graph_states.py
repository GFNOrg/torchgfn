import pytest
import torch
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from gfn.actions import GraphActionType
from gfn.states import GraphStates


class MyGraphStates(GraphStates):
    # Initial state: a graph with 2 nodes and 1 edge
    s0 = GeometricData(
        x=torch.tensor([[1.0], [2.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[0.5]]),
    )

    # Sink state: a graph with 2 nodes and 1 edge (different from s0)
    sf = GeometricData(
        x=torch.tensor([[3.0], [4.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[0.7]]),
    )


@pytest.fixture
def datas():
    """Creates a list of 10 GeometricData objects"""
    return [
        GeometricData(
            x=torch.tensor([[i], [i + 0.5]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[i * 0.1]]),
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


def test_extend_empty_state(empty_graph_state, simple_graph_state):
    """Test extending an empty state with a non-empty state"""
    empty_graph_state.extend(simple_graph_state)

    # Check that the empty state now has the same content as the simple state
    assert empty_graph_state.tensor.batch_shape == simple_graph_state.tensor.batch_shape
    assert torch.equal(empty_graph_state.tensor.x, simple_graph_state.tensor.x)
    assert torch.equal(
        empty_graph_state.tensor.edge_index, simple_graph_state.tensor.edge_index
    )
    assert torch.equal(
        empty_graph_state.tensor.edge_attr, simple_graph_state.tensor.edge_attr
    )
    assert torch.equal(empty_graph_state.tensor.batch, simple_graph_state.tensor.batch)


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
    expected_nodes = 3 * 2 * 4  # T * nodes_per_graph * B
    expected_edges = 3 * 1 * 4  # T * edges_per_graph * B

    # The actual count might be higher due to padding with sink states
    assert state1.tensor.num_nodes >= expected_nodes
    assert state1.tensor.num_edges >= expected_edges


def test_getitem(datas):
    """Test indexing into GraphStates"""
    # Create a batch with 3 graphs
    batch = GeometricBatch.from_data_list(datas[:3])
    batch.batch_shape = (3,)
    states = MyGraphStates(batch)

    # Get a single graph
    single_state = states[1]
    assert single_state.tensor.batch_shape == (1,)
    assert single_state.tensor.num_nodes == 2
    assert torch.allclose(single_state.tensor.x, datas[1].x)

    # Get multiple graphs
    multi_state = states[[0, 2]]
    assert multi_state.tensor.batch_shape == (2,)
    assert multi_state.tensor.num_nodes == 4
    assert torch.allclose(multi_state.tensor.get_example(0).x, datas[0].x)
    assert torch.allclose(multi_state.tensor.get_example(1).x, datas[2].x)


def test_clone(simple_graph_state):
    """Test cloning a GraphStates object"""
    cloned = simple_graph_state.clone()

    # Check that the clone has the same content
    assert cloned.tensor.batch_shape == simple_graph_state.tensor.batch_shape
    assert torch.equal(cloned.tensor.x, simple_graph_state.tensor.x)
    assert torch.equal(cloned.tensor.edge_index, simple_graph_state.tensor.edge_index)
    assert torch.equal(cloned.tensor.edge_attr, simple_graph_state.tensor.edge_attr)

    # Modify the clone and check that the original is unchanged
    cloned.tensor.x[0, 0] = 99.0
    assert cloned.tensor.x[0, 0] == 99.0
    assert simple_graph_state.tensor.x[0, 0] == 0.0


def test_is_initial_state(datas):
    """Test is_initial_state property"""
    # Create a batch with s0 and a different graph
    s0 = MyGraphStates.s0.clone()
    different = datas[9]
    batch = GeometricBatch.from_data_list([s0, different])
    batch.batch_shape = (2,)
    states = MyGraphStates(batch)

    # Check is_initial_state
    is_initial = states.is_initial_state
    assert is_initial[0].item()
    assert not is_initial[1].item()


def test_is_sink_state(datas):
    """Test is_sink_state property"""
    # Create a batch with sf and a different graph
    sf = MyGraphStates.sf.clone()
    different = datas[9]
    batch = GeometricBatch.from_data_list([sf, different])
    batch.batch_shape = (2,)
    states = MyGraphStates(batch)

    # Check is_sink_state
    is_sink = states.is_sink_state
    assert is_sink[0].item()
    assert not is_sink[1].item()


def test_from_batch_shape():
    """Test creating states from batch shape"""
    # Create states with initial state
    states = MyGraphStates.from_batch_shape((3,))
    assert states.tensor.batch_shape[0] == 3
    assert states.tensor.num_nodes == 6  # 3 graphs * 2 nodes per graph

    # Check all graphs are s0
    is_initial = states.is_initial_state
    assert torch.all(is_initial)

    # Create states with sink state
    sink_states = MyGraphStates.from_batch_shape((2,), sink=True)
    assert sink_states.tensor.batch_shape[0] == 2
    assert sink_states.tensor.num_nodes == 4  # 2 graphs * 2 nodes per graph

    # Check all graphs are sf
    is_sink = sink_states.is_sink_state
    assert torch.all(is_sink)


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
    assert masks["features"].shape == (1, 1)  # 1 feature dimension
    assert masks["features"][0, 0].item()  # All features allowed

    # Check edge_index masks
    assert len(masks["edge_index"]) == 1  # 1 graph
    assert torch.all(
        masks["edge_index"][0] == torch.tensor([[False, False], [True, False]])
    )


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
    assert masks["action_type"][0, GraphActionType.EXIT].item()  # Can exit

    # Check features mask
    assert masks["features"].shape == (1, 1)  # 1 feature dimension
    assert masks["features"][0, 0].item()  # All features allowed

    # Check edge_index masks
    assert len(masks["edge_index"]) == 1  # 1 graph
    assert torch.all(
        masks["edge_index"][0] == torch.tensor([[False, True], [False, False]])
    )


def test_stack(datas):
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
