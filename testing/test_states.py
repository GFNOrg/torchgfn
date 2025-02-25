import pytest
import torch
from gfn.states import GraphStates
from tensordict import TensorDict


class MyGraphStates(GraphStates):
    s0 = TensorDict({
        "node_feature": torch.tensor([[1.0], [2.0]]),
        "node_index": torch.tensor([0, 1]),
        "edge_feature": torch.tensor([[0.5]]),
        "edge_index": torch.tensor([[0, 1]]),
    })
    sf = TensorDict({
        "node_feature": torch.tensor([[3.0], [4.0]]),
        "node_index": torch.tensor([2, 3]),
        "edge_feature": torch.tensor([[0.7]]),
        "edge_index": torch.tensor([[2, 3]]),
    })

@pytest.fixture
def simple_graph_state():
    """Creates a simple graph state with 2 nodes and 1 edge"""
    tensor = TensorDict({
        "node_feature": torch.tensor([[1.0], [2.0]]),
        "node_index": torch.tensor([0, 1]),
        "edge_feature": torch.tensor([[0.5]]),
        "edge_index": torch.tensor([[0, 1]]),
        "batch_ptr": torch.tensor([0, 2]),
        "batch_shape": torch.tensor([1])
    })
    return MyGraphStates(tensor)

@pytest.fixture
def empty_graph_state():
    """Creates an empty graph state"""
    tensor = TensorDict({
        "node_feature": torch.tensor([]),
        "node_index": torch.tensor([]),
        "edge_feature": torch.tensor([]),
        "edge_index": torch.tensor([]).reshape(0, 2),
        "batch_ptr": torch.tensor([0]),
        "batch_shape": torch.tensor([0])
    })
    return MyGraphStates(tensor)

def test_extend_empty_state(empty_graph_state, simple_graph_state):
    """Test extending an empty state with a non-empty state"""
    empty_graph_state.extend(simple_graph_state)
    
    assert torch.equal(empty_graph_state.tensor["batch_shape"], simple_graph_state.tensor["batch_shape"])
    assert torch.equal(empty_graph_state.tensor["node_feature"], simple_graph_state.tensor["node_feature"])
    assert torch.equal(empty_graph_state.tensor["edge_index"], simple_graph_state.tensor["edge_index"])
    assert torch.equal(empty_graph_state.tensor["edge_feature"], simple_graph_state.tensor["edge_feature"])
    assert torch.equal(empty_graph_state.tensor["batch_ptr"], simple_graph_state.tensor["batch_ptr"])

def test_extend_1d_batch(simple_graph_state):
    """Test extending two 1D batch states"""
    other_state = simple_graph_state.clone()
    
    # The node indices should be different after extend
    original_node_indices = simple_graph_state.tensor["node_index"].clone()
    
    simple_graph_state.extend(other_state)
    
    assert simple_graph_state.tensor["batch_shape"][0] == 2
    assert len(simple_graph_state.tensor["node_feature"]) == 4
    assert len(simple_graph_state.tensor["edge_feature"]) == 2
    
    # Check that node indices were properly updated (should be unique)
    new_node_indices = simple_graph_state.tensor["node_index"]
    assert len(torch.unique(new_node_indices)) == len(new_node_indices)
    assert not torch.equal(new_node_indices[:2], new_node_indices[2:])

def test_extend_2d_batch():
    """Test extending two 2D batch states"""
    # Create 2D batch states (T=2, B=1)
    tensor1 = TensorDict({
        "node_feature": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "node_index": torch.tensor([0, 1, 2, 3]),
        "edge_feature": torch.tensor([[0.5], [0.6]]),
        "edge_index": torch.tensor([[0, 1], [2, 3]]),
        "batch_ptr": torch.tensor([0, 2, 4]),
        "batch_shape": torch.tensor([2, 1])
    })
    state1 = MyGraphStates(tensor1)

    # Create another state with different time length (T=3, B=1)
    tensor2 = TensorDict({
        "node_feature": torch.tensor([[5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]),
        "node_index": torch.tensor([4, 5, 6, 7, 8, 9]),
        "edge_feature": torch.tensor([[0.7], [0.8], [0.9]]),
        "edge_index": torch.tensor([[4, 5], [6, 7], [8, 9]]),
        "batch_ptr": torch.tensor([0, 2, 4, 6]),
        "batch_shape": torch.tensor([3, 1])
    })
    state2 = MyGraphStates(tensor2)

    state1.extend(state2)

    # Check final shape should be (max_len=3, B=2)
    assert torch.equal(state1.tensor["batch_shape"], torch.tensor([3, 2]))
    
    # Check that we have the correct number of nodes and edges
    # Each time step has 2 nodes and 1 edge, so for 3 time steps and 2 batches:
    expected_nodes = 3 * 2 * 2  # T * nodes_per_timestep * B
    expected_edges = 3 * 1 * 2  # T * edges_per_timestep * B
    assert len(state1.tensor["node_feature"]) == expected_nodes
    assert len(state1.tensor["edge_feature"]) == expected_edges

def test_extend_with_common_indices(simple_graph_state):
    """Test extending states with common node indices"""
    # Create a state with overlapping node indices
    tensor = TensorDict({
        "node_feature": torch.tensor([[3.0], [4.0]]),
        "node_index": torch.tensor([1, 2]),  # Note: index 1 overlaps
        "edge_feature": torch.tensor([[0.7]]),
        "edge_index": torch.tensor([[1, 2]]),
        "batch_ptr": torch.tensor([0, 2]),
        "batch_shape": torch.tensor([1])
    })
    other_state = MyGraphStates(tensor)

    simple_graph_state.extend(other_state)

    # Check that node indices are unique after extend
    assert len(torch.unique(simple_graph_state.tensor["node_index"])) == 4
    
    # Check that edge indices were properly updated
    edge_indices = simple_graph_state.tensor["edge_index"]
    assert torch.all(edge_indices >= 0)

def test_stack_1d_batch():
    """Test stacking multiple 1D batch states"""
    # Create first state
    tensor1 = TensorDict({
        "node_feature": torch.tensor([[1.0], [2.0]]),
        "node_index": torch.tensor([0, 1]),
        "edge_feature": torch.tensor([[0.5]]),
        "edge_index": torch.tensor([[0, 1]]),
        "batch_ptr": torch.tensor([0, 2]),
        "batch_shape": torch.tensor([1])
    })
    state1 = MyGraphStates(tensor1)

    # Create second state with different values
    tensor2 = TensorDict({
        "node_feature": torch.tensor([[3.0], [4.0]]),
        "node_index": torch.tensor([0, 5]),
        "edge_feature": torch.tensor([[0.7]]),
        "edge_index": torch.tensor([[2, 3]]),
        "batch_ptr": torch.tensor([0, 2]),
        "batch_shape": torch.tensor([1])
    })
    state2 = MyGraphStates(tensor2)

    # Stack the states
    stacked = MyGraphStates.stack([state1, state2])

    # Check the batch shape is correct (2, 1)
    assert torch.equal(stacked.tensor["batch_shape"], torch.tensor([2, 1]))
    
    # Check that node features are preserved and ordered correctly
    assert torch.equal(stacked.tensor["node_feature"], 
                      torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    
    # Check that edge features are preserved and ordered correctly
    assert torch.equal(stacked.tensor["edge_feature"], 
                      torch.tensor([[0.5], [0.7]]))
    
    # Check that node indices are unique
    assert len(torch.unique(stacked.tensor["node_index"])) == 4
    
    # Check batch pointers are correct
    assert torch.equal(stacked.tensor["batch_ptr"], torch.tensor([0, 2, 4]))

def test_stack_empty_states():
    """Test stacking empty states"""
    # Create empty state
    tensor = TensorDict({
        "node_feature": torch.tensor([]),
        "node_index": torch.tensor([]),
        "edge_feature": torch.tensor([]),
        "edge_index": torch.tensor([]).reshape(0, 2),
        "batch_ptr": torch.tensor([0]),
        "batch_shape": torch.tensor([0])
    })
    empty_state = MyGraphStates(tensor)

    # Stack multiple empty states
    stacked = MyGraphStates.stack([empty_state, empty_state])

    # Check the batch shape is correct (2, 0)
    assert torch.equal(stacked.tensor["batch_shape"], torch.tensor([2, 0]))
    
    # Check that tensors are empty
    assert stacked.tensor["node_feature"].numel() == 0
    assert stacked.tensor["edge_feature"].numel() == 0
    assert stacked.tensor["edge_index"].numel() == 0
    
    # Check batch pointers are correct
    assert torch.equal(stacked.tensor["batch_ptr"], torch.tensor([0]))
