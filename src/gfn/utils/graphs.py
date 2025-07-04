from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data


def get_edge_indices(
    n_nodes: int,
    is_directed: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the source and target node indices for the edges.

    Args:
        n_nodes: The number of nodes in the graph.
        is_directed: Whether the graph is directed.
        device: The device to run the computation on.

    Returns:
        A tuple of two tensors, the source and target node indices.
    """
    if is_directed:
        # Upper triangle.
        i_up, j_up = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)
        # Lower triangle.
        i_lo, j_lo = torch.tril_indices(n_nodes, n_nodes, offset=-1, device=device)

        ei0 = torch.cat([i_up, i_lo])  # Combine them
        ei1 = torch.cat([j_up, j_lo])
    else:
        ei0, ei1 = torch.triu_indices(n_nodes, n_nodes, offset=1, device=device)

    return ei0, ei1


class GeometricBatch(Batch):
    """A batch of graphs.

    This class extends `torch_geometric.data.Batch` to support extending a batch
    with another batch, and to support stacking a list of `Data` objects into a
    single batch.

    Attributes:
        tensor: The underlying `torch_geometric.data.Data` object.
        batch_shape: The shape of the batch.
        batch_ptrs: A tensor of pointers to the start of each graph in the batch.
    """

    def extend(self, other: GeometricBatch) -> None:
        """Extends the current batch with another batch.

        Args:
            other: The batch to extend with.
        """
        self_x, other_x = self.tensor.x, other.tensor.x
        self_edge_index, other_edge_index = (
            self.tensor.edge_index,
            other.tensor.edge_index,
        )
        self_edge_attr, other_edge_attr = self.tensor.edge_attr, other.tensor.edge_attr
        self_ptr, other_ptr = self.tensor.ptr, other.tensor.ptr
        self_batch, other_batch = self.tensor.batch, other.tensor.batch
        self_batch_ptrs, other_batch_ptrs = self.batch_ptrs, other.batch_ptrs
        _self_slice_dict, _other_slice_dict = (
            self.tensor._slice_dict,
            other.tensor._slice_dict,
        )
        _self_inc_edge_index, _other_inc_edge_index = (
            self.tensor._inc_dict["edge_index"],
            other.tensor._inc_dict["edge_index"],
        )

        # Update the batch shape and pointers
        if len(self.batch_shape) == 1:
            # Simple concatenation for 1D batch
            new_batch_shape = (self.batch_shape[0] + other.batch_shape[0],)
            self_nodes = self.tensor.num_nodes
            _slice_dict = {
                "x": torch.cat(
                    [
                        _self_slice_dict["x"],
                        _self_slice_dict["x"][-1] + _other_slice_dict["x"][1:],
                    ]
                ),
                "edge_index": torch.cat(
                    [
                        _self_slice_dict["edge_index"],
                        _self_slice_dict["edge_index"][-1]
                        + _other_slice_dict["edge_index"][1:],
                    ]
                ),
                "edge_attr": torch.cat(
                    [
                        _self_slice_dict["edge_attr"],
                        _self_slice_dict["edge_attr"][-1]
                        + _other_slice_dict["edge_attr"][1:],
                    ]
                ),
            }

            # Create the new batch
            self.tensor = GeometricBatch(
                x=torch.cat([self_x, other_x], dim=0),
                edge_index=torch.cat(
                    [self_edge_index, self_nodes + other_edge_index], dim=1
                ),
                edge_attr=torch.cat([self_edge_attr, other_edge_attr], dim=0),
                ptr=torch.cat([self_ptr, self_nodes + other_ptr[1:]], dim=0),
                batch=torch.cat([self_batch, len(self) + other_batch], dim=0),
            )
            self.tensor.batch_shape = new_batch_shape
            self.batch_ptrs = torch.cat(
                [self_batch_ptrs, self_batch_ptrs.numel() + other_batch_ptrs], dim=0
            )
            self.tensor._slice_dict = _slice_dict
            self.tensor._inc_dict = {
                "x": torch.zeros(self.tensor.num_graphs),
                "edge_index": torch.cat(
                    [
                        _self_inc_edge_index,
                        self_ptr[-1] + _other_inc_edge_index,
                    ]
                ),
                "edge_attr": torch.zeros(self.tensor.num_graphs),
            }

        else:
            # Handle the case where batch_shape is (T, B)
            # and we want to concatenate along the B dimension
            self_batch_shape, other_batch_shape = self.batch_shape, other.batch_shape
            assert len(self_batch_shape) == 2 and len(other_batch_shape) == 2
            max_len = max(self_batch_shape[0], other_batch_shape[0])

            # Extend both batches to the same length T with sink states if needed
            if self.batch_shape[0] < max_len:
                sink_states = self.make_sink_states_tensor(
                    (max_len - self_batch_shape[0], self_batch_shape[1])
                )
                self_nodes = self_x.size(0)
                self_x = torch.cat([self_x, sink_states.x], dim=0)
                self_edge_index = torch.cat(
                    [self_edge_index, self_nodes + sink_states.edge_index], dim=1
                )
                self_edge_attr = torch.cat(
                    [self_edge_attr, sink_states.edge_attr], dim=0
                )
                self_batch = torch.cat(
                    [self_batch, len(self) + sink_states.batch], dim=0
                )
                sink_states_batch_ptrs = torch.arange(
                    sink_states.num_graphs, device=self.device
                ).view(sink_states.batch_shape)
                self_batch_ptrs = torch.cat(
                    [self_batch_ptrs, len(self) + sink_states_batch_ptrs], dim=0
                )
                _self_slice_dict = {
                    attr: torch.cat(
                        [
                            _self_slice_dict[attr],
                            _self_slice_dict[attr][-1]
                            + sink_states._slice_dict[attr][1:],
                        ]
                    )
                    for attr in _self_slice_dict.keys()
                }
                _self_inc_edge_index = torch.cat(
                    [
                        _self_inc_edge_index,
                        self_ptr[-1].to(_self_inc_edge_index.device)
                        + sink_states._inc_dict["edge_index"],
                    ]
                )
                self_ptr = torch.cat([self_ptr, self_nodes + sink_states.ptr[1:]], dim=0)

            if other.batch_shape[0] < max_len:
                sink_states = other.make_sink_states_tensor(
                    (max_len - other_batch_shape[0], other_batch_shape[1])
                )
                other_nodes = other_x.size(0)
                other_x = torch.cat([other_x, sink_states.x], dim=0)
                other_edge_index = torch.cat(
                    [other_edge_index, other_nodes + sink_states.edge_index], dim=1
                )
                other_edge_attr = torch.cat(
                    [other_edge_attr, sink_states.edge_attr], dim=0
                )
                other_batch = torch.cat(
                    [other_batch, len(other) + sink_states.batch], dim=0
                )
                sink_states_batch_ptrs = torch.arange(
                    sink_states.num_graphs, device=self.device
                ).view(sink_states.batch_shape)
                other_batch_ptrs = torch.cat(
                    [other_batch_ptrs, len(other) + sink_states_batch_ptrs], dim=0
                )
                _other_slice_dict = {
                    attr: torch.cat(
                        [
                            _other_slice_dict[attr],
                            _other_slice_dict[attr][-1]
                            + sink_states._slice_dict[attr][1:],
                        ]
                    )
                    for attr in _other_slice_dict.keys()
                }
                _other_inc_edge_index = torch.cat(
                    [
                        _other_inc_edge_index,
                        other_ptr[-1].to(_other_inc_edge_index.device)
                        + sink_states._inc_dict["edge_index"],
                    ]
                )
                other_ptr = torch.cat(
                    [other_ptr, other_nodes + sink_states.ptr[1:]], dim=0
                )

            _slice_dict = {
                attr: torch.cat(
                    [
                        _self_slice_dict[attr],
                        _self_slice_dict[attr][-1] + _other_slice_dict[attr][1:],
                    ]
                )
                for attr in _self_slice_dict.keys()
            }

            self.tensor = GeometricBatch(
                x=torch.cat([self_x, other_x], dim=0),
                edge_index=torch.cat(
                    [self_edge_index, self_ptr[-1] + other_edge_index], dim=1
                ),
                edge_attr=torch.cat([self_edge_attr, other_edge_attr], dim=0),
                ptr=torch.cat([self_ptr, self_ptr[-1] + other_ptr[1:]], dim=0),
                batch=torch.cat([self_batch, (len(self_ptr) - 1) + other_batch], dim=0),
            )
            new_batch_shape = (max_len, self_batch_shape[1] + other_batch_shape[1])
            self.tensor.batch_shape = new_batch_shape
            new_batch_ptrs = torch.cat(
                [self_batch_ptrs, self_batch_ptrs.numel() + other_batch_ptrs], dim=1
            )
            self.batch_ptrs = new_batch_ptrs
            self.tensor._slice_dict = _slice_dict

            self.tensor._inc_dict = {
                "x": torch.zeros(self.tensor.num_graphs),
                "edge_index": torch.cat(
                    [
                        _self_inc_edge_index,
                        self_ptr[-1].to(_self_inc_edge_index.device)
                        + _other_inc_edge_index,
                    ]
                ),
                "edge_attr": torch.zeros(self.tensor.num_graphs),
            }

    @classmethod
    def stack(cls, data_list: list[Data]) -> GeometricBatch:
        """Stacks a list of `Data` objects into a single `GeometricBatch`.

        Args:
            data_list: A list of `Data` objects to stack.

        Returns:
            A new `GeometricBatch` containing the stacked graphs.
        """
        xs = []
        edge_indices = []
        edge_attrs = []
        ptrs = [torch.zeros([1], dtype=torch.long, device=data_list[0].device)]
        batches = []
        _slice_dict = {
            "x": [torch.tensor([0])],
            "edge_index": [torch.tensor([0])],
            "edge_attr": [torch.tensor([0])],
        }
        edge_index_inc = []
        offset = 0
        for state in data_list:
            xs.append(state.tensor.x)
            edge_attrs.append(state.tensor.edge_attr)
            edge_indices.append(state.tensor.edge_index + ptrs[-1][-1])
            edge_index_inc.append(
                state.tensor._inc_dict["edge_index"]
                + ptrs[-1][-1].to(state.tensor._inc_dict["edge_index"].device)
            )
            ptrs.append(state.tensor.ptr[1:] + ptrs[-1][-1])
            batches.append(state.tensor.batch + offset)
            offset += len(state)
            _slice_dict["x"].append(
                state.tensor._slice_dict["x"][1:] + _slice_dict["x"][-1][-1]
            )
            _slice_dict["edge_index"].append(
                state.tensor._slice_dict["edge_index"][1:]
                + _slice_dict["edge_index"][-1][-1]
            )
            _slice_dict["edge_attr"].append(
                state.tensor._slice_dict["edge_attr"][1:]
                + _slice_dict["edge_attr"][-1][-1]
            )

        # Create a new batch
        batch = cls(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_attr=torch.cat(edge_attrs, dim=0),
            ptr=torch.cat(ptrs, dim=0),
            batch=torch.cat(batches, dim=0),
        )
        batch._inc_dict = {
            "x": torch.zeros(batch.num_graphs),
            "edge_index": torch.cat(edge_index_inc, dim=0),
            "edge_attr": torch.zeros(batch.num_graphs),
        }
        batch._slice_dict = {
            "x": torch.cat(_slice_dict["x"], dim=0),
            "edge_index": torch.cat(_slice_dict["edge_index"], dim=0),
            "edge_attr": torch.cat(_slice_dict["edge_attr"], dim=0),
        }
        single_batch_shape = (
            data_list[0].batch_shape if hasattr(data_list[0], "batch_shape") else ()
        )
        batch.batch_shape = (len(data_list),) + single_batch_shape
        return batch


def data_share_storage(a: Data, b: Data) -> bool:
    """True ⇢ every tensor attribute in `a` points to the same storage in `b`.

    Args:
        a: The first Data object.
        b: The second Data object.

    Returns:
        True if every tensor attribute in `a` points to the same storage in `b`,
        False otherwise.
    """
    for key, ta in a:
        if not torch.is_tensor(ta):
            continue
        tb = getattr(b, key, None)
        if not (torch.is_tensor(tb) and ta.data_ptr() == tb.data_ptr()):
            return False
    return True
