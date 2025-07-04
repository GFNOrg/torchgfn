from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Batch, Data

if TYPE_CHECKING:
    from gfn.states import GraphStates


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


def graph_states_share_storage(a: GraphStates, b: GraphStates) -> bool:
    """Helper function to check if two GraphStates objects share storage.

    Returns:
        True if *any* tensor storage is shared between the two GraphStates.
    """

    def _tensor_ptrs(g: Data) -> tuple[int, ...]:
        """Return the data_ptr() of every tensor field in the graph."""
        out: list[int] = []
        for key, t in g:
            if torch.is_tensor(t) and t.numel() > 0:  # ignore empty tensors.
                out.append(t.data_ptr())

        return tuple(out)

    ptrs_a = {  # hash-set for O(1) look-ups.
        ptr for g in a.data.flat for ptr in _tensor_ptrs(g)
    }

    for g in b.data.flat:
        for ptr in _tensor_ptrs(g):
            if ptr in ptrs_a:
                return True  # first hit confirms shared storage.
    return False


def from_edge_indices(
    ei0: int | torch.Tensor,
    ei1: int | torch.Tensor,
    n_nodes: int,
    is_directed: bool,
) -> int | torch.Tensor:
    """Return the index (or indices) corresponding to the provided edge(s).

    This is the inverse operation of :func:`get_edge_indices`.  Given the
    source- and target-node indices of one or several edges, this function
    returns the *position* of each edge in the enumeration produced by
    ``get_edge_indices`` for the same ``n_nodes``/``is_directed`` setting.

    The enumeration rules are the same as in ``get_edge_indices``:

    1. ``is_directed = False``  →  only the strict upper–triangular part
       (``i < j``) is enumerated using ``torch.triu_indices`` with
       ``offset=1``.
    2. ``is_directed = True``   →  the strict upper part is enumerated
       first, followed by the strict lower part (``i > j``).

    Parameters
    ----------
    ei0 / ei1
        Source- and target-node indices. They can be Python ``int`` or
        *matching-shape* tensors.  If undirected, the orientation is
        ignored (``(i, j)`` and ``(j, i)`` map to the same index).
    n_nodes
        Number of nodes in the graph.
    is_directed
        Whether the graph is directed.

    Returns
    -------
    int | torch.Tensor
        The position(s) of each edge in the ordering returned by
        ``get_edge_indices``.
    """

    # Convert to tensors for a unified implementation while preserving the
    # original return type.
    scalar_input = not torch.is_tensor(ei0) and not torch.is_tensor(ei1)

    if scalar_input:
        # Promote to tensor for easier math; keep device on CPU for ints.
        i = torch.tensor(ei0, dtype=torch.long)
        j = torch.tensor(ei1, dtype=torch.long)
    else:
        i = ei0 if torch.is_tensor(ei0) else torch.tensor(ei0, device=ei1.device)
        j = ei1 if torch.is_tensor(ei1) else torch.tensor(ei1, device=ei0.device)

    # Sanity checks
    if torch.any(i == j):
        raise ValueError("Self-loops (i == j) are not enumerated by get_edge_indices.")

    # Ensure both tensors share the same dtype/device.
    if i.dtype != torch.long:
        i = i.long()
    if j.dtype != torch.long:
        j = j.long()

    if is_directed:
        # Number of edges in the upper triangular part.
        n_up = n_nodes * (n_nodes - 1) // 2

        # Masks for upper and lower triangles.
        upper_mask = i < j
        lower_mask = j < i  # same as i > j & avoids computing again.

        # Allocate container for result.
        idx = torch.empty_like(i)

        # --- Upper triangle -------------------------------------------------
        if upper_mask.any():
            iu = i[upper_mask]
            ju = j[upper_mask]

            preceding = iu * (n_nodes - 1) - (iu * (iu - 1)) // 2
            offset = ju - iu - 1
            idx[upper_mask] = preceding + offset

        # --- Lower triangle -------------------------------------------------
        if lower_mask.any():
            il = i[lower_mask]
            jl = j[lower_mask]

            preceding = (il * (il - 1)) // 2  # number of edges before row *il*
            offset = jl  # j ranges 0..i-1
            idx[lower_mask] = n_up + preceding + offset

    else:  # undirected ------------------------------------------------------
        # Ensure orientation is i < j for each edge.
        swapped_mask = i > j
        if swapped_mask.any():
            i, j = torch.where(swapped_mask, j, i), torch.where(swapped_mask, i, j)

        preceding = i * (n_nodes - 1) - (i * (i - 1)) // 2
        offset = j - i - 1
        idx = preceding + offset

    # Return in the original type.
    if scalar_input:
        return int(idx.item())
    return idx


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
            # Restore batch pointers for the concatenated batch.
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


def hash_graph(data, directed: bool) -> str:
    """
    Hash a PyG `Data` object (edge_index, edge_attr, x).
    Produces the same hash for graphs that are element-wise identical.
    """

    def _hash_update(t: torch.Tensor | None, h: hashlib.blake2b):
        """Update the hash with the contents of a tensor or placeholder."""
        if torch.is_tensor(t):
            h.update(t.contiguous().view(-1).cpu().numpy().tobytes())
        else:
            h.update(b"\0")  # placeholder for None.

        return h

    h = hashlib.blake2b(digest_size=16)

    # First, sort edge index, to remove isomorphisms.
    t = getattr(data, "edge_index", None)

    if torch.is_tensor(t):
        # Sorts individual edges to have source nodes < target nodes.
        if not directed:
            # Loop over edges.
            for i in range(t.shape[1]):
                idx = torch.argsort(t[:, i])  # Undirected case!
                t[:, i] = t[idx, i]  # Undirected case!

        # Sorts edges such that the source nodes are in ascending order.
        idx_per_edge = torch.argsort(t[0, :])  # Directed & undirected case!
        t = t[:, idx_per_edge]
    h = _hash_update(t, h)

    # Apply idx_per_edge to sort edge attributes.
    t = getattr(data, "edge_attr", None)
    if torch.is_tensor(t):
        t = t[idx_per_edge, ...]  # Sort edge attributes by ascending source nodes.
    h = _hash_update(t, h)

    # Finally, hash node features.
    t = getattr(data, "x", None)
    h = _hash_update(t, h)

    return h.hexdigest()


def compare_data_objects(a: Data, b: Data) -> bool:
    """Compare two Data objects along the main fields."""
    for attr in ("edge_index", "edge_attr", "x"):
        ta, tb = getattr(a, attr, None), getattr(b, attr, None)

        # One has a tensor, the other doesn't.
        if torch.is_tensor(ta) != torch.is_tensor(tb):
            return False

        # Both are tensors → compare contents (includes shape & dtype).
        if torch.is_tensor(ta) and not torch.equal(ta, tb):  # type: ignore
            return False

    return True
