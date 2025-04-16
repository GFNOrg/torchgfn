import torch


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
