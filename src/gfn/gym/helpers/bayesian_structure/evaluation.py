"""
The code is adapted from:
https://github.com/larslorch/dibs/blob/master/dibs/metrics.py
"""

import numpy as np
import torch
from sklearn import metrics
from torch_geometric.data import Data as GeometricData
from torch_geometric.utils import to_dense_adj
from tqdm.auto import trange

from gfn.gflownet import GFlowNet
from gfn.gym.bayesian_structure import BayesianStructure


def posterior_estimate(
    gflownet: GFlowNet,
    env: BayesianStructure,
    num_samples=1000,
    batch_size=100,
    verbose=True,
) -> torch.Tensor:
    """Get the posterior estimate of DAG-GFlowNet as a collection of graphs
    sampled from the GFlowNet.

    Args:
        gflownet: `GFlowNet` instance.
        env: `BayesianStructure` environment.
        rng: Optional random generator instance.
        num_samples: The number of samples in the posterior approximation.
        verbose: If True, display a progress bar for the sampling process.

    Returns:
        posterior: torch.Tensor with shape `(B, N, N)`, where `B` is the number of sample
            graphs in the posterior approximation, and `N` is the number of variables
            in a graph.
    """

    n_batches = num_samples // batch_size
    n_batches += 1 if num_samples % batch_size > 0 else 0
    samples = []
    for it in trange(n_batches, disable=(not verbose), desc="Posterior estimate"):
        # Sample a batch of graphs
        trajectories = gflownet.sample_trajectories(
            env,
            n=batch_size if it < n_batches - 1 else num_samples % batch_size,
            save_logprobs=False,
            epsilon=0.0,
        )
        samples.extend(
            [
                to_dense_adj(
                    state.tensor.edge_index, max_num_nodes=state.tensor.num_nodes
                )
                for state in trajectories.terminating_states
            ]
        )
    return torch.concat(samples, dim=0)


def expected_shd(posterior_samples: torch.Tensor, gt_graph: GeometricData) -> float:
    """Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.

    Args:
        posterior_samples: Samples from the posterior. The tensor must have size
            `(B, N, N)`, where `B` is the number of sample graphs from the posterior
            approximation, and `N` is the number of variables in the graphs.
        gt_graph: GeometricData instance representing the ground-truth graph.

    Returns:
        e_shd: The Expected SHD.
    """
    posterior_samples = posterior_samples.cpu()
    gt_graph_adj = to_dense_adj(
        gt_graph.edge_index, max_num_nodes=gt_graph.num_nodes  # pyright: ignore
    )

    # Compute the pairwise differences
    diff = (posterior_samples - gt_graph_adj).abs()
    diff = diff + diff.transpose(2, 1)

    # Ignore double edges
    diff = diff.clamp(max=1)
    shds = diff.sum((1, 2)) / 2

    return shds.mean().item()


def expected_edges(posterior_samples: torch.Tensor) -> float:
    """Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.

    Args:
        posterior_samples: Samples from the posterior. The tensor must have size
            `(B, N, N)`, where `B` is the number of sample graphs from the posterior
            approximation, and `N` is the number of variables in the graphs.

    Returns:
        e_edges: The expected number of edges.
    """
    num_edges = posterior_samples.sum((1, 2))
    return num_edges.mean().item()


def threshold_metrics(posterior_samples: torch.Tensor, gt_graph: GeometricData) -> dict:
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).

    Args:
        posterior_samples: Samples from the posterior. The tensor must have size
            `(B, N, N)`, where `B` is the number of sample graphs from the posterior
            approximation, and `N` is the number of variables in the graphs.
        gt_graph: GeometricData instance representing the ground-truth graph.

    Returns:
        A dictionary containing the following metrics:
            - False Positive Rate
            - True Positive Rate
            - Area Under the Receiver Operating Characteristic Curve
            - Precision
            - Recall
            - Area Under the Precision-Recall Curve
            - Average Precision
    """
    gt_graph_adj = to_dense_adj(
        gt_graph.edge_index, max_num_nodes=gt_graph.num_nodes  # pyright: ignore
    )
    posterior_samples_np = posterior_samples.cpu().numpy()
    gt_graph_adj_np = gt_graph_adj.cpu().numpy()

    # Expected marginal edge features
    p_edge = np.mean(posterior_samples_np, axis=0)
    p_edge_flat = p_edge.reshape(-1)
    gt_flat = gt_graph_adj_np.reshape(-1)

    # Threshold metrics
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "prc_auc": prc_auc,
        "ave_prec": ave_prec,
    }
