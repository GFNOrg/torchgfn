import math
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import torch
from scipy.special import gammaln
from torch_geometric.utils import to_dense_adj

from gfn.gym.helpers.bayesian_structure.priors import BasePrior
from gfn.states import GraphStates


def logdet(matrix: torch.Tensor) -> torch.Tensor:
    # Compute log-determinant using torch.linalg.slogdet
    _, ld = torch.linalg.slogdet(matrix)
    return ld


class BaseScore(ABC):
    """Base class for the scorer.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """

    def __init__(self, data: pd.DataFrame, prior: BasePrior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_nodes = len(self.column_names)
        self.prior.num_variables = self.num_nodes

    @abstractmethod
    def state_evaluator(self, state: GraphStates) -> torch.Tensor:
        pass


class BGeScore(BaseScore):
    r"""
    BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one variable.

    prior : callable or BasePrior instance
        A callable that returns the log prior contribution given num_parents.

    mean_obs : np.ndarray, optional
        Mean parameter of the Normal prior over the mean μ. This array must
        have size (N,), where N is the number of variables. Default is 0.

    alpha_mu : float, default=1.
        Precision parameter for the Normal prior over the mean μ.

    alpha_w : float, optional
        Degrees of freedom for the Wishart prior over the precision matrix W.
        Must satisfy alpha_w > N - 1. Default is N + 2.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        prior: BasePrior,
        mean_obs: Optional[torch.Tensor] = None,
        alpha_mu: float = 1.0,
        alpha_w: Optional[float] = None,
    ):
        super().__init__(data, prior)
        self.num_nodes = len(self.column_names)
        if mean_obs is None:
            mean_obs = torch.zeros((self.num_nodes,), dtype=torch.float32)
        if alpha_w is None:
            alpha_w = self.num_nodes + 2.0
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        # Compute t = (α_mu * (α_w - N - 1)) / (α_mu + 1)
        self.t = (self.alpha_mu * (self.alpha_w - self.num_nodes - 1)) / (
            self.alpha_mu + 1
        )

        # Build the regularized matrix R
        T = self.t * torch.eye(self.num_nodes)
        data_array = torch.tensor(self.data.values, dtype=torch.float32)
        data_mean = torch.mean(data_array, dim=0, keepdim=True)
        data_centered = data_array - data_mean
        self.R = (
            T
            + data_centered.T @ data_centered
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * ((data_mean - self.mean_obs).T @ (data_mean - self.mean_obs))
        )

        # Precompute gamma function normalization term for 0,...,N-1 parents.
        all_parents = torch.arange(self.num_nodes)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(
                0.5
                * (self.num_samples + self.alpha_w - self.num_nodes + all_parents + 1)
            )
            - gammaln(0.5 * (self.alpha_w - self.num_nodes + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5
            * (self.alpha_w - self.num_nodes + 2 * all_parents + 1)
            * math.log(self.t)
        )

    def state_evaluator(self, states: GraphStates) -> torch.Tensor:
        """
        Evaluate the BGe score for the given states.
        Expecting state.tensor.to_data_list() to return a list of graph objects,
        each of which has attributes 'edge_index' and a method to convert the
        sparse representation to an adjacency matrix.
        """
        batch_size = states.batch_shape[0]
        scores = []

        for i in range(batch_size):
            graph = states[i].tensor
            # Convert the graph object to an adjacency matrix.
            # Here we assume a helper function 'to_dense_adj' is available.
            if graph.edge_index.shape[1] == 0:  # No edges
                scores.append(0.0)
                continue

            adj_matrix = to_dense_adj(
                graph.edge_index, max_num_nodes=len(self.column_names)
            ).squeeze(0)
            score = self._calculate_bge_score(adj_matrix)
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32, device=states.device)

    def _calculate_bge_score(self, adj_matrix: torch.Tensor) -> float:
        """
        Calculate the BGe score for a single graph represented by its adjacency matrix.
        The score is computed as the sum of local scores over all nodes.
        """
        total_score = 0.0

        _adj_matrix = torch.zeros_like(adj_matrix)
        for indices in adj_matrix.nonzero():
            source, target = indices.tolist()

            # Score before adding the edge
            parents_before = _adj_matrix[:, target].nonzero().flatten().tolist()
            local_score_before = self.local_score(target, parents_before)

            # Score after adding the edge
            parents_after = parents_before + [source]
            local_score_after = self.local_score(target, parents_after)

            _adj_matrix[source, target] = 1
            total_score += local_score_after - local_score_before

        return total_score

    def local_score(self, target: int, parents: list[int]) -> float:
        """
        Calculate the local BGe score.

        Args:
            target (int): The target node index.
            parents (list[int]): The indices of the parents of the target node.
        """
        num_parents = len(parents)
        # self.prior.num_variables = num_parents

        if num_parents > 0:
            variables = [target] + parents
            R_parents = self.R[parents, :][:, parents]
            R_all = self.R[variables, :][:, variables]
            # torch.linalg.slogdet returns (sign, logdet)
            _, logdet_Rp = torch.linalg.slogdet(R_parents)
            _, logdet_Rall = torch.linalg.slogdet(R_all)

            tmp_var = self.num_samples + self.alpha_w - self.num_nodes + num_parents
            log_term_r = 0.5 * (tmp_var) * logdet_Rp - 0.5 * (tmp_var + 1) * logdet_Rall
        else:
            log_term_r = (
                -0.5
                * (self.num_samples + self.alpha_w - self.num_nodes + 1)
                * torch.log(torch.abs(self.R[target, target]))
            )

        local_score = (
            self.log_gamma_term[num_parents].item()
            + log_term_r.item()
            + self.prior(num_parents)
        )

        return local_score
