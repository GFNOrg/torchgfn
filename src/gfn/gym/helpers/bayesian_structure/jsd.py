"""
The code is adapted from:
https://github.com/GFNOrg/GFN_vs_HVI/blob/master/dags/dag_gflownet/utils/exhaustive.py
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import permutations
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pgmpy.estimators import ExhaustiveSearch
from pgmpy.utils.mathext import powerset
from scipy.special import logsumexp
from torch_geometric.data import Data as GeometricData
from tqdm import tqdm

from gfn.actions import GraphActions, GraphActionType
from gfn.estimators import Estimator
from gfn.gym.bayesian_structure import BayesianStructure
from gfn.gym.helpers.bayesian_structure.scores import BaseScore

# https://oeis.org/A003024
NUM_DAGS = [1, 1, 3, 25, 543, 29281, 3781503]


class GraphCollection:
    def __init__(self):
        self.edges, self.lengths = [], []
        self.mapping = defaultdict(int)
        self.mapping.default_factory = lambda: len(self.mapping)

    def append(self, graph: nx.DiGraph) -> None:
        self.edges.extend(
            [self.mapping[edge] for edge in graph.edges()]  # pyright: ignore
        )
        self.lengths.append(graph.number_of_edges())

    def freeze(self) -> "GraphCollection":
        self.edges = np.asarray(self.edges, dtype=np.int32)
        self.lengths = np.asarray(self.lengths, dtype=np.int32)
        self.mapping = [
            edge for (edge, _) in sorted(self.mapping.items(), key=lambda x: x[1])
        ]
        return self

    def is_frozen(self) -> bool:
        return isinstance(self.mapping, list)

    def to_dict(self, prefix: str | None = None) -> dict:
        prefix = f"{prefix}_" if (prefix is not None) else ""
        return {
            f"{prefix}edges": self.edges,
            f"{prefix}lengths": self.lengths,
            f"{prefix}mapping": self.mapping,
        }


@dataclass
class FullPosterior:
    log_probas: np.ndarray
    graphs: GraphCollection
    closures: GraphCollection
    markov: GraphCollection

    def to_dict(self) -> dict:
        # Ensure that "graphs" has been frozen
        if not self.graphs.is_frozen():
            raise ValueError('Graphs must be frozen. Call "graphs.freeze()".')

        offset, output = 0, dict()
        for length, log_prob in zip(self.graphs.lengths, self.log_probas):
            edges_indices = self.graphs.edges[offset : offset + length]
            edges = [self.graphs.mapping[idx] for idx in edges_indices]
            output[frozenset(edges)] = log_prob
            offset += length

        return output


def jensen_shannon_divergence(
    full_posterior: FullPosterior, posterior: FullPosterior
) -> float:
    # Convert to dictionaries to align distributions
    full_posterior_dict = full_posterior.to_dict()
    posterior_dict = posterior.to_dict()

    # Get an (arbitrary ordering of the graphs)
    graphs = list(full_posterior_dict.keys())
    graphs = sorted(graphs, key=len)

    # Get the two distributions aligned
    full_posterior_list, posterior_list = [], []
    for graph in graphs:
        full_posterior_list.append(full_posterior_dict[graph])
        posterior_list.append(posterior_dict[graph])
    full_posterior_arr = np.array(full_posterior_list, dtype=np.float_)
    posterior_arr = np.array(posterior_list, dtype=np.float_)

    # Compute the mean distribution
    mean = np.log(0.5) + np.logaddexp(full_posterior_arr, posterior_arr)

    # Compute the JSD
    KL_full_posterior = np.exp(full_posterior_arr) * (full_posterior_arr - mean)
    KL_posterior = np.exp(posterior_arr) * (posterior_arr - mean)
    return 0.5 * np.sum(KL_full_posterior + KL_posterior)


def get_full_posterior(
    scorer: BaseScore,
    data: pd.DataFrame,
    env: BayesianStructure,
    nodelist: list[str],
    verbose: bool = True,
) -> FullPosterior:
    estimator = ExhaustiveSearch(data, scoring_method=scorer, use_cache=False)

    log_probas = []
    graphs = GraphCollection()
    closures = GraphCollection()
    markov = GraphCollection()

    nx_graphs = list(estimator.all_dags())
    data_array = np.empty(len(nx_graphs), dtype=object)
    for i, graph in enumerate(nx_graphs):
        data_array[i] = nx_to_geometric_data(graph, env, nodelist)
    scores = scorer.state_evaluator(env.States(data_array))
    log_probas = scores.cpu().numpy()
    # Normalize the log-joint distribution to get the posterior
    log_probas -= cast(np.ndarray, logsumexp(log_probas))

    with tqdm(nx_graphs, total=NUM_DAGS[data.shape[1]], disable=(not verbose)) as pbar:
        for graph in pbar:  # Enumerate all possible DAGs
            graphs.append(graph)
            closures.append(nx.transitive_closure_dag(graph))
            markov.append(get_markov_blanket_graph(graph))

    return FullPosterior(
        log_probas=log_probas,
        graphs=graphs.freeze(),
        closures=closures.freeze(),
        markov=markov.freeze(),
    )


def get_gfn_exact_posterior(
    gfn_state_graph: nx.DiGraph, verbose: bool = True
) -> FullPosterior:
    # Get the source graph
    in_degrees = gfn_state_graph.in_degree(gfn_state_graph)
    source_graphs = [
        gfn_state_graph.nodes[node]["graph"]
        for node, in_degree in in_degrees
        if in_degree == 0
    ]
    assert len(source_graphs) == 1
    assert len(source_graphs[0].edges) == 0
    num_variables = len(source_graphs[0])

    log_probas = []
    graphs = GraphCollection()
    closures = GraphCollection()
    markov = GraphCollection()

    for node in tqdm(
        nx.topological_sort(gfn_state_graph),
        total=NUM_DAGS[num_variables],
        disable=(not verbose),
    ):
        graph = gfn_state_graph.nodes[node]["graph"]
        log_probas.append(gfn_state_graph.nodes[node]["terminal_log_flow"])

        graphs.append(graph)
        closures.append(nx.transitive_closure_dag(graph))
        markov.append(get_markov_blanket_graph(graph))

    # The log-posterior is already normalized
    log_probas = np.asarray(log_probas, dtype=np.float_)

    return FullPosterior(
        log_probas,
        graphs=graphs.freeze(),
        closures=closures.freeze(),
        markov=markov.freeze(),
    )


def posterior_exact(
    env: BayesianStructure,
    estimator: Estimator,
    nodelist: list[str],
    batch_size: int = 256,
) -> nx.DiGraph:
    gfn_cache = get_gflownet_cache(env, estimator, nodelist, batch_size)
    gfn_state_graph, source_state_graph = construct_state_dag_with_bfs(
        gfn_cache, nodelist
    )
    gfn_state_graph = push_source_flow_to_terminal_states(
        gfn_state_graph, source_state_graph
    )
    return gfn_state_graph


def get_gflownet_cache(
    env: BayesianStructure,
    estimator: Estimator,
    nodelist: list[str],
    batch_size: int = 256,
) -> dict[frozenset, np.ndarray]:
    """Cache the results of the GFlowNet for all the states.

    This function caches the log-probabilities for all the actions and for
    all the states of the GFlowNet.

    Parameters
    ----------
    env : BayesianStructure
        The Bayesian structure learning environment.
    estimator : Estimator
        The GFlowNet policy estimator.
    nodelist : list[str]
        List of node names to ensure consistent node encoding in adjacency matrices.
    batch_size : int, default=256
        Batch size for processing states through the GFlowNet.

    Returns
    -------
    cache : dict of (frozenset, np.ndarray)
        The cache of log-probabilities returned by the GFlowNet. The keys of
        the cache are the graphs (encoded as a frozenset of their edges), and
        the corresponding value is an array of size `(num_variables ** 2 + 1,)`
        containing the log-probabilities of all the actions in that state
        (including the "stop" action, at the last index).
    """
    cache = dict()

    node_names = {i: name for i, name in enumerate(nodelist)}

    graphs = all_dags(env, len(nodelist), nodelist=nodelist)
    batch_idx = 0
    while batch_idx < len(graphs):
        _graphs = graphs[batch_idx : batch_idx + batch_size]
        _bsz = len(_graphs)
        data_array = np.empty(_bsz, dtype=object)
        for i, graph in enumerate(_graphs):
            data_array[i] = graph

        state = env.States(data_array)
        with torch.no_grad():
            estimator_output = estimator(state)
            dist = estimator.to_probability_distribution(state, estimator_output)

        exit_log_prob = dist.dists[GraphActions.ACTION_TYPE_KEY].log_prob(
            torch.tensor(
                GraphActionType.EXIT, dtype=torch.long, device=env.device
            ).repeat(_bsz)
        )
        exit_log_prob = exit_log_prob.unsqueeze(1)

        add_edge_log_prob = dist.dists[GraphActions.ACTION_TYPE_KEY].log_prob(
            torch.tensor(
                GraphActionType.ADD_EDGE, dtype=torch.long, device=env.device
            ).repeat(_bsz)
        )
        edge_action_log_probs = []
        for edge_index_action in range(len(nodelist) ** 2):
            action_log_prob = dist.dists[GraphActions.EDGE_INDEX_KEY].log_prob(
                torch.tensor(
                    edge_index_action, dtype=torch.long, device=env.device
                ).repeat(_bsz)
            )
            edge_action_log_probs.append(
                (add_edge_log_prob + action_log_prob).unsqueeze(1)
            )
        edge_action_log_probs = torch.cat(edge_action_log_probs, dim=1)

        log_probs = torch.cat([edge_action_log_probs, exit_log_prob], dim=1)

        # TODO: check if no possible edge can be added,
        for graph, log_prob in zip(_graphs, log_probs.cpu().numpy()):
            edges = frozenset(
                (node_names[source.item()], node_names[target.item()])  # pyright: ignore
                for source, target in graph.edge_index.T  # pyright: ignore
            )
            cache[edges] = log_prob

        batch_idx += _bsz

    return cache


def construct_state_dag_with_bfs(
    gflownet_cache: dict[frozenset, np.ndarray],
    nodelist: list[str],
    source_graph: nx.DiGraph | None = None,
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Constructs the state-action space of the GFlowNet.

    This function performs Breadth-First Search on the GFlowNet state-action space
    starting from the source state, in order to construct a networkx.DiGraph object
    where each node is a GFlowNet state and each edge is labeled with the action
    and the log probability of taking that action. Each node is also labeled with
    the stop_action_log_flow which contains the probability of terminating at that state.

    Parameters
    ----------
    gflownet_cache : dict[frozenset, np.ndarray]
        The cache of log-probabilities returned by the GFlowNet.
    nodelist : list[str]
        The list of nodes.
    source_graph : nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space.
    source_graph : nx.DiGraph instance
        The graph representing the source state.
    """
    gfn_state_graph = nx.DiGraph()
    is_state_queued = defaultdict(bool)
    states_to_visit = deque()

    if source_graph is None:
        source_graph = nx.DiGraph()
        source_graph.add_nodes_from(nodelist)
    source_graph_key = frozenset(source_graph.edges)

    gfn_state_graph.add_node(source_graph_key, graph=source_graph)
    states_to_visit.append(source_graph)
    is_state_queued[source_graph_key] = True
    while len(states_to_visit) > 0:
        current_graph = states_to_visit.popleft()
        current_graph_key = frozenset(current_graph.edges)
        children = get_children(current_graph, gflownet_cache, nodelist)
        for child_graph, action, log_prob in children:
            if action is None:  # stop action
                # Encode the stop action as a node attribute
                gfn_state_graph.nodes[current_graph_key][
                    "stop_action_log_flow"
                ] = log_prob
            else:
                child_graph_key = frozenset(child_graph.edges)
                if child_graph_key not in gfn_state_graph:
                    gfn_state_graph.add_node(child_graph_key, graph=child_graph)
                gfn_state_graph.add_edge(
                    current_graph_key,
                    child_graph_key,
                    action=action,
                    log_prob_action=log_prob,
                )
                already_visited = is_state_queued[child_graph_key]
                if not already_visited:
                    states_to_visit.append(child_graph)
                    is_state_queued[child_graph_key] = True

    return gfn_state_graph, source_graph


def push_source_flow_to_terminal_states(
    gfn_state_graph: nx.DiGraph,
    source_state_graph: nx.DiGraph,
) -> nx.DiGraph:
    """Compute a hashable key for a graph.

    This function traverses the GFlowNet state-action space graph (DAG) in a
    topologically sorted order and "pushes" the log_flow from each node to
    its children according to the log_prob_action specified on the edges.
    The topological sort ensures that all the flow has "arrived" at a node
    before "moving" its flow to its children.

    Parameters
    ----------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space where each node represents one GFlowNet
        state and each edge represents one GFlowNet action.
    source_state_graph: nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space but now each node has an attribute
        named log_flow, which is -np.inf for non-terminal states and
        the marginal log probability for the terminal states.
    """
    # Initialize log_flow to be -np.inf (flow = 0) for all nodes
    nx.set_node_attributes(gfn_state_graph, -np.inf, "log_flow")  # pyright: ignore

    # Except initialize log_flow to be 0 (flow = 1) for source node
    source_node_key = frozenset(source_state_graph.edges)
    nx.set_node_attributes(gfn_state_graph, {source_node_key: 0}, "log_flow")

    # Push flow through sorted graph
    for state in nx.topological_sort(gfn_state_graph):
        current_node = gfn_state_graph.nodes[state]
        log_flow_incoming = current_node[
            "log_flow"
        ]  # log_flow is log probability of reaching this node starting from source node

        # Compute terminal_log_flow
        stop_action_log_flow = current_node[
            "stop_action_log_flow"
        ]  # probability of taking stop action from this node

        # terminal prob = incoming probability * p(stop action at this node)
        current_node["terminal_log_flow"] = log_flow_incoming + stop_action_log_flow

        # Push flow along edges to children
        edges = gfn_state_graph.edges(state, data=True)
        for _, child, edge_attr in edges:
            log_prob_action = edge_attr["log_prob_action"]
            existing_log_flow_child = gfn_state_graph.nodes[child]["log_flow"]
            updated_log_flow_child = np.logaddexp(
                existing_log_flow_child, log_flow_incoming + log_prob_action
            )
            nx.set_node_attributes(
                gfn_state_graph, {child: updated_log_flow_child}, "log_flow"
            )

    return gfn_state_graph


############################
##### Helper functions #####
############################


def get_markov_blanket(graph: nx.DiGraph, node: str) -> set[str]:
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))

    mb_nodes = parents | children
    for child in children:
        mb_nodes |= set(graph.predecessors(child))
    mb_nodes.discard(node)

    return mb_nodes


def get_markov_blanket_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Build an undirected graph where two nodes are connected if
    one node is in the Markov blanket of another.
    """

    def _s(node1: str, node2: str) -> tuple[str, str]:
        return (node2, node1) if (node1 > node2) else (node1, node2)

    # Make it a directed graph to control the order of nodes in each
    # edges, to avoid mapping the same edge to 2 entries in mapping.
    mb_graph = nx.DiGraph()
    mb_graph.add_nodes_from(graph.nodes)

    edges = set()
    for node in graph.nodes:
        edges |= set(_s(node, mb_node) for mb_node in get_markov_blanket(graph, node))
    mb_graph.add_edges_from(edges)

    return mb_graph


def all_dags(
    env: BayesianStructure,
    num_variables: int,
    nodelist: list[str] | None = None,
) -> list[GeometricData]:
    # Adapted from: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/ExhaustiveSearch.py
    if nodelist is None:
        nodelist = [str(i) for i in range(num_variables)]

    edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    all_graphs = powerset(edges)  # 2^(n*(n-1)) graphs

    graphs = []
    for graph_edges in all_graphs:
        graph = nx.DiGraph(graph_edges)  # TODO: use torch_geometric
        graph.add_nodes_from(nodelist)
        if nx.is_directed_acyclic_graph(graph):
            graphs.append(nx_to_geometric_data(graph, env, nodelist))
    return graphs


def nx_to_geometric_data(
    graph: nx.DiGraph, env: BayesianStructure, nodelist: list[str]
) -> GeometricData:
    node_indices = {node: i for i, node in enumerate(nodelist)}
    if graph.edges():
        edge_index = torch.tensor(
            [
                [node_indices[source], node_indices[target]]
                for source, target in graph.edges()
            ],
            dtype=torch.long,
            device=env.device,
        ).T
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=env.device)

    return GeometricData(
        x=env.s0.x,
        edge_attr=env.s0.edge_attr,
        edge_index=edge_index,
        device=env.device,
    )


def get_valid_actions(graph: nx.DiGraph) -> set[tuple[str, str]]:
    """Gets the list of valid actions.

    The valid actions correspond to directed edges that can be added to the
    current graph, such that adding any of those edges would still yield a
    DAG. In other words, those are edges that (1) are not already present in
    the graph, and (2) would not introduce a directed cycle.

    Parameters
    ----------
    graph : nx.DiGraph instance
        The current graph.

    Returns
    -------
    edges : set of tuples
        A set of directed edges, encoded as a tuple of nodes from `graph`,
        corresponding to the valid actions in the state `graph`.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph is not a valid DAG.")

    all_edges = set(permutations(graph.nodes, 2))
    edges_already_present = set(graph.edges())

    # Build the transitive closure of the transpose graph
    closure = nx.transitive_closure_dag(graph.reverse())
    edges_cycle = set(closure.edges())

    return all_edges - (edges_already_present | edges_cycle)


def get_children(
    graph: nx.DiGraph,
    gfn_cache: dict[frozenset, np.ndarray],
    nodelist: list[str],
) -> set[tuple[nx.DiGraph, tuple[str, str] | None, np.ndarray]]:
    """Gets all the children of a graph.

    This function returns a set of the next states, from a particular state
    `graph`, with its corresponding log-probability. Note that the set of
    children includes the stop action, encoded as a `None` action, for which
    the child graph is the same as the current graph.

    Parameters
    ----------
    graph : nx.DiGraph instance
        The current graph.

    gfn_cache : dict
        The cache of log-probabilities returned by the GFlowNet. See
        `dag_gflownet.utils.gflownet.get_gflownet_cache` for details.

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    Returns
    -------
    children : set of tuples
        The set of all the next state from the current graph, with their
        corresponding log-probability. Each child is represented as
        `(next_graph, action, log_prob)`, where `next_graph` is a nx.DiGraph
        instance, `action` is the edge added (as a tuple of nodes), and
        `log_prob` is the log-probability of this action. Not that the "stop"
        action is encoded as the action `None`.
    """
    node2idx = dict((node, idx) for (idx, node) in enumerate(nodelist))
    valid_actions = get_valid_actions(graph)
    num_variables = len(nodelist)

    log_pi = gfn_cache[frozenset(graph.edges())]
    children: set[tuple[nx.DiGraph, tuple[str, str] | None, np.ndarray]] = {
        (graph, None, log_pi[-1])
    }  # The stop action
    for source, target in valid_actions:
        action = node2idx[source] * num_variables + node2idx[target]
        next_graph = cast(nx.DiGraph, graph.copy())
        next_graph.add_edge(source, target)
        _log_prob = cast(np.ndarray, log_pi[action])
        children.add((next_graph, (source, target), _log_prob))

    return children
