"""Centrality functions used to select the central nodes C (Section 4.1).

Every function maps a graph to a dict {node: score}; the n_s nodes with the highest
scores are selected (centrality_selection).
"""

from functools import partial
import os
import pickle

import networkx as nx
import torch
from networkx.algorithms.community import fast_label_propagation_communities
from torch_geometric.utils import to_networkx


def load_centralities(path):
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def compute_centralities(graph, centrality_cache_path, centrality_func=nx.degree_centrality):
    centralities = load_centralities(centrality_cache_path)
    if centralities is None:
        centralities = centrality_func(to_networkx(graph, to_undirected=True))
    return centralities


def centrality_selection(centralities, num_selected_nodes, device):
    scores = torch.tensor(list(centralities.values()), device=device)
    top_values, top_indices = torch.topk(scores, num_selected_nodes)
    nodes = list(centralities.keys())
    return top_values, torch.tensor([nodes[i] for i in top_indices], device=device)


def _out_community_degree(edge_index, num_nodes, node_to_comm):
    degrees = torch.zeros(num_nodes, device=edge_index.device)
    for src, dst in edge_index.T.tolist():
        if node_to_comm[src] != node_to_comm[dst]:
            degrees[src] += 1
            degrees[dst] += 1
    return degrees


def label_prop(graph, centrality_cache_path):
    """Label propagation communities; in each community the node with the most edges to
    other communities is a candidate, scored by that out-community degree."""
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    G = nx.Graph()
    G.add_edges_from(edge_index.cpu().numpy().T)
    communities = list(fast_label_propagation_communities(G, seed=42))
    node_to_comm = {node: i for i, com in enumerate(communities) for node in com}
    out_degrees = _out_community_degree(edge_index, num_nodes, node_to_comm)
    winners = [max(com, key=lambda v: out_degrees[v]) for com in communities]
    scores = torch.zeros_like(out_degrees)
    scores[winners] = out_degrees[winners]
    return {node: scores[node].item() for node in range(num_nodes)}


def computation_tree_size(graph, centrality_cache_path, hops=4):
    """Size of each node's L-hop computation tree, ((A + I)^L 1)_v: the number of length-L
    walks (with self-loops, as in GCN) ending at v, i.e. how many messages v aggregates over
    L message-passing layers (Arnaiz-Rodriguez & Errica, 2026, Def. 4.1)."""
    n = graph.num_nodes
    edge_index = graph.edge_index.cpu()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    counts = torch.ones(n, dtype=torch.float64)
    for _ in range(hops):
        nxt = counts.clone()  # self-loop term
        nxt.index_add_(0, edge_index[1], counts[edge_index[0]])
        counts = nxt
    return {node: counts[node].item() for node in range(n)}


def root_proximity(graph, centrality_cache_path):
    """Tree-NeighborsMatch: minus the hop distance to the root (graph.root_mask), so the
    top n_s nodes are the top levels of the tree."""
    G = to_networkx(graph, to_undirected=True)
    roots = torch.nonzero(graph.root_mask).flatten().tolist()
    dist = nx.multi_source_dijkstra_path_length(G, roots)
    far = graph.num_nodes + 1
    return {node: -float(dist.get(node, far)) for node in range(graph.num_nodes)}


CRITERION_TO_FUNC = {
    "degree": partial(compute_centralities, centrality_func=nx.degree_centrality),
    "pagerank": partial(compute_centralities, centrality_func=nx.pagerank),
    "label_prop": label_prop,
    **{f"comp_tree_{L}": partial(computation_tree_size, hops=L) for L in range(1, 13)},
    "root_proximity": root_proximity,
}
