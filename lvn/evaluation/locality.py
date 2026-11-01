"""Locality metrics comparing an input graph G with a transformed graph R(G).

For LVN-transformed graphs, every LVN copy is mapped back to its central node,
so both graphs are compared on the same node set.

- distance_distortion: ||D_G - D_R(G)||_F over pairs of non-central nodes
  (Barbero et al., 2024), plus a version normalised by ||D_G||_F.
- khop_jaccard: mean Jaccard similarity of k-hop neighbourhoods of
  non-central nodes in G vs. R(G), with LVN copies mapped back to their
  central node.
"""

import networkx as nx
import numpy as np
import torch


def lvn_node_mapping(data):
    """Map each node of an LVN-transformed graph to a node of the original graph.

    Returns a LongTensor `mapping` of size data.num_nodes where mapping[i] is the
    original node index of transformed node i (LVN copies map to their central node).
    """
    central_nodes = data.central_nodes.cpu()
    num_original = data.original_nodes.numel()
    keep = torch.ones(num_original, dtype=torch.bool)
    keep[central_nodes] = False
    non_central = torch.arange(num_original)[keep]

    virtual_mask = data.virtual_mask.cpu()
    mapping = torch.empty(virtual_mask.numel(), dtype=torch.long)
    mapping[~virtual_mask] = non_central
    # LVN groups are stored consecutively in the order of data.central_nodes.
    groups = data.virtual_node_groups.cpu()[virtual_mask]
    _, group_rank = torch.unique(groups, return_inverse=True)
    mapping[virtual_mask] = central_nodes[group_rank]
    return mapping


def _to_nx(edge_index, num_nodes, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())
    return G


def _distance_matrix(G, nodes, unreachable):
    index = {v: i for i, v in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), float(unreachable))
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        if src not in index:
            continue
        for dst, d in lengths.items():
            if dst in index:
                D[index[src], index[dst]] = d
    return D


def _collapsed_distances(G_r, mapping, nodes, unreachable):
    """Shortest-path distances in R(G) between original nodes, where an
    original central node is represented by the closest of its copies."""
    members = {}
    for i, v in enumerate(mapping.tolist()):
        members.setdefault(v, []).append(i)
    index = {v: i for i, v in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), float(unreachable))
    for v in nodes:
        lengths = nx.multi_source_dijkstra_path_length(G_r, members[v])
        for j, d in lengths.items():
            u = mapping[j].item()
            if u in index:
                D[index[v], index[u]] = min(D[index[v], index[u]], d)
    return D


def distance_distortion(
    edge_index_g,
    num_nodes_g,
    edge_index_r,
    mapping,
    nodes=None,
    directed=False,
):
    """||D_G - D_R(G)||_F on `nodes` (default: all original nodes present in R(G)).

    `mapping[i]` gives the original node of transformed node i; nodes of R(G)
    without an original counterpart (e.g. a global virtual node) should be
    mapped to -1. Unreachable pairs get distance num_nodes_g in both graphs.
    """
    mapping = torch.as_tensor(mapping).cpu()
    if nodes is None:
        nodes = sorted(set(mapping.tolist()) - {-1})
    G = _to_nx(edge_index_g.cpu(), num_nodes_g, directed)
    G_r = _to_nx(edge_index_r.cpu(), mapping.numel(), directed)
    D_g = _distance_matrix(G, nodes, num_nodes_g)
    D_r = _collapsed_distances(G_r, mapping, nodes, num_nodes_g)
    diff = np.linalg.norm(D_g - D_r)
    norm = np.linalg.norm(D_g)
    return {
        "distortion": float(diff),
        "distortion_normalized": float(diff / norm) if norm > 0 else 0.0,
        "max_abs_change": float(np.abs(D_g - D_r).max()) if len(nodes) else 0.0,
    }


def khop_jaccard(
    edge_index_g, num_nodes_g, edge_index_r, mapping, nodes, ks=(1, 2, 3), directed=False
):
    """Mean Jaccard similarity between k-hop neighbourhoods in G and R(G)
    (neighbourhoods in R(G) are mapped back to original node ids). For directed
    graphs the k-hop neighbourhood is the set of nodes reachable within k arcs."""
    mapping = torch.as_tensor(mapping).cpu()
    G = _to_nx(edge_index_g.cpu(), num_nodes_g, directed)
    G_r = _to_nx(edge_index_r.cpu(), mapping.numel(), directed)
    members = {}
    for i, v in enumerate(mapping.tolist()):
        members.setdefault(v, []).append(i)

    results = {}
    for k in ks:
        scores = []
        for v in nodes:
            n_g = set(nx.single_source_shortest_path_length(G, v, cutoff=k)) - {v}
            reach = nx.multi_source_dijkstra_path_length(G_r, members[v], cutoff=k)
            n_r = {mapping[j].item() for j in reach} - {v, -1}
            union = n_g | n_r
            scores.append(len(n_g & n_r) / len(union) if union else 1.0)
        results[f"jaccard_{k}hop"] = float(np.mean(scores)) if scores else 1.0
    return results


def lvn_locality(original_data, transformed_data, ks=(1, 2, 3), directed=False):
    """Convenience wrapper for LVN: metrics over non-central original nodes."""
    mapping = lvn_node_mapping(transformed_data)
    central = set(transformed_data.central_nodes.cpu().tolist())
    nodes = [v for v in range(original_data.num_nodes) if v not in central]
    out = distance_distortion(
        original_data.edge_index,
        original_data.num_nodes,
        transformed_data.edge_index,
        mapping,
        nodes=nodes,
        directed=directed,
    )
    out.update(
        khop_jaccard(
            original_data.edge_index,
            original_data.num_nodes,
            transformed_data.edge_index,
            mapping,
            nodes,
            ks,
            directed=directed,
        )
    )
    return out
