"""Structural properties of the LVN transform (lvn/transforms/lvn.py)."""

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected

from lvn.evaluation.locality import lvn_locality, lvn_node_mapping
from lvn.transforms import LocalVirtualNodes


def _graph(G, with_labels=False):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    data = Data(x=torch.randn(n, 3), edge_index=to_undirected(from_networkx(G).edge_index, num_nodes=n), num_nodes=n)
    if with_labels:
        data.y = torch.randint(0, 3, (n,))
    return data


def _adjacent_hubs():
    # Two adjacent hubs (0, 1), each with its own leaves: paths between leaves of different
    # hubs go through the hub-hub edge.
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edges_from((0, leaf) for leaf in range(2, 6))
    G.add_edges_from((1, leaf) for leaf in range(6, 10))
    return G


GRAPHS = {
    "adjacent_hubs": _adjacent_hubs(),
    "barbell": nx.barbell_graph(5, 3),
    "karate": nx.karate_club_graph(),
    "er": nx.erdos_renyi_graph(40, 0.12, seed=0),
    "ba": nx.barabasi_albert_graph(60, 2, seed=1),
}
SETTINGS = [(1, 2), (3, 3), (5, 4)]
STRATEGIES = ["undirected", "directed"]


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("n_s, n_c", SETTINGS)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_preserves_distances(name, n_s, n_c, strategy):
    """Distances between non-central nodes are unchanged (Proposition 1; directed distances
    for the directed strategy)."""
    data = _graph(GRAPHS[name])
    out = LocalVirtualNodes(n_s, n_c=n_c, strategy=strategy)(data)
    metrics = lvn_locality(data, out, directed=strategy == "directed")
    assert metrics["distortion"] == 0.0, metrics
    if strategy == "undirected":
        assert metrics["jaccard_1hop"] == 1.0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_adjacent_central_nodes_stay_connected(strategy):
    """The LVN groups of two adjacent central nodes are fully connected (second union of
    Eq. 8)."""
    data = _graph(_adjacent_hubs())
    n_c = 3
    out = LocalVirtualNodes(2, n_c=n_c, strategy=strategy)(data)
    groups, vm = out.virtual_node_groups, out.virtual_mask
    src, dst = out.edge_index
    cross = vm[src] & vm[dst] & (groups[src] != groups[dst])
    assert int(cross.sum()) == 2 * n_c * n_c  # all pairs, both directions
    G = nx.Graph(out.edge_index.t().tolist())
    assert nx.is_connected(G)


@pytest.mark.parametrize("name", list(GRAPHS))
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_quotient_recovers_input(name, strategy):
    """Collapsing every LVN group onto its central node gives back the input graph."""
    data = _graph(GRAPHS[name])
    out = LocalVirtualNodes(3, n_c=3, strategy=strategy)(data)
    mapping = lvn_node_mapping(out)
    s, d = mapping[out.edge_index[0]], mapping[out.edge_index[1]]
    keep = s != d
    quotient = {tuple(sorted(e)) for e in torch.stack([s[keep], d[keep]]).t().tolist()}
    assert quotient == {tuple(sorted(e)) for e in data.edge_index.t().tolist()}


def test_directed_each_neighbour_receives_from_one_lvn():
    data = _graph(nx.barabasi_albert_graph(40, 2, seed=3))
    n_c = 3
    out = LocalVirtualNodes(4, n_c=n_c, strategy="directed")(data)
    vm, groups = out.virtual_mask, out.virtual_node_groups
    src, dst = out.edge_index
    to_regular = vm[src] & ~vm[dst]
    for w in dst[to_regular].unique().tolist():
        senders = src[to_regular & (dst == w)]
        # one sender per adjacent group
        assert senders.numel() == groups[senders].unique().numel()
    from_regular = ~vm[src] & vm[dst]
    for g in groups[vm].unique().tolist():
        members = (vm & (groups == g)).nonzero().flatten().tolist()
        senders = [set(src[from_regular & (dst == m)].tolist()) for m in members]
        assert all(s == senders[0] for s in senders)  # every LVN receives the whole neighbourhood


def test_round_robin_assignment():
    t = LocalVirtualNodes(1, n_c=3, strategy="directed")
    assert t._assign([9, 4, 7, 1, 5]) == {1: 0, 4: 1, 5: 2, 7: 0, 9: 1}


def test_node_labels_and_groups():
    data = _graph(nx.karate_club_graph(), with_labels=True)
    out = LocalVirtualNodes(3, n_c=2)(data)
    assert torch.equal(out.y, data.y[out.group_orig])
    assert out.virtual_node_groups.unique().numel() == data.num_nodes
    assert int(out.virtual_mask.sum()) == 3 * 2


def test_edge_attr_remapping():
    data = _graph(nx.karate_club_graph())
    data.edge_attr = torch.arange(data.edge_index.size(1)).float().unsqueeze(-1)
    out = LocalVirtualNodes(3, n_c=2, keep_edge_attr=True)(data)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    mapping = lvn_node_mapping(out)
    lookup = {tuple(e): a for e, a in zip(data.edge_index.t().tolist(), data.edge_attr.flatten().tolist())}
    for (s, d), a in zip(out.edge_index.t().tolist(), out.edge_attr.flatten().tolist()):
        u, v = int(mapping[s]), int(mapping[d])
        assert a in (lookup[(u, v)], lookup[(v, u)])
