"""Node-level splits are paired: for a given seed the same original nodes are used for training,
validation and testing with and without LVNs."""
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected

from lvn.training.node_level import create_split_masks, original_to_row
from lvn.transforms import LocalVirtualNodes

N = 60


def graphs():
    G = nx.barabasi_albert_graph(N, 2, seed=0)
    torch.manual_seed(0)
    g = Data(x=torch.randn(N, 8), y=torch.randint(0, 3, (N,)),
             edge_index=to_undirected(from_networkx(G).edge_index), num_nodes=N)
    return {
        "base": g,
        "undirected": LocalVirtualNodes(5, n_c=3, strategy="undirected")(g),
        "directed": LocalVirtualNodes(5, n_c=3, strategy="directed")(g),
    }


def split_nodes(g, seed):
    orig = g.group_orig if "group_orig" in g else torch.arange(N)
    return [sorted(orig[m].tolist()) for m in create_split_masks(g, N, seed)]


def test_splits_identical_with_and_without_lvns():
    gs = graphs()
    for seed in (0, 1, 7):
        ref = split_nodes(gs["base"], seed)
        assert sorted(sum(ref, [])) == list(range(N))
        for name, g in gs.items():
            assert split_nodes(g, seed) == ref, name


def test_rows_map_to_original_nodes():
    for name, g in graphs().items():
        rows = original_to_row(g, N)
        assert rows.unique().numel() == N, name
        if "group_orig" in g:
            assert torch.equal(g.group_orig[rows], torch.arange(N)), name
