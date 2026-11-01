"""LVN = width expansion of the central nodes with tied weights, made usable by the LVN embeddings.

A GCN / GIN on LVN(G) equals a model on G in which every central node c carries n_c blocks of
width p (one per LVN): blocks update with the shared weights, messages into c are broadcast to
every block and messages out of c are the sum of its blocks. Without distinct LVN embeddings the
blocks stay identical at every layer (Lemma 2), also with the directed strategy.
"""
import networkx as nx
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import from_networkx, to_undirected

from lvn.evaluation.locality import lvn_node_mapping
from lvn.transforms import LocalVirtualNodes


@pytest.fixture(autouse=True)
def float64():
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def setup(n_c=3, n_s=4, seed=0, p=8, strategy="undirected"):
    torch.manual_seed(seed)
    G = nx.barabasi_albert_graph(25, 2, seed=seed)
    g = Data(x=torch.randn(25, p), edge_index=to_undirected(from_networkx(G).edge_index), num_nodes=25)
    lvn = LocalVirtualNodes(n_s, n_c=n_c, strategy=strategy)(g.clone())
    mapping = lvn_node_mapping(lvn)
    local = lvn.virtual_node_local_indices
    emb = torch.randn(n_c, p)  # LVN embeddings (add mode)
    x_lvn = lvn.x.clone()
    x_lvn[lvn.virtual_mask] += emb[local[lvn.virtual_mask]]
    return g, lvn, mapping, local, emb, x_lvn


def wide_state(g, lvn, mapping, local, h_lvn, n_c):
    """LVN node states in the wide-hub layout on G: [n, n_c, p] (other nodes use block 0)."""
    H = torch.zeros(g.num_nodes, n_c, h_lvn.size(1))
    for x in range(lvn.num_nodes):
        j = int(local[x]) if bool(lvn.virtual_mask[x]) else 0
        H[int(mapping[x]), j] = h_lvn[x]
    return H


def wide_gcn_layer(g, central, H, n_c, W, b):
    n = g.num_nodes
    is_c = torch.zeros(n, dtype=torch.bool)
    is_c[central] = True
    A = torch.zeros(n, n)
    A[g.edge_index[0], g.edge_index[1]] = 1.0
    deg = 1.0 + A @ torch.where(is_c, float(n_c), 1.0)
    out_msg = torch.where(is_c[:, None], H.sum(1), H[:, 0])
    agg = (A / torch.sqrt(deg[:, None] * deg[None, :])) @ (out_msg @ W.T)
    Z = (H @ W.T) / deg[:, None, None] + agg[:, None, :] + b
    Z[~is_c, 1:] = 0.0
    return Z


@pytest.mark.parametrize("n_c", [2, 3])
def test_gcn_on_lvn_equals_wide_hub_model(n_c):
    g, lvn, mapping, local, emb, x_lvn = setup(n_c=n_c)
    p = x_lvn.size(1)
    convs = [GCNConv(p, p) for _ in range(3)]
    h, H = x_lvn, wide_state(g, lvn, mapping, local, x_lvn, n_c)
    for i, conv in enumerate(convs):
        h = conv(h, lvn.edge_index)
        H = wide_gcn_layer(g, lvn.central_nodes, H, n_c, conv.lin.weight, conv.bias)
        if i < len(convs) - 1:
            h, H = torch.relu(h), torch.relu(H)
    assert torch.allclose(wide_state(g, lvn, mapping, local, h, n_c), H, atol=1e-10)


def test_gin_on_lvn_equals_wide_hub_model():
    n_c = 3
    g, lvn, mapping, local, emb, x_lvn = setup(n_c=n_c)
    p = x_lvn.size(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(p, p), torch.nn.ReLU(), torch.nn.Linear(p, p))
    h = GINConv(mlp)(x_lvn, lvn.edge_index)
    H = wide_state(g, lvn, mapping, local, x_lvn, n_c)
    n = g.num_nodes
    is_c = torch.zeros(n, dtype=torch.bool)
    is_c[lvn.central_nodes] = True
    A = torch.zeros(n, n)
    A[g.edge_index[0], g.edge_index[1]] = 1.0
    out_msg = torch.where(is_c[:, None], H.sum(1), H[:, 0])
    Z = mlp(H + (A @ out_msg)[:, None, :])
    Z[~is_c, 1:] = 0.0
    assert torch.allclose(wide_state(g, lvn, mapping, local, h, n_c), Z, atol=1e-10)


@pytest.mark.parametrize("strategy", ["undirected", "directed"])
def test_identical_embeddings_keep_lvns_identical(strategy):
    _, lvn, *_ = setup(strategy=strategy)
    p = lvn.x.size(1)
    x = lvn.x.clone()
    x[lvn.virtual_mask] += torch.randn(p)  # the same vector for every LVN
    h = x
    for conv in [GCNConv(p, p) for _ in range(3)]:
        h = torch.relu(conv(h, lvn.edge_index))
    groups = lvn.virtual_node_groups[lvn.virtual_mask]
    hv = h[lvn.virtual_mask]
    for gid in groups.unique():
        block = hv[groups == gid]
        assert torch.allclose(block, block[0].expand_as(block), atol=1e-10)


@pytest.mark.parametrize("strategy", ["undirected", "directed"])
def test_distinct_embeddings_make_lvns_distinct(strategy):
    _, lvn, _, _, _, x_lvn = setup(strategy=strategy)
    p = x_lvn.size(1)
    h = torch.relu(GCNConv(p, p)(x_lvn, lvn.edge_index))
    groups = lvn.virtual_node_groups[lvn.virtual_mask]
    hv = h[lvn.virtual_mask]
    for gid in groups.unique():
        block = hv[groups == gid]
        assert not torch.allclose(block[0], block[1])
