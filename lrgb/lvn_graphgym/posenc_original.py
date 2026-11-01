"""Positional encodings of the input graph for LVN-transformed graphs.

PE statistics (RWSE, LapPE, ...) are computed on the graph before the LVN transform instead of
on the transformed graph, so the transform does not change the encodings of the original
nodes; every LVN takes the encoding of its central node. The input graph is recovered by
collapsing every LVN group onto its central node.

compute_posenc_stats is the function of the LRGB / GraphGPS code base
(graphgps/transform/posenc_stats.py, MIT License); adjust the import to your checkout.
"""
import torch
from torch_geometric.data import Data

from graphgps.transform.posenc_stats import compute_posenc_stats


def original_index(data):
    """Input-graph node of every node of the LVN-transformed graph."""
    vm = data.virtual_mask.cpu().bool()
    central = data.central_nodes.cpu().long()
    num_original = data.original_nodes.numel()
    keep = torch.ones(num_original, dtype=torch.bool)
    keep[central] = False
    mapping = torch.empty(vm.numel(), dtype=torch.long)
    mapping[~vm] = torch.arange(num_original)[keep]
    if vm.any():
        _, rank = torch.unique(data.virtual_node_groups.cpu()[vm], return_inverse=True)
        mapping[vm] = central[rank]
    return mapping, num_original


def compute_posenc_stats_original(data, pe_types, is_undirected, cfg):
    if "virtual_mask" not in data or not bool(data.virtual_mask.any()):
        return compute_posenc_stats(data, pe_types=pe_types, is_undirected=is_undirected, cfg=cfg)
    mapping, n0 = original_index(data)
    src, dst = mapping[data.edge_index[0].cpu()], mapping[data.edge_index[1].cpu()]
    keep = src != dst
    edge_index = torch.unique(torch.stack([src[keep], dst[keep]]), dim=1)
    orig = compute_posenc_stats(Data(edge_index=edge_index, num_nodes=n0), pe_types=pe_types,
                                is_undirected=is_undirected, cfg=cfg)
    for key in (orig.keys() if callable(orig.keys) else orig.keys):
        if key in ("edge_index", "num_nodes"):
            continue
        value = getattr(orig, key)
        if torch.is_tensor(value) and value.size(0) == n0:
            value = value[mapping]  # LVNs inherit the encoding of their central node
        setattr(data, key, value)
    return data
