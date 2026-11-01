import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected

from .constants import DATASET_TO_CLS, DATASET_TO_TYPE
from .neighborsmatch import DictionaryLookupDataset


def load_graph_dataset(name, root, transform=None, tree_depth=3):
    if name == "tree-neighbors":
        dataset = DictionaryLookupDataset(depth=tree_depth, transform=transform)
    else:
        dataset = TUDataset(root=root, name=name.upper(), transform=transform)
    dataset = list(dataset)
    labels = torch.cat([data.y.view(-1) for data in dataset])
    binary = len(torch.unique(labels)) == 2
    for data in dataset:
        if binary:
            data.y = data.y.float()
        if data.x is None:
            # Social networks have no node features: a constant feature (as in FoSR, PANDA).
            data.x = torch.ones((data.num_nodes, 1))
    return dataset


def load_node_dataset(name, root, transform=None):
    cls = DATASET_TO_CLS[name]
    dataset = cls(root=root, name=name, transform=transform)
    dataset._data.edge_index, dataset._data.edge_attr = to_undirected(
        dataset._data.edge_index, dataset._data.edge_attr
    )
    return dataset


def load_dataset(name, root, transform=None, tree_depth=3):
    name = name.lower()
    if DATASET_TO_TYPE[name] == "graph":
        return load_graph_dataset(name, root, transform=transform, tree_depth=tree_depth)
    return load_node_dataset(name, root, transform=transform)
