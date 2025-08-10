import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import TUDataset
import networkx as nx

from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree, to_undirected, to_networkx
from tqdm import tqdm

from .constants import DATASET_TO_CLS, DATASET_TO_TYPE


def load_graph_dataset(name, root, transform=None, extra_feature=None):
    dataset = TUDataset(
        root=root,
        name=name.upper(),
        transform=transform,
    )
    dataset = list(dataset)
    all_labels = []
    for data in dataset:
        if data.y.dim() > 1:
            all_labels.extend(data.y.squeeze().tolist())
        else:
            all_labels.append(data.y.item())
    num_classes = len(torch.unique(torch.tensor(all_labels)))

    for data in dataset:
        if isinstance(data, Data):
            if num_classes == 2:
                data.y = data.y.float()
            if extra_feature == "ones":
                feat = torch.ones((data.num_nodes, 1)).to(data.edge_index.device)
            elif extra_feature == "degree":
                feat = (
                    degree(data.edge_index[0], data.num_nodes)
                    .to(data.edge_index.device)
                    .unsqueeze(-1)
                )
                feat = torch.log(feat + 1)
            else:
                feat = None

            x = data.x
            if x is None:
                if extra_feature is None:
                    x = torch.ones((data.num_nodes, 1)).to(data.edge_index.device)
                else:
                    x = feat
            elif extra_feature is not None:
                x = torch.cat([x, feat], dim=-1)
            data.x = x

        elif isinstance(data, HeteroData):
            data["node"].y = data["node"].y.float()
            if extra_feature == "ones":
                data["node"].x = torch.ones((data["node"].num_nodes, 1)).to(
                    data["node", "orig", "node"].edge_index.device
                )
            elif extra_feature == "degree":
                feats = []
                for edge_type in data.edge_types:
                    feat = (
                        degree(
                            data.edge_index_dict[edge_type][0], data["node"].num_nodes
                        )
                        .to(data.edge_index_dict[edge_type].device)
                        .unsqueeze(-1)
                    )
                    feats.append(feat)
                feat = sum(feats)
                feat = torch.log(feat + 1)
            else:
                feat = None

            if "x" not in data["node"]:
                if extra_feature is None:
                    x = torch.ones((data["node"].num_nodes, 1)).to(
                        data["node", "orig", "node"].edge_index.device
                    )
                else:
                    x = feat
                data["node"].x = x
            elif extra_feature is not None:
                x = torch.cat([data["node"].x, feat], dim=-1)
                data["node"].x = x
    return dataset


def load_node_dataset(name, root, transform, undirected=True, **kwargs):
    name = name.lower()

    constructor = DATASET_TO_CLS[name]
    args = dict(root=root, transform=transform, **kwargs)
    if "name" in constructor.__init__.__annotations__:
        args["name"] = name
    dataset = constructor(**args)
    # dataset._data.edge_index, dataset._data.edge_attr = remove_self_loops(
    #     dataset._data.edge_index, dataset._data.edge_attr
    # )
    if undirected:
        dataset._data.edge_index, dataset._data.edge_attr = to_undirected(
            dataset._data.edge_index, dataset._data.edge_attr
        )

    if dataset.num_classes == 2:
        dataset._data.y = dataset._data.y.to(torch.float32)

    return dataset


def load_dataset(
    name, root, transform=None, extra_feature=None, undirected=True, **kwargs
):
    name = name.lower()
    if DATASET_TO_TYPE[name] == "graph":
        return load_graph_dataset(
            name, root, transform=transform, extra_feature=extra_feature
        )
    elif DATASET_TO_TYPE[name] == "node":
        return load_node_dataset(
            name, root, transform=transform, undirected=undirected, **kwargs
        )

    raise ValueError(f"Unknown dataset type for {name}")


def get_dataset_stats(root):
    dataset_stats = {}
    for dataset_name in tqdm(DATASET_TO_CLS.keys()):
        if dataset_name == "tree-neighbors" or DATASET_TO_TYPE[dataset_name] != "graph":
            continue

        dataset = load_graph_dataset(dataset_name, root, extra_feature="degree")
        dataset_stats[dataset_name] = {"num_graphs": len(dataset)}

        max_degrees = []
        min_degrees = []
        average_degrees = []
        num_nodes = []
        num_edges = []
        diameters = []

        for graph in dataset:
            graph_nx = to_networkx(
                graph, to_undirected=True
            )  # Ensure undirected for diameter calculation
            degrees = degree(graph.edge_index[0], graph.num_nodes)
            max_degrees.append(max(degrees).item())
            min_degrees.append(min(degrees).item())
            average_degrees.append(torch.mean(degrees).item())
            num_nodes.append(graph.num_nodes)
            num_edges.append(graph.edge_index.shape[1])
            if nx.is_connected(graph_nx):
                diameters.append(nx.diameter(graph_nx))
            else:
                diameters.append(np.nan)

        dataset_stats[dataset_name]["avg_max_degree"] = np.mean(max_degrees)
        dataset_stats[dataset_name]["avg_min_degree"] = np.mean(min_degrees)
        dataset_stats[dataset_name]["avg_avg_degree"] = np.mean(average_degrees)
        dataset_stats[dataset_name]["avg_nodes"] = np.mean(num_nodes)
        dataset_stats[dataset_name]["avg_edges"] = np.mean(num_edges)
        dataset_stats[dataset_name]["avg_diameter"] = np.mean(
            [d for d in diameters if not np.isnan(d)]
        )

    df = pd.DataFrame(dataset_stats).T
    df.index.name = "Dataset"
    df = df.round(2)
    df.to_csv(os.path.join(root, "stats.csv"))
    return df
