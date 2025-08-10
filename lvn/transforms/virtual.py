from collections import Counter
from typing import Optional
from functools import partial
import os
import pickle

import torch
import networkx as nx
from networkx.algorithms.community import fast_label_propagation_communities

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph, to_networkx, is_undirected, to_undirected
from torch_geometric.nn import global_max_pool

from ..logging import logger


def comm_aware_degree(edge_index, num_nodes, node_to_comm):
    out_comm_degrees = torch.zeros(num_nodes, device=edge_index.device)
    for src, dst in edge_index.T.tolist():
        if node_to_comm[src] != node_to_comm[dst]:
            out_comm_degrees[src] += 1
            out_comm_degrees[dst] += 1

    return out_comm_degrees


def get_random_nodes(graph, num_selected_nodes, centrality_cache_path):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    return torch.topk(
        torch.randn(num_nodes, device=edge_index.device), num_selected_nodes
    )[1]


def compute_centralities(
    graph,
    centrality_cache_path,
    centrality_func=nx.degree_centrality,
):
    centralities = load_centralities(centrality_cache_path)
    if centralities is None:
        G = to_networkx(graph, to_undirected=True)
        centralities = centrality_func(G)
    else:
        logger.info("Loaded centralities for %s", centrality_cache_path)

    return centralities


def centrality_selection(
    centralities,
    num_selected_nodes,
    device,
):
    centralities_tensor = torch.tensor(list(centralities.values()), device=device)
    topk_values, topk_indices = torch.topk(centralities_tensor, num_selected_nodes)
    topk_nodes = [list(centralities.keys())[i] for i in topk_indices]
    return topk_values, torch.tensor(topk_nodes, device=centralities_tensor.device)


def label_prop(graph, centrality_cache_path):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    G = nx.Graph()
    G.add_edges_from(edge_index.cpu().numpy().T)
    communities = list(fast_label_propagation_communities(G, seed=42))
    node_to_comm = {}
    for index, com in enumerate(communities):
        for node in com:
            node_to_comm[node] = index

    out_comm_degrees = comm_aware_degree(edge_index, num_nodes, node_to_comm)

    max_degree_nodes = []
    for com in communities:
        max_degree_node = max(com, key=lambda x: out_comm_degrees[x])
        max_degree_nodes.append(max_degree_node)

    comm_aware_centralities = torch.zeros_like(
        out_comm_degrees, device=edge_index.device
    )
    comm_aware_centralities[max_degree_nodes] = out_comm_degrees[max_degree_nodes]

    # return dict
    return {node: comm_aware_centralities[node].item() for node in range(num_nodes)}


def load_centralities(path):
    if path and os.path.exists(path) and os.path.isfile(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


CRITERION_TO_FUNC = {
    # "random": get_random_nodes,
    "label_prop": label_prop,
    **{
        f"{centrality}": partial(
            compute_centralities,
            centrality_func=getattr(
                nx,
                f"{'pagerank' if centrality == 'pagerank' else centrality + '_centrality'}",
            ),
        )
        for centrality in ["degree", "closeness", "betweenness", "pagerank"]
    },
}


class AddVirtualNodes(BaseTransform):
    def __init__(
        self,
        num_selected_nodes: Optional[int] = None,
        workers_per_node: int = 2,
        replication_factor: int = 1,
        return_hetero_data: bool = False,
        criterion: str = "degree",
        centrality_cache_path: str = None,
        directed: bool = False,
    ):
        self.num_selected_nodes = num_selected_nodes
        self.workers_per_node = workers_per_node
        self.replication_factor = min(replication_factor, workers_per_node)
        self.return_hetero_data = return_hetero_data
        self.criterion = criterion
        self.centrality_cache_path = centrality_cache_path
        self.directed = directed
        super().__init__()

    def __call__(self, data: Data) -> Data:
        if not is_undirected(data.edge_index):
            raise ValueError("The input graph must be undirected.")

        data = data.clone()
        device = data.edge_index.device
        original_num_nodes = data.num_nodes
        is_node_prediction = "y" in data and (
            "train_mask" in data or data.y.shape[0] == original_num_nodes
        )
        num_selected_nodes = min(self.num_selected_nodes, data.num_nodes)
        centralities = CRITERION_TO_FUNC[self.criterion](
            data, self.centrality_cache_path
        )
        top_k_centrality_scores, central_nodes = centrality_selection(
            centralities,
            num_selected_nodes,
            device,
        )

        new_nodes = torch.arange(
            data.num_nodes,
            data.num_nodes + num_selected_nodes * self.workers_per_node,
            device=device,
        )

        non_central_nodes_mask = torch.ones(
            data.num_nodes, device=device, dtype=torch.bool
        )
        non_central_nodes_mask[central_nodes] = False
        index_shift = torch.cumsum(~non_central_nodes_mask, dim=0)
        data.original_nodes = torch.arange(data.num_nodes, device=device)
        data.central_nodes = central_nodes
        data.index_shift = index_shift

        num_selected_nodes = len(central_nodes)
        central_nodes_to_index = {
            cent_node: i for i, cent_node in enumerate(central_nodes.tolist())
        }

        central_edges_out_mask = torch.isin(data.edge_index[1], central_nodes)
        central_edges_out = data.edge_index[:, central_edges_out_mask]
        central_edges_out_copies = torch.repeat_interleave(
            central_edges_out, repeats=self.replication_factor, dim=1
        )
        central_edges_dest_indices = []
        central_edge_counter = Counter()
        for node in central_edges_out_copies[1].tolist():
            local_index = central_edge_counter[node] % self.workers_per_node
            global_index = (
                data.num_nodes
                + central_nodes_to_index[node] * self.workers_per_node
                + local_index
            )
            central_edges_dest_indices.append(global_index)
            central_edge_counter.update([node])

        central_edges_dest_indices = torch.tensor(central_edges_dest_indices)
        central_edges_out_copies[1] = central_edges_dest_indices
        central_edges_out = central_edges_out_copies

        if self.directed:
            central_edges_in_mask = torch.isin(data.edge_index[0], central_nodes)
            central_edges_in = data.edge_index[:, central_edges_in_mask]
            central_edges_in_copies = central_edges_in.clone()
            central_edges_src_indices = []
            central_edge_counter = Counter()
            for node in central_edges_in_copies[0].tolist():
                local_index = central_edge_counter[node] % self.workers_per_node
                global_index = (
                    data.num_nodes
                    + central_nodes_to_index[node] * self.workers_per_node
                    + local_index
                )
                central_edges_src_indices.append(global_index)
                central_edge_counter.update([node])

            central_edges_src_indices = torch.tensor(central_edges_src_indices)
            central_edges_in_copies[0] = central_edges_src_indices
            central_edges_in = central_edges_in_copies

            central_edges_out = torch.cat([central_edges_out, central_edges_in], dim=1)

        # virt_to_virt_edges = torch.tensor(
        #     [
        #         edge
        #         for virt_nodes in torch.split(new_nodes, self.workers_per_node)
        #         for edge in itertools.combinations(virt_nodes, 2)
        #     ],
        #     device=device,
        # ).T
        # virt_to_virt_edges = to_undirected(virt_to_virt_edges)

        central_edges_out_expanded = []
        for source, dest in central_edges_out.T.tolist():
            if dest in central_nodes_to_index:
                virt_index_start = (
                    data.num_nodes
                    + central_nodes_to_index[dest] * self.workers_per_node
                )
                for virt_index in range(
                    virt_index_start, virt_index_start + self.workers_per_node
                ):
                    central_edges_out_expanded.append([source, virt_index])
                    central_edges_out_expanded.append([virt_index, source])
            else:
                central_edges_out_expanded.append([source, dest])
        central_edges_out_expanded = torch.tensor(
            central_edges_out_expanded, device=device
        ).T

        if not self.directed:
            central_edges_out_expanded = to_undirected(central_edges_out_expanded)

        edge_types = torch.cat(
            [
                torch.zeros(data.edge_index.shape[1], device=device, dtype=torch.long),
                torch.ones(
                    central_edges_out_expanded.shape[1], device=device, dtype=torch.long
                ),
                # torch.full((virt_to_virt_edges.shape[1],), 2.0, device=device),
            ],
        )
        data.edge_index = torch.cat(
            [
                data.edge_index,
                central_edges_out_expanded,
                # virt_to_virt_edges,
            ],
            dim=1,
        )
        combined = torch.cat([data.edge_index, edge_types.unsqueeze(0)], dim=0)
        combined_unique = torch.unique(combined, dim=1)
        data.edge_index = combined_unique[:2]
        edge_types = combined_unique[2]

        etype_to_str = {
            0: "orig",
            1: "centr",
            2: "virt",
        }

        virtual_mask = torch.zeros(
            data.num_nodes + num_selected_nodes * self.workers_per_node,
            dtype=torch.bool,
            device=device,
        )
        virtual_node_groups = torch.arange(
            data.num_nodes + num_selected_nodes * self.workers_per_node, device=device
        )
        virtual_node_groups[data.num_nodes :] = (
            torch.arange(num_selected_nodes, device=device).repeat_interleave(
                self.workers_per_node, dim=0
            )
            + data.num_nodes
        )
        virtual_mask[data.num_nodes :] = True
        data.virtual_mask = virtual_mask
        data.virtual_node_groups = virtual_node_groups
        data.num_nodes = data.num_nodes + num_selected_nodes * self.workers_per_node

        if "x" in data:
            new_nodes_x = data.x[central_nodes]
            new_nodes_x = torch.repeat_interleave(
                new_nodes_x, repeats=self.workers_per_node, dim=0
            )
            new_x = torch.cat([data.x, new_nodes_x], dim=0)
            data.x = new_x

        if is_node_prediction:
            new_nodes_y = data.y[central_nodes]
            new_nodes_y = torch.repeat_interleave(
                new_nodes_y, repeats=self.workers_per_node
            )
            new_y = torch.cat([data.y, new_nodes_y], dim=0)
            data.y = new_y

            available_masks = [
                mask_name
                for mask_name in ["train_mask", "test_mask", "val_mask"]
                if mask_name in data
            ]
            for mask_name in available_masks:
                new_mask = data[mask_name][central_nodes]
                new_mask = torch.repeat_interleave(
                    new_mask, repeats=self.workers_per_node, dim=0
                )
                new_mask = torch.cat([data[mask_name], new_mask], dim=0)
                data[mask_name] = new_mask

        nodes_to_keep_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        nodes_to_keep_mask[central_nodes] = False

        nodes_to_keep = torch.arange(data.num_nodes, device=device)[nodes_to_keep_mask]
        edge_index, edge_types = subgraph(
            nodes_to_keep,
            data.edge_index,
            edge_types,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        data.edge_index = edge_index
        data.edge_types = edge_types
        data.virtual_mask = data.virtual_mask[nodes_to_keep]
        data.virtual_node_groups = data.virtual_node_groups[nodes_to_keep]
        data.virtual_node_groups = torch.unique(
            data.virtual_node_groups, return_inverse=True
        )[1]
        _, counts = torch.unique(data.virtual_node_groups, return_counts=True)
        arange = torch.arange(data.virtual_node_groups.size(0), device=device)
        data.virtual_node_local_indices = arange - torch.repeat_interleave(
            arange[counts.cumsum(0) - counts], counts
        )
        unique_groups, group_counts = torch.unique(
            data.virtual_node_groups, return_counts=True
        )
        virtual_group_sizes = torch.zeros(
            len(data.virtual_node_groups), dtype=torch.long, device=device
        )
        for group, size in zip(unique_groups, group_counts):
            mask = data.virtual_node_groups == group
            virtual_group_sizes[mask] = size
        data.virtual_group_sizes = virtual_group_sizes

        if "x" in data:
            data.x = data.x[nodes_to_keep]

        if is_node_prediction:
            data.y = data.y[nodes_to_keep]
            data.y = global_max_pool(data.y, data.virtual_node_groups)

            for mask_name in available_masks:
                data[mask_name] = data[mask_name][nodes_to_keep]
                data[mask_name] = global_max_pool(
                    data[mask_name].float(), data.virtual_node_groups
                ).bool()

        data.num_nodes = nodes_to_keep_mask.sum().item()

        if self.return_hetero_data:
            hetero_data = HeteroData()
            hetero_data["node"].x = data.x
            hetero_data["node"].num_nodes = data.num_nodes
            hetero_data["node"].virtual_mask = data.virtual_mask
            hetero_data["node"].virtual_node_groups = data.virtual_node_groups
            hetero_data.y = data.y

            for e_typ in [0, 1, 2]:
                hetero_data[
                    "node", etype_to_str[e_typ], "node"
                ].edge_index = data.edge_index[:, edge_types == e_typ]

            return hetero_data

        return data

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(num_selected_nodes={self.num_selected_nodes}, workers_per_node={self.workers_per_node}, replication_factor={self.replication_factor}, criterion={self.criterion})"""
