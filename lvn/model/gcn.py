import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    global_mean_pool,
    global_max_pool,
    global_sort_pool,
    global_add_pool,
)
import warnings

from .layer import (
    TrainableGroupPositionalEncoding,
    TrainableSeparateGroupPositionalEncoding,
)

# Suppress warnings from global_sort_pool
warnings.filterwarnings("ignore", message=".*global_sort_pool.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

POOLING_TO_FUNC = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "sort": global_sort_pool,
    "add": global_add_pool,
}


def group_edge_dropout_by_group_size(edge_index, node_to_group_size, dropout_rate=0.2):
    device = edge_index.device
    src = edge_index[0]
    src_group_sizes = node_to_group_size[src]

    # Slightly faster: direct boolean mask instead of torch.where
    trivial_mask = src_group_sizes == 1
    random_vals = torch.rand(edge_index.size(1), device=device)

    edge_mask = trivial_mask | (random_vals < (1 - dropout_rate))
    return edge_index[:, edge_mask]


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.dropout_linear = args.dropout_linear
        self.initial_ff = args.initial_ff
        self.virtual_emb = args.virtual_emb
        self.dataset = args.dataset
        self.layer_norm = args.layer_norm
        self.residual = args.residual
        self.add_last_layer_ff = args.last_layer_ff
        self.virtual_group_pooling = args.virtual_group_pooling
        self.virtual_dropout_rate = args.virtual_dropout_rate
        self.virtual_emb_type = args.virtual_emb_type
        if args.add_virtual_nodes:
            self.workers_per_node = args.workers_per_node[0]

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if args.initial_ff:
                inp = self.hidden_size
            else:
                if i == 0:
                    inp = self.in_features
                else:
                    inp = self.hidden_size

            if args.last_layer_ff:
                out = self.hidden_size
            else:
                if i == self.num_layers - 1:
                    if self.out_features <= 2:
                        out = 1
                    else:
                        out = self.out_features
                else:
                    out = self.hidden_size

            if args.conv_type == "gcn":
                conv_layer = GCNConv(inp, out)
            elif args.conv_type == "gin":
                conv_layer = GINConv(
                    nn.Sequential(
                        nn.Linear(inp, out),
                        nn.BatchNorm1d(out),
                        nn.ReLU(),
                        nn.Linear(out, out),
                    )
                )
            else:
                raise ValueError(f"Unknown conv type: {args.conv_type}")
            self.conv_layers.append(conv_layer)

        if args.dropout_rate > 0:
            self.dropout_layer = nn.Dropout(args.dropout_rate)
        if args.initial_ff:
            self.initial_ff = nn.Linear(self.in_features, self.hidden_size, bias=True)
        if args.last_layer_ff:
            self.last_layer_ff = nn.Linear(
                self.hidden_size, self.out_features, bias=False
            )
        if args.virtual_emb:
            if self.virtual_emb_type == "shared":
                self.pe_layer = TrainableGroupPositionalEncoding(
                    self.hidden_size,
                    self.workers_per_node,
                    args.virtual_emb_method,
                )
            elif self.virtual_emb_type == "separate":
                self.pe_layer = TrainableSeparateGroupPositionalEncoding(
                    self.hidden_size,
                    args.num_selected_nodes,
                    self.workers_per_node,
                    args.virtual_emb_method,
                )
        if self.layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(self.num_layers):
                self.layer_norms.append(nn.LayerNorm(self.hidden_size))

        self.x_pre_gnn = None

        self.args = args

    def pre_forward(self, graph):
        x = graph.x

        if self.initial_ff:
            x = self.initial_ff(x)
        if self.virtual_emb:
            if self.virtual_emb_type == "shared":
                x = self.pe_layer(
                    x, graph.virtual_node_local_indices, graph.virtual_mask
                )
            elif self.virtual_emb_type == "separate":
                x = self.pe_layer(
                    x,
                    graph.virtual_node_local_indices,
                    graph.virtual_mask,
                    graph.batch
                    if "batch" in graph
                    else torch.zeros(graph.num_nodes, device=graph.x.device),
                )

        self.x_pre_gnn = x

        if self.dropout_rate > 0 and self.dropout_linear:
            x = self.dropout_layer(x)

        return x

    def post_forward(self, graph, x, pool=True):
        original_edge_index = graph.edge_index.clone()
        for index, layer in enumerate(self.conv_layers):
            if self.virtual_dropout_rate > 0 and self.training:
                edge_index = group_edge_dropout_by_group_size(
                    original_edge_index,
                    graph.virtual_group_sizes,
                    self.virtual_dropout_rate,
                )
            else:
                edge_index = original_edge_index
            new_x = layer(x, edge_index)
            if index < self.num_layers - 1:
                new_x = F.relu(new_x)
                if self.dropout_rate > 0:
                    new_x = self.dropout_layer(new_x)

            if self.residual:
                x = x + new_x
            else:
                x = new_x

            if self.layer_norm:
                new_x = self.layer_norms[index](new_x)

        if self.add_last_layer_ff:
            if self.dropout_rate > 0:
                x = self.dropout_layer(x)
            x = self.last_layer_ff(x)

        if "virtual_node_groups" in graph and "train_mask" in graph:
            if self.virtual_group_pooling == "sort":
                x = POOLING_TO_FUNC["sort"](
                    x, graph.virtual_node_groups, self.workers_per_node - 1
                )
            else:
                x = POOLING_TO_FUNC[self.virtual_group_pooling](
                    x, graph.virtual_node_groups
                )

        if pool:
            x = global_mean_pool(x, graph.batch)

        return x

    def forward(self, graph, pool=True):
        x = self.pre_forward(graph)
        x = self.post_forward(graph, x, pool=pool)
        return x
