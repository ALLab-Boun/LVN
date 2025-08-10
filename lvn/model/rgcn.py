import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, HeteroConv, GCNConv, GATConv


class RGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.edge_types = args.edge_types

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                inp = self.in_features
            else:
                inp = self.hidden_size

            if i == self.num_layers - 1:
                if self.out_features <= 2:
                    out = 1
                else:
                    out = self.out_features
            else:
                out = self.hidden_size

            self.conv_layers.append(
                HeteroConv(
                    {edge_type: GCNConv(inp, out) for edge_type in self.edge_types}
                )
            )

        self.dropout_layer = nn.Dropout(args.dropout_rate)

    def forward(self, graph, pooling="mean", embed=False):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        if embed:
            layers = self.conv_layers[:-1]
        else:
            layers = self.conv_layers

        for index, layer in enumerate(layers):
            x_dict = layer(x_dict, edge_index_dict)
            if index < self.num_layers - 1:
                x_dict = {node_type: F.relu(x_dict[node_type]) for node_type in x_dict}
                x_dict = {
                    node_type: self.dropout_layer(x_dict[node_type])
                    for node_type in x_dict
                }

        if pooling:
            if pooling == "mean":
                x = global_mean_pool(x_dict["node"], graph["node"].batch)
                return x
            else:
                raise NotImplementedError

        return x_dict["node"]


class RGAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.edge_types = args.edge_types

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                inp = self.in_features
            else:
                inp = self.hidden_size

            if i == self.num_layers - 1:
                if self.out_features <= 2:
                    out = 1
                else:
                    out = self.out_features
            else:
                out = self.hidden_size

            self.conv_layers.append(
                HeteroConv(
                    {edge_type: GATConv(inp, out) for edge_type in self.edge_types}
                )
            )

        self.dropout_layer = nn.Dropout(args.dropout_rate)

    def forward(self, graph):
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict
        for index, layer in enumerate(self.conv_layers):
            x_dict = layer(x_dict, edge_index_dict)
            if index < self.num_layers - 1:
                x_dict = {node_type: F.elu(x_dict[node_type]) for node_type in x_dict}
                x_dict = {
                    node_type: self.dropout_layer(x_dict[node_type])
                    for node_type in x_dict
                }

        x = global_mean_pool(x_dict["node"], graph["node"].batch)
        return x
