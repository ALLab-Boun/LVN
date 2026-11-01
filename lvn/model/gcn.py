import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_max_pool, global_mean_pool

from .embedding import LVNEmbedding


class GNN(nn.Module):
    """GCN / GIN / GAT backbone with optional LVN embeddings.

    Graph-level tasks: mean over all nodes (LVNs included). Node-level tasks: the
    representations of the LVNs of a group are averaged into one prediction for the
    central node they replace (one output row per group)."""

    def __init__(self, args):
        super().__init__()
        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.dropout_linear = args.dropout_linear
        self.layer_norm = args.layer_norm
        self.residual = args.residual
        self.dataset = args.dataset
        self.root_readout = args.root_readout
        self.use_last_layer_ff = args.last_layer_ff
        self.use_initial_ff = args.initial_ff
        tree = self.dataset == "tree-neighbors"
        # Binary tasks use one logit (BCE); Tree-NeighborsMatch uses cross-entropy.
        logits = 1 if self.out_features <= 2 and not tree else self.out_features

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            inp = self.hidden_size if (args.initial_ff or i > 0) else self.in_features
            last = i == self.num_layers - 1
            out = logits if (last and not args.last_layer_ff) else self.hidden_size
            if args.conv_type == "gcn":
                conv = GCNConv(inp, out)
            elif args.conv_type == "gin":
                conv = GINConv(nn.Sequential(nn.Linear(inp, out), nn.BatchNorm1d(out), nn.ReLU(), nn.Linear(out, out)))
            elif args.conv_type == "gat":
                conv = GATConv(inp, out)  # single head
            else:
                raise ValueError(f"Unknown conv type: {args.conv_type}")
            self.conv_layers.append(conv)

        self.dropout_layer = nn.Dropout(args.dropout_rate) if args.dropout_rate > 0 else None
        if args.initial_ff:
            self.initial_ff = nn.Linear(self.in_features, self.hidden_size)
        if args.last_layer_ff:
            self.last_layer_ff = nn.Linear(self.hidden_size, logits, bias=False)
        if self.layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)])

        self.lvn_embedding = None
        if args.add_virtual_nodes and args.virtual_emb:
            self.lvn_embedding = LVNEmbedding(
                self.hidden_size, args.n_c, args.virtual_emb_method, args.virtual_emb_init_std
            )
        if tree:
            self.tree_encoder = TreeNeighborsEncoder(args.tree_neighbors_inp_size, self.hidden_size)

    def forward(self, graph, pool=True):
        x = graph.x
        if self.dataset == "tree-neighbors":
            x = self.tree_encoder(x)
        if self.use_initial_ff:
            x = self.initial_ff(x)
        if self.lvn_embedding is not None:
            x = self.lvn_embedding(x, graph.virtual_node_local_indices, graph.virtual_mask)
        if self.dropout_layer is not None and self.dropout_linear:
            x = self.dropout_layer(x)

        for index, conv in enumerate(self.conv_layers):
            new_x = conv(x, graph.edge_index)
            if index < self.num_layers - 1:
                new_x = F.relu(new_x)
                if self.dropout_layer is not None:
                    new_x = self.dropout_layer(new_x)
            x = x + new_x if self.residual else new_x
            if self.layer_norm:
                x = self.layer_norms[index](x)

        if "root_mask" in graph:
            # Tree-NeighborsMatch: read out the root (the LVNs of the root if it is replaced).
            batch = graph.batch if "batch" in graph and graph.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pool_fn = global_max_pool if self.root_readout == "max" else global_mean_pool
            x = pool_fn(x[graph.root_mask], batch[graph.root_mask])

        if self.use_last_layer_ff:
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            x = self.last_layer_ff(x)

        if not pool and "virtual_node_groups" in graph and "root_mask" not in graph:
            # Node-level: one prediction per group.
            x = global_mean_pool(x, graph.virtual_node_groups)

        if pool and "root_mask" not in graph:
            x = global_mean_pool(x, graph.batch)
        return x


class TreeNeighborsEncoder(nn.Module):
    """Embeds the (key, value) pair of every Tree-NeighborsMatch node (Alon & Yahav, 2021)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.keys = nn.Embedding(input_size + 1, hidden_size)
        self.values = nn.Embedding(input_size + 1, hidden_size)

    def forward(self, x):
        return self.keys(x[:, 0]) + self.values(x[:, 1])
