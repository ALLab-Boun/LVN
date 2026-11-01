"""Local Virtual Nodes (LVN): every selected central node c is replaced by a group of n_c
LVNs that inherit its neighbourhood (Section 4.2).

- "undirected": every LVN is connected to every neighbour of c. If two central nodes are
  adjacent, all pairs of their LVNs are connected. Shortest-path distances between the
  non-central nodes are preserved exactly (Proposition 1).
- "directed": every LVN receives messages from every neighbour of c, but each neighbour
  receives messages from exactly one LVN, assigned round-robin over the neighbours sorted
  by node index (pi_c(j_k) = (k - 1) mod n_c). Adjacent central nodes are connected as in
  the undirected strategy.

Collapsing each group back to its central node recovers the input graph.
"""

from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected

from .centrality import CRITERION_TO_FUNC, centrality_selection

EDGE_ORIG, EDGE_CENTRAL = 0, 1


class LocalVirtualNodes(BaseTransform):
    def __init__(
        self,
        num_selected_nodes: int,
        n_c: int = 2,
        strategy: str = "undirected",
        criterion: str = "degree",
        centrality_cache_path: Optional[str] = None,
        keep_edge_attr: bool = False,
        directed_input: bool = False,
    ):
        if strategy not in ("undirected", "directed"):
            raise ValueError(f"Unknown strategy: {strategy}")
        if directed_input and strategy != "undirected":
            raise ValueError("directed_input is supported for the undirected strategy")
        self.num_selected_nodes = num_selected_nodes
        self.n_c = n_c
        self.strategy = strategy
        self.criterion = criterion
        self.centrality_cache_path = centrality_cache_path
        # Remap edge attributes (e.g. bond types on Peptides): every edge copies the
        # attribute of the input edge it is derived from.
        self.keep_edge_attr = keep_edge_attr
        # Directed input graphs (Tree-NeighborsMatch, arcs child -> parent): LVNs inherit
        # every in- and out-arc of the central node; arcs are mapped one by one.
        self.directed_input = directed_input
        super().__init__()

    def _assign(self, neighbours):
        """Round-robin assignment of the sorted neighbours of a central node to its LVNs."""
        return {w: i % self.n_c for i, w in enumerate(sorted(neighbours))}

    def forward(self, data: Data) -> Data:
        return self(data)

    def __call__(self, data: Data) -> Data:
        if not self.directed_input and not is_undirected(data.edge_index, num_nodes=data.num_nodes):
            raise ValueError("The input graph must be undirected.")

        data = data.clone()
        device = data.edge_index.device
        num_nodes = data.num_nodes
        n_c = self.n_c
        is_node_prediction = "y" in data and data.y is not None and data.y.shape[0] == num_nodes

        k = min(self.num_selected_nodes, num_nodes)
        centralities = CRITERION_TO_FUNC[self.criterion](data, self.centrality_cache_path)
        _, central_nodes = centrality_selection(centralities, k, device)
        central_nodes = central_nodes.cpu()
        central_list = central_nodes.tolist()
        group_of = {c: g for g, c in enumerate(central_list)}

        # New node ids: non-central nodes keep their relative order, followed by the LVNs
        # (group-major).
        keep = torch.ones(num_nodes, dtype=torch.bool)
        keep[central_nodes] = False
        non_central = torch.arange(num_nodes)[keep]
        new_id = torch.full((num_nodes,), -1, dtype=torch.long)
        new_id[non_central] = torch.arange(non_central.numel())
        base = non_central.numel()

        def lvn_id(c, j):
            return base + group_of[c] * n_c + j

        src, dst = data.edge_index.cpu()
        once = torch.ones_like(src, dtype=torch.bool) if self.directed_input else src < dst
        pairs = torch.stack([src[once], dst[once]]).t().tolist()
        pair_edge_ids = torch.nonzero(once).flatten().tolist()

        assign = None
        if self.strategy == "directed":
            neighbours = {c: [] for c in central_list}
            for u, v in pairs:
                if u in neighbours:
                    neighbours[u].append(v)
                if v in neighbours:
                    neighbours[v].append(u)
            assign = {c: self._assign(nbrs) for c, nbrs in neighbours.items()}

        new_edges, types, origin, seen = [], [], [], set()
        arcs, arc_origin = [], []  # one-way arcs of the directed strategy

        def add(a, b, t, e):
            # e: index of the input edge this edge is derived from.
            key = (a, b) if self.directed_input else (min(a, b), max(a, b))
            if key in seen:
                return
            seen.add(key)
            new_edges.append((a, b))
            types.append(t)
            origin.append(e)

        for (u, v), e in zip(pairs, pair_edge_ids):
            u_c, v_c = u in group_of, v in group_of
            if not u_c and not v_c:
                add(new_id[u].item(), new_id[v].item(), EDGE_ORIG, e)
            elif u == v:
                # Self-loop of a central node (directed input only): one per LVN.
                for j in range(n_c):
                    add(lvn_id(u, j), lvn_id(u, j), EDGE_CENTRAL, e)
            elif self.strategy == "undirected" or (u_c and v_c):
                us = [lvn_id(u, j) for j in range(n_c)] if u_c else [new_id[u].item()]
                vs = [lvn_id(v, j) for j in range(n_c)] if v_c else [new_id[v].item()]
                for a in us:
                    for b in vs:
                        add(a, b, EDGE_CENTRAL, e)
            else:  # directed, exactly one endpoint is central
                c, w = (u, v) if u_c else (v, u)
                for j in range(n_c):
                    arcs.append((new_id[w].item(), lvn_id(c, j)))
                    arc_origin.append(e)
                arcs.append((lvn_id(c, assign[c][w]), new_id[w].item()))
                arc_origin.append(e)

        total = base + k * n_c
        edge_index = torch.tensor(new_edges, dtype=torch.long).reshape(-1, 2).t()
        edge_types = torch.tensor(types, dtype=torch.long)
        origin = torch.tensor(origin, dtype=torch.long)
        if not self.directed_input:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_types = torch.cat([edge_types, edge_types])
            origin = torch.cat([origin, origin])
        if arcs:
            edge_index = torch.cat([edge_index, torch.tensor(arcs, dtype=torch.long).t()], dim=1)
            edge_types = torch.cat([edge_types, torch.full((len(arcs),), EDGE_CENTRAL, dtype=torch.long)])
            origin = torch.cat([origin, torch.tensor(arc_origin, dtype=torch.long)])

        if self.keep_edge_attr and "edge_attr" in data and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.cpu()[origin].to(data.edge_attr.device)
        elif "edge_attr" in data:
            del data.edge_attr

        # Node attributes -------------------------------------------------------------
        virtual_mask = torch.zeros(total, dtype=torch.bool)
        virtual_mask[base:] = True
        local_idx = torch.zeros(total, dtype=torch.long)
        local_idx[base:] = torch.arange(n_c).repeat(k)
        # Groups: one per non-central node and one per LVN group.
        groups = torch.arange(total)
        groups[base:] = base + torch.arange(k).repeat_interleave(n_c)

        if "x" in data and data.x is not None:
            x = data.x
            data.x = torch.cat(
                [x[non_central.to(x.device)], x[central_nodes.to(x.device)].repeat_interleave(n_c, dim=0)]
            )
        if "root_mask" in data and data.root_mask is not None:
            # Tree-NeighborsMatch readout: every LVN of the root is a root node.
            m = data.root_mask
            data.root_mask = torch.cat(
                [m[non_central.to(m.device)], m[central_nodes.to(m.device)].repeat_interleave(n_c)]
            )
        if is_node_prediction:
            # Node-level predictions are made per group (an LVN group predicts for its central node).
            y = data.y
            data.y = torch.cat([y[non_central.to(y.device)], y[central_nodes.to(y.device)]])

        data.edge_index = edge_index.to(device)
        data.edge_types = edge_types.to(device)
        data.num_nodes = total
        data.virtual_mask = virtual_mask.to(device)
        data.virtual_node_local_indices = local_idx.to(device)
        data.virtual_node_groups = groups.to(device)
        # Original node of each group (node-level splits are drawn over the original nodes).
        data.group_orig = torch.cat([non_central, central_nodes]).to(device)
        data.central_nodes = central_nodes.to(device)
        data.original_nodes = torch.arange(num_nodes, device=device)
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_selected_nodes={self.num_selected_nodes}, "
            f"n_c={self.n_c}, strategy={self.strategy}, criterion={self.criterion})"
        )


def build_lvn_transform(args, centrality_cache_path=None):
    return LocalVirtualNodes(
        num_selected_nodes=args.num_selected_nodes,
        n_c=args.n_c,
        strategy=args.lvn_strategy,
        criterion=args.criterion,
        centrality_cache_path=centrality_cache_path,
        directed_input=args.dataset == "tree-neighbors",
    )
