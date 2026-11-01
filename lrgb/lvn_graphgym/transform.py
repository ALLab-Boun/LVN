from lvn.transforms import LocalVirtualNodes


def lvn_pre_transform(cfg):
    """LVN pre-transform for the dataset (applied once, before the PE statistics). Edge
    attributes (bond types) are copied to the edges derived from them."""
    return LocalVirtualNodes(
        num_selected_nodes=cfg.lvn.num_selected_nodes,
        n_c=cfg.lvn.n_c,
        strategy=cfg.lvn.strategy,
        criterion=cfg.lvn.criterion,
        keep_edge_attr=True,
    )
