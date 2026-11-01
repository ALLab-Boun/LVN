from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("lvn")
def lvn_config(cfg):
    cfg.lvn = CN()
    cfg.lvn.enable = False
    cfg.lvn.strategy = "undirected"     # undirected | directed
    cfg.lvn.num_selected_nodes = 10     # n_s
    cfg.lvn.n_c = 2                     # LVNs per central node
    cfg.lvn.criterion = "comp_tree_6"   # see lvn.transforms.CRITERION_TO_FUNC
    cfg.lvn.embedding_method = "add"    # add | replace
    # Positional/structural encodings (RWSE, LapPE) computed on the input graph; the LVNs
    # inherit the encoding of their central node.
    cfg.lvn.pe_on_original = True
