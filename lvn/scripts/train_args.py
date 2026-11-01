from argparse import ArgumentParser

from ..data import DATASET_TO_CLS
from ..transforms import CRITERION_TO_FUNC


def str2bool(value):
    return str(value).lower() == "true"


def parse_arguments(args=None):
    p = ArgumentParser(description="Train a GNN with Local Virtual Nodes (LVN).")
    # Data and runs
    p.add_argument("--dataset", default="mutag", choices=DATASET_TO_CLS.keys())
    p.add_argument("--data-path", default="./datasets/", help="Dataset root (default: ./datasets/)")
    p.add_argument("--n-runs", type=int, default=1, help="Number of runs, one data split each (default: 1)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="Run r uses the split with seed split_seed + r, so configurations are compared on "
                        "identical splits; -1: unseeded (default: 0)")
    p.add_argument("--tree-depth", type=int, default=3, help="Tree-NeighborsMatch depth (default: 3)")
    p.add_argument("--device", default="cuda")
    # Backbone
    p.add_argument("--conv-type", default="gcn", choices=["gcn", "gin", "gat"])
    p.add_argument("--num-layers", type=int, default=5, help="Message-passing layers (default: 5)")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--initial-ff", type=str2bool, default=True, help="Linear input layer (default: true)")
    p.add_argument("--last-layer-ff", type=str2bool, default=False,
                   help="Linear output layer after the last message-passing layer (default: false)")
    p.add_argument("--dropout-rate", type=float, default=0.5)
    p.add_argument("--dropout-linear", type=str2bool, default=False,
                   help="Dropout after the input layer (default: false)")
    p.add_argument("--residual", type=str2bool, default=False)
    p.add_argument("--layer-norm", type=str2bool, default=False)
    p.add_argument("--root-readout", default="mean", choices=["mean", "max"],
                   help="Tree-NeighborsMatch: pooling over the LVNs of the root (default: mean)")
    # LVN
    p.add_argument("--add-virtual-nodes", type=str2bool, default=False, help="Use LVNs (default: false)")
    p.add_argument("--lvn-strategy", default="undirected", choices=["undirected", "directed"],
                   help="Edge distribution strategy (default: undirected)")
    p.add_argument("--num-selected-nodes", type=int, default=10, help="n_s, number of central nodes (default: 10)")
    p.add_argument("--n-c", type=int, default=2, help="n_c, number of LVNs per central node (default: 2)")
    p.add_argument("--criterion", default="degree", choices=CRITERION_TO_FUNC.keys(),
                   help="Centrality function for selecting the central nodes; comp_tree_L: computation-tree "
                        "size with L hops (default: degree)")
    p.add_argument("--virtual-emb", type=str2bool, default=True, help="Trainable LVN embeddings (default: true)")
    p.add_argument("--virtual-emb-method", default="add", choices=["add", "replace"])
    p.add_argument("--virtual-emb-init-std", type=float, default=None,
                   help="Std of the normal initialisation of the LVN embeddings (default: 1)")
    p.add_argument("--freeze-virtual-emb-after", type=int, default=-1,
                   help="Freeze the LVN embeddings after this many optimiser steps; 0: fixed at their random "
                        "initialisation, -1: never (default: -1)")
    # Optimisation
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-scheduler-patience", type=int, default=10)
    p.add_argument("--lr-scheduler-factor", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience in epochs (default: 50)")
    p.add_argument("--early-stopping-start", type=int, default=50,
                   help="No early stopping before this epoch (default: 50)")
    p.add_argument("--eval-freq", type=int, default=1)
    p.add_argument("--stopping-criterion", default="validation", choices=["validation", "train"])
    p.add_argument("--stop-at-perfect-train", type=str2bool, default=False,
                   help="With --stopping-criterion train: stop at 100%% train accuracy (default: false)")
    # Output
    p.add_argument("--output-dir", default=None, help="If set, write eval.json (and splits) here")
    return p.parse_args(args)
