import torch
from torch_geometric.datasets import Planetoid, TUDataset, WebKB, WikipediaNetwork

from .neighborsmatch import DictionaryLookupDataset

GRAPH_DATASETS = ("reddit-binary", "imdb-binary", "mutag", "enzymes", "proteins", "collab")
NODE_DATASETS = ("cora", "citeseer", "texas", "cornell", "wisconsin", "chameleon")

DATASET_TO_CLS = {
    **{name: TUDataset for name in GRAPH_DATASETS},
    "cora": Planetoid,
    "citeseer": Planetoid,
    "texas": WebKB,
    "cornell": WebKB,
    "wisconsin": WebKB,
    "chameleon": WikipediaNetwork,
    "tree-neighbors": DictionaryLookupDataset,
}

DATASET_TO_TYPE = {
    **{name: "graph" for name in GRAPH_DATASETS},
    **{name: "node" for name in NODE_DATASETS},
    "tree-neighbors": "graph",
}

_BINARY = ("reddit-binary", "imdb-binary", "mutag", "proteins")
DATASET_TO_LOSS = {
    name: torch.nn.BCEWithLogitsLoss() if name in _BINARY else torch.nn.CrossEntropyLoss()
    for name in DATASET_TO_CLS
}
