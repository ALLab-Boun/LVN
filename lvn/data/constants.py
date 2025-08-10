import torch
from torch_geometric.datasets import (
    TUDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)

DATASET_TO_CLS = {
    "reddit-binary": TUDataset,
    "imdb-binary": TUDataset,
    "mutag": TUDataset,
    "enzymes": TUDataset,
    "proteins": TUDataset,
    "collab": TUDataset,
    "cornell": WebKB,
    "texas": WebKB,
    "wisconsin": WebKB,
    "chameleon": WikipediaNetwork,
    "cora": Planetoid,
    "citeseer": Planetoid,
}


DATASET_TO_TYPE = {
    "reddit-binary": "graph",
    "imdb-binary": "graph",
    "mutag": "graph",
    "enzymes": "graph",
    "proteins": "graph",
    "collab": "graph",
    "cornell": "node",
    "texas": "node",
    "wisconsin": "node",
    "chameleon": "node",
    "cora": "node",
    "citeseer": "node",
}


DATASET_TO_LOSS = {
    "reddit-binary": torch.nn.BCEWithLogitsLoss(),
    "imdb-binary": torch.nn.BCEWithLogitsLoss(),
    "mutag": torch.nn.BCEWithLogitsLoss(),
    "enzymes": torch.nn.CrossEntropyLoss(),
    "proteins": torch.nn.BCEWithLogitsLoss(),
    "collab": torch.nn.CrossEntropyLoss(),
    "cornell": torch.nn.CrossEntropyLoss(),
    "texas": torch.nn.CrossEntropyLoss(),
    "wisconsin": torch.nn.CrossEntropyLoss(),
    "chameleon": torch.nn.CrossEntropyLoss(),
    "cora": torch.nn.CrossEntropyLoss(),
    "citeseer": torch.nn.CrossEntropyLoss(),
}


DATASET_TO_METRIC = {
    "reddit-binary": "acc",
    "imdb-binary": "acc",
    "mutag": "acc",
    "enzymes": "acc",
    "proteins": "acc",
    "collab": "acc",
    "cornell": "acc",
    "texas": "acc",
    "wisconsin": "acc",
    "chameleon": "acc",
    "cora": "acc",
    "citeseer": "acc",
}
