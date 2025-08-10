from .gcn import GCN
from .rgcn import RGCN, RGAT

MODEL_TO_CLS = {
    "gcn": GCN,
    "rgcn": RGCN,
    "rgat": RGAT,
}

__all__ = ["GCN", "RGCN", "RGAT", "MODEL_TO_CLS"]
