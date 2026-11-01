"""LVN for GraphGym-based LRGB code (Peptides experiments of the paper).

Importing this package registers the `lvn` config group and the `lvn` node encoder.
See lrgb/README.md for the three hooks to add to the training code.
"""
from . import config, encoder  # noqa: F401  (registration)
from .posenc_original import compute_posenc_stats_original
from .transform import lvn_pre_transform

__all__ = ["compute_posenc_stats_original", "lvn_pre_transform"]
