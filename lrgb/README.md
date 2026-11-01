# LVN on the Long Range Graph Benchmark (Peptides)

The Peptides results were obtained with a GraphGym-based LRGB code base (the LRGB repository,
https://github.com/vijaydwivedi75/lrgb, with the GraphGPS modules) and the GCN configuration tuned
for LRGB by Tönshoff et al. (2023). This folder contains the LVN parts; the rest of the training
code is the unmodified LRGB/GraphGPS code.

## Contents

- `lvn_graphgym/config.py`: the `lvn` config group (`cfg.lvn.*`).
- `lvn_graphgym/transform.py`: the LVN pre-transform (the `lvn` package of this repository,
  with edge attributes copied to the derived edges).
- `lvn_graphgym/posenc_original.py`: RWSE / LapPE computed on the input graph; LVNs inherit the
  encoding of their central node.
- `lvn_graphgym/encoder.py`: the LVN embeddings (node encoder `lvn`).
- `configs/`: the selected settings for Peptides-func and Peptides-struct, undirected and
  directed LVN, with the GCN backbone (computation-tree centrality with L = 6, n_s = 10).

## Hooks in the LRGB code

1. Make `lvn` (this repository, `pip install -e .`) and `lvn_graphgym` importable and import
   `lvn_graphgym` in `main.py`, next to the other `graphgps` imports, so the config group and the
   encoder are registered.
2. Dataset loading (`graphgps/loader/master_loader.py`): when `cfg.lvn.enable`, apply
   `lvn_pre_transform(cfg)` to every graph before the positional encodings are precomputed, and
   compute them with `compute_posenc_stats_original` instead of `compute_posenc_stats` when
   `cfg.lvn.pe_on_original`:

   ```python
   from lvn_graphgym import compute_posenc_stats_original, lvn_pre_transform

   if cfg.lvn.enable:
       pre_transform_in_memory(dataset, lvn_pre_transform(cfg), show_progress=True)
   pe_fn = compute_posenc_stats_original if cfg.lvn.enable and cfg.lvn.pe_on_original else compute_posenc_stats
   pre_transform_in_memory(dataset, partial(pe_fn, pe_types=pe_enabled_list, is_undirected=True, cfg=cfg))
   ```
3. Model (`graphgps/network/custom_gnn.py`): after the feature encoder and before the
   message-passing layers, apply the LVN embeddings:

   ```python
   self.lvn_embedding = register.node_encoder_dict["lvn"](cfg.gnn.dim_inner) if cfg.lvn.enable else None
   ...
   batch = self.encoder(batch)
   if self.lvn_embedding is not None:
       batch = self.lvn_embedding(batch)
   ```

   The graph readout (`graph_pooling: mean`) averages over all nodes, LVNs included, as for the
   other benchmarks.

Each configuration is trained with seeds 0-4 (`seed 0` ... `seed 4` on the command line); mean ± standard deviation:

| | Peptides-func (test AP) | Peptides-struct (test MAE) |
|---|---|---|
| GCN (Tönshoff et al. configuration) | 0.6881 ± 0.0026 | 0.2471 ± 0.0012 |
| LVN (undirected) | 0.6810 ± 0.0091 | 0.2494 ± 0.0020 |
| LVN (directed) | 0.6903 ± 0.0016 | 0.2475 ± 0.0015 |

`posenc_original.py` calls `compute_posenc_stats` of the LRGB/GraphGPS code (MIT License).
