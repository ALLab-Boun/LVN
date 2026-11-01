# Local Virtual Nodes (LVN)

Official implementation of ["Local Virtual Nodes for Alleviating Over-Squashing in Graph Neural Networks"](https://arxiv.org/abs/2508.20597).

LVN replaces each of the n_s most central nodes of a graph with a group of n_c local virtual nodes
(LVNs) that inherit its neighbourhood, and gives the i-th LVN of every group a trainable embedding
e_i shared across groups. The transformation preserves the shortest-path distances between the
other nodes, and the distinct LVN embeddings let the LVNs of a group carry different
representations.

Two edge distribution strategies are implemented (`lvn/transforms/lvn.py`):

- **undirected**: every LVN is connected to every neighbour of its central node; the LVNs of two
  adjacent central nodes are all connected to each other.
- **directed**: every LVN receives messages from every neighbour of its central node, but each
  neighbour receives messages from exactly one LVN, assigned round-robin over the neighbours
  sorted by node index, pi_c(j_k) = (k - 1) mod n_c. Adjacent central nodes are connected as in
  the undirected strategy.

Central nodes are selected by degree, PageRank, label-propagation or computation-tree centrality
(`lvn/transforms/centrality.py`).

## Setup

```bash
pip install -r requirements.txt   # torch 2.3.1, torch_geometric 2.3.0
pip install -e .
pytest tests
```

## Training

```bash
train --dataset imdb-binary --conv-type gcn --num-layers 5 --hidden-size 64 \
      --add-virtual-nodes true --lvn-strategy undirected \
      --num-selected-nodes 3 --n-c 8 --criterion degree --virtual-emb-method replace \
      --n-runs 50 --output-dir output/imdb
```

`--num-selected-nodes` is n_s and `--n-c` is n_c. `--conv-type` selects GCN, GIN (use
`--last-layer-ff true`) or a single-head GAT. Run `r` uses the random split with seed
`--split-seed + r`, so configurations are compared on the same splits (graph classification:
80/10/10 of the graphs; node classification: 60/20/20 of the original nodes). Results (mean and 95%
confidence interval over the runs) are logged and written to `<output-dir>/eval.json`.

LVN-embedding ablations: `--virtual-emb false` (no embeddings), `--freeze-virtual-emb-after 0`
(embeddings fixed at their random initialisation), `--virtual-emb-init-std 0.01` (small
initialisation).

## Reproducing the results

`configs/` contains the validation-selected LVN settings of the paper, with the reported results:

- `configs/tu.json`: REDDIT-BINARY, IMDB-BINARY, MUTAG, ENZYMES, PROTEINS, COLLAB
  × GCN / GIN / GAT × undirected / directed (50 paired splits).
- `configs/node.json`: Cora, Citeseer, Texas, Cornell, Wisconsin, Chameleon
  × GCN / GIN / GAT × undirected / directed (50 paired splits).
- `configs/tree.json`: Tree-NeighborsMatch, depths 2–5 (training accuracy, 3 seeds).

```bash
reproduce configs/tu.json --only mutag --device cuda --output-dir output/tu
```

`--only` selects configurations by name; the summary compares the obtained and reported numbers.
The Peptides (LRGB) experiments use a GraphGym code base; see [`lrgb/`](lrgb/README.md).

## Citation

```bibtex
@article{karabulut2025local,
  title   = {Local Virtual Nodes for Alleviating Over-Squashing in Graph Neural Networks},
  author  = {Karabulut, Tu{\u{g}}rul Hasan and Bayta{\c{s}}, {\.I}nci M.},
  journal = {arXiv preprint arXiv:2508.20597},
  year    = {2025}
}
```
