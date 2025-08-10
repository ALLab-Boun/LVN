# Local Virtual Nodes for Alleviating Over-Squashing in GNNs

This is the official repository of the paper ["Local Virtual Nodes for Alleviating Over-Squashing in Graph Neural Networks"](https://github.com/ALLab-Boun/LVN).

## Setup

Essential requirements:
- Python 3.10.12
- PyTorch 2.3.1
- PyG 2.4.0
- Networkx 3.3
- Numpy 2.0
- Wandb 0.17.5

Check "requirements.txt" for all the dependencies.

After installing the dependencies (preferably with venv/uv), run:
```bash
pip install -e .
```

## Training

Here is an example training command:
```bash
train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "imdb-binary" \
    --dropout-linear False \
    --hidden-size 64 \
    --initial-ff True \
    --lr "1e-3" \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 5 \
    --num-selected-nodes 15 \
    --patience 100 \
    --virtual-directed True \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 7,7 \
    --save \
    --output-name benchmark_final_graph_directed
```

The results will be displayed on the output and written in folder output/benchmark_final_graph_directed.

Note that **num-selected-nodes** refers to **$n_s$** and **workers-per-node** refers to **$n_c$**.

## Reproducing the results

To reproduce the graph classification and node classification results in the paper, run the .sh files in scripts folder.

Due to random data splits and inherent dataset variance, results may vary but should closely match those reported in the paper.