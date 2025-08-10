train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "collab" \
    --dropout-linear False \
    --hidden-size 64 \
    --initial-ff True \
    --lr "1e-3" \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 5 \
    --num-selected-nodes 5 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 8,8 \
    --save \
    --output-name benchmark_final_graph_directed



train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "enzymes" \
    --hidden-size 64 \
    --initial-ff True \
    --lr "1e-3" \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 5 \
    --num-selected-nodes 3 \
    --patience 100 \
    --virtual-directed True \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 8,8 \
    --save \
    --output-name benchmark_final_graph_directed


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

train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "pagerank" \
    --dataset "mutag" \
    --hidden-size 64 \
    --initial-ff True \
    --lr "1e-3" \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 5 \
    --num-selected-nodes 5 \
    --patience 100 \
    --virtual-directed True \
    --virtual-emb True \
    --virtual-emb-method "add" \
    --workers-per-node 2,2 \
    --save \
    --output-name benchmark_final_graph_directed

train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "pagerank" \
    --dataset "proteins" \
    --hidden-size 64 \
    --initial-ff True \
    --lr "1e-3" \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 5 \
    --num-selected-nodes 5 \
    --patience 100 \
    --virtual-directed True \
    --virtual-emb True \
    --virtual-emb-method "add" \
    --workers-per-node 6,6 \
    --save \
    --output-name benchmark_final_graph_directed


train \
    --add-virtual-nodes True \
    --batch-size 64 \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "reddit-binary" \
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
    --workers-per-node 9,9 \
    --save \
    --output-name benchmark_final_graph_directed