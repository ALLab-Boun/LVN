train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "label_prop" \
    --dataset "chameleon" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 30 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "add" \
    --workers-per-node 2,2 \
    --save \
    --output-name benchmark_final_node


train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "citeseer" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 25 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 8,8 \
    --save \
    --output-name benchmark_final_node


train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "pagerank" \
    --dataset "cora" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 45 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 3,3 \
    --save \
    --output-name benchmark_final_node


train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "cornell" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 30 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "replace" \
    --workers-per-node 8,8 \
    --save \
    --output-name benchmark_final_node


train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "pagerank" \
    --dataset "texas" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 15 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "add" \
    --workers-per-node 8,8 \
    --save \
    --output-name benchmark_final_node


train \
    --add-virtual-nodes True \
    --conv-type "gcn" \
    --criterion "degree" \
    --dataset "wisconsin" \
    --dropout-linear False \
    --epochs 5000 \
    --hidden-size 128 \
    --initial-ff True \
    --lr "1e-3" \
    --lr-scheduler-patience 25 \
    --model "gcn" \
    --n-runs 50 \
    --num-layers 4 \
    --num-selected-nodes 15 \
    --patience 100 \
    --virtual-directed False \
    --virtual-emb True \
    --virtual-emb-method "add" \
    --workers-per-node 7,7 \
    --save \
    --output-name benchmark_final_node