from copy import deepcopy

import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from tqdm import tqdm
from collections import defaultdict

from ..data import load_dataset, DATASET_TO_LOSS, DATASET_TO_METRIC, DATASET_TO_CLS
from ..evaluation import evaluate
from ..transforms import AddVirtualNodes
from ..logging import logger
from .util import auxiliary_virtual_emb_loss


def preprocess_batched_groups(node_groups, batch_ptr):
    group_offsets = torch.cumsum(
        torch.tensor(
            [0]
            + [
                node_groups[batch_ptr[i] : batch_ptr[i + 1]].max() + 1
                for i in range(len(batch_ptr) - 1)
            ],
            device=node_groups.device,
        ),
        dim=0,
    )

    processed_groups = node_groups.clone()
    for i in range(len(batch_ptr) - 1):
        processed_groups[batch_ptr[i] : batch_ptr[i + 1]] += group_offsets[i]

    return processed_groups


class GraphLevelTrainer:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.dataset_root = args.data_path
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.eval_freq = args.eval_freq
        self.patience = args.patience
        self.stopping_criterion = args.stopping_criterion
        self.do_random_split = True
        self.loss_func = DATASET_TO_LOSS[self.dataset_name]
        self.is_hetero_data = True if (args.model in ["rgcn", "rgat"]) else False
        transforms = []
        if args.add_virtual_nodes:
            transforms.extend(
                [
                    T.ToDevice(args.device),
                    AddVirtualNodes(
                        num_selected_nodes=args.num_selected_nodes,
                        workers_per_node=args.workers_per_node[0],
                        replication_factor=args.workers_per_node[1],
                        return_hetero_data=self.is_hetero_data,
                        criterion=args.criterion,
                        directed=args.virtual_directed,
                    ),
                ]
            )
        else:
            transforms.extend([T.ToDevice(args.device)])

        if args.featureless:
            transforms.extend(
                [T.Constant(value=1.0, cat=False), T.ToDevice(args.device)]
            )

        transform = T.Compose(transforms)

        self.dataset = load_dataset(
            self.dataset_name,
            self.dataset_root,
            transform=transform,
            extra_feature=args.feature,
        )
        example_graph = self.dataset[0]
        if isinstance(example_graph, HeteroData):
            self.in_features = example_graph["node"].x.shape[-1]
            self.edge_types = example_graph.edge_types
        else:
            self.in_features = example_graph.x.shape[-1]
            self.edge_types = None

        self.out_features = len(
            torch.unique(next(iter(DataLoader(self.dataset, len(self.dataset)))).y)
        )

        if self.args.dataset == "tree-neighbors":
            dataset = DATASET_TO_CLS[self.args.dataset]()
            self.tree_neighbors_inp_size, _ = dataset.get_dims()
            self.in_features = args.hidden_size
            self.out_features = self.out_features + 1

        self.virtual_emb_history = defaultdict(list)
        self.split_history = defaultdict(list)

    def train(self, model, run_id=0):
        if self.do_random_split:
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [0.8, 0.1, 0.1]
            )
            train_idx = train_dataset.indices
            val_idx = val_dataset.indices
            test_idx = test_dataset.indices
            self.split_history[run_id] = {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }

        best_state = None
        best_val_eval = 0
        val_goal = 0
        best_train_eval = 0
        train_goal = 0
        early_stop_count = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
        )

        progress = tqdm(range(self.epochs), disable=self.args.wandb)
        for epoch in progress:
            total_loss = 0
            model.train()
            for batch in train_loader:
                if self.args.virtual_emb:
                    virtual_emb = model.pe_layer.pe.weight.data.clone()
                    self.virtual_emb_history[run_id].append(virtual_emb)

                if "virtual_node_groups" in batch:
                    # assign unique indices to virtual groups within batch.
                    batch.virtual_node_groups = preprocess_batched_groups(
                        batch.virtual_node_groups, batch.ptr
                    )
                pred = model(batch, pool=False if "root_mask" in batch else True)
                pred = pred.squeeze(1)
                y = batch.y
                loss = self.loss_func(pred, y)

                if self.args.virtual_reg_alpha > 0:
                    loss += self.args.virtual_reg_alpha * auxiliary_virtual_emb_loss(
                        model.pe_layer.pe.weight
                    )

                loss.backward()

                total_loss += loss

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step(total_loss)

            if epoch % self.eval_freq == 0:
                eval_dict = evaluate(
                    model,
                    loaders_dict={"val": validation_loader, "train": train_loader},
                    loss_func=self.loss_func,
                    eval_metric=DATASET_TO_METRIC[self.dataset_name],
                )
                val_accuracy = eval_dict["val_eval"]
                train_accuracy = eval_dict["train_eval"]
                val_loss = eval_dict["val_loss"]
                train_loss = eval_dict["train_loss"]

                if self.args.stopping_criterion == "train":
                    if train_accuracy > train_goal:
                        best_train_eval = train_accuracy
                        early_stop_count = 0
                        best_state = deepcopy(model.state_dict())
                        train_goal = train_accuracy * 1.01
                    elif train_accuracy > best_train_eval:
                        best_train_eval = train_accuracy
                        best_state = deepcopy(model.state_dict())
                        early_stop_count += self.eval_freq
                    else:
                        early_stop_count += self.eval_freq
                else:
                    if val_accuracy > val_goal:
                        best_val_eval = val_accuracy
                        early_stop_count = 0
                        best_state = deepcopy(model.state_dict())
                        val_goal = val_accuracy * 1.01
                    elif val_accuracy > best_val_eval:
                        best_val_eval = val_accuracy
                        best_state = deepcopy(model.state_dict())
                        early_stop_count += self.eval_freq
                    else:
                        early_stop_count += self.eval_freq

                if early_stop_count >= self.patience:
                    break

                progress.set_postfix_str(
                    f"Run {run_id}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}"
                )

        model.load_state_dict(best_state)

        eval_dict = evaluate(
            model,
            {"val": validation_loader, "train": train_loader, "test": test_loader},
            loss_func=self.loss_func,
            eval_metric=DATASET_TO_METRIC[self.dataset_name],
        )
        val_accuracy = eval_dict["val_eval"]
        train_accuracy = eval_dict["train_eval"]
        test_accuracy = eval_dict["test_eval"]

        logger.info(
            f"Run: {run_id}, Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}, Test Acc: {test_accuracy:.3f}"
        )

        self.running_time = progress.format_dict["elapsed"]

        return {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
        }
