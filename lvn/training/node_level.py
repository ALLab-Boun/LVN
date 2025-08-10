from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.datasets import HeterophilousGraphDataset
from tqdm import tqdm


from ..data import load_dataset, DATASET_TO_LOSS, DATASET_TO_CLS, DATASET_TO_METRIC
from ..evaluation import evaluate
from ..transforms import AddVirtualNodes
from .util import gpu_memory_usage
from ..logging import logger


def create_split_masks(graph, split_ratio=(0.6, 0.2, 0.2), seed=None):
    """
    Create train/val/test masks respecting virtual node groups.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = graph.edge_index.device

    if hasattr(graph, "virtual_node_groups"):
        num_nodes = graph.virtual_node_groups.max().item() + 1
    else:
        num_nodes = graph.num_nodes

    num_train = int(split_ratio[0] * num_nodes)
    num_val = int(split_ratio[1] * num_nodes)
    indices = torch.randperm(num_nodes, device=device)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train : num_train + num_val]] = True
    test_mask[indices[num_train + num_val :]] = True

    return train_mask, val_mask, test_mask


class NodeLevelTrainer:
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
        self.loss_func = DATASET_TO_LOSS[self.dataset_name]
        self.is_hetero_data = True if (args.model in ["rgcn", "rgat"]) else False
        transforms = [T.ToDevice(args.device)]
        if DATASET_TO_CLS[self.dataset_name] != HeterophilousGraphDataset:
            transforms.append(T.NormalizeFeatures())
        if args.add_virtual_nodes:
            transforms.append(
                AddVirtualNodes(
                    num_selected_nodes=args.num_selected_nodes,
                    workers_per_node=args.workers_per_node[0],
                    replication_factor=args.workers_per_node[1],
                    return_hetero_data=self.is_hetero_data,
                    criterion=args.criterion,
                    directed=args.virtual_directed,
                    centrality_cache_path=f"{self.dataset_root}{self.dataset_name}_{args.criterion}.pkl",
                )
            )
        transform = T.Compose(transforms)
        self.dataset = load_dataset(
            self.dataset_name,
            self.dataset_root,
            transform=transform,
        )
        self.in_features = self.dataset.num_features
        self.out_features = self.dataset.num_classes
        self.dataset = list(self.dataset)
        self.graph = self.dataset[0]

        logger.info("Graph after transformations: %s", self.graph)

        if self.graph.train_mask.ndim > 1:
            self.num_splits = self.graph.train_mask.shape[-1]
        else:
            self.num_splits = 1

        if isinstance(self.graph, HeteroData):
            self.edge_types = self.graph.edge_types
        else:
            self.edge_types = None

    def train(self, model, run_id=1):
        best_val_eval = 0
        val_goal = 0
        best_train_eval = 0
        train_goal = 0
        best_train_loss = 1e10
        best_state = model.state_dict()
        early_stop_count = 0
        best_epoch = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
        )

        graph = self.graph
        train_mask, val_mask, test_mask = create_split_masks(graph)
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

        progress = tqdm(range(self.epochs), disable=self.args.wandb)
        for epoch in progress:
            model.train()
            optimizer.zero_grad()
            pred = model(graph, pool=False)
            if pred.shape[-1] == 1:
                pred = pred.squeeze()
            loss = self.loss_func(pred[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            if epoch % self.eval_freq == 0:
                eval_dict = evaluate(
                    model,
                    dataset=self.dataset,
                    loss_func=self.loss_func,
                    eval_metric=DATASET_TO_METRIC[self.dataset_name],
                )

                train_eval = eval_dict["train_eval"]
                val_eval = eval_dict["val_eval"]

                if self.args.stopping_criterion == "train":
                    if train_eval > train_goal:
                        best_train_eval = train_eval
                        early_stop_count = 0
                        best_state = deepcopy(model.state_dict())
                        train_goal = train_eval * 1.01
                    elif train_eval > best_train_eval:
                        best_train_eval = train_eval
                        best_state = deepcopy(model.state_dict())
                        early_stop_count += self.eval_freq
                    else:
                        early_stop_count += self.eval_freq
                else:
                    if val_eval > val_goal:
                        best_val_eval = val_eval
                        early_stop_count = 0
                        best_state = deepcopy(model.state_dict())
                        val_goal = val_eval * 1.01
                    elif val_eval > best_val_eval:
                        best_val_eval = val_eval
                        best_state = deepcopy(model.state_dict())
                        early_stop_count += self.eval_freq
                    else:
                        early_stop_count += self.eval_freq

                if early_stop_count >= self.patience:
                    break

                model.train()

                if self.args.device != "cpu":
                    memory_usage = gpu_memory_usage()
                else:
                    memory_usage = 0.0

                train_eval = eval_dict["train_eval"]
                val_eval = eval_dict["val_eval"]
                test_eval = eval_dict["test_eval"]
                train_loss = eval_dict["train_loss"]
                # val_loss = eval_dict["val_loss"]
                # test_loss = eval_dict["test_loss"]

                progress.set_postfix(
                    train_loss=f"{train_loss:.3f}",
                    train_acc=f"{train_eval:.3f}",
                    val_acc=f"{val_eval:.3f}",
                    test_acc=f"{test_eval:.3f}",
                    best_epoch=best_epoch,
                    mem_use=f"{memory_usage:.2f} GB",
                    lr=optimizer.param_groups[0]["lr"],
                )

        model.load_state_dict(best_state)

        eval_dict = evaluate(
            model,
            dataset=self.dataset,
            eval_metric=DATASET_TO_METRIC[self.dataset_name],
        )

        train_eval = eval_dict["train_eval"]
        val_eval = eval_dict["val_eval"]
        test_eval = eval_dict["test_eval"]

        print(
            f"Run {run_id}, Train Acc: {train_eval:.3f}, Val Acc: {val_eval:.3f}, Test Acc: {test_eval:.3f}"
        )

        self.running_time = progress.format_dict["elapsed"]

        return {
            "train_eval": train_eval,
            "val_eval": val_eval,
            "test_accuracy": test_eval,
        }
