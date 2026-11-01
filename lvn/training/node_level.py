import numpy as np
import torch
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..data import DATASET_TO_LOSS, load_dataset
from ..evaluation import evaluate_masks
from ..logging import logger
from ..transforms import build_lvn_transform
from .common import EarlyStopping


def split_indices(num_nodes, seed, split_ratio=(0.6, 0.2, 0.2)):
    """Random train/val/test split of the original nodes."""
    perm = np.random.RandomState(seed).permutation(num_nodes)
    num_train = int(split_ratio[0] * num_nodes)
    num_val = int(split_ratio[1] * num_nodes)
    return perm[:num_train], perm[num_train : num_train + num_val], perm[num_train + num_val :]


def original_to_row(graph, num_original_nodes):
    """Output row (group) of every original node; identity without LVNs."""
    device = graph.edge_index.device
    if "group_orig" not in graph:
        return torch.arange(num_original_nodes, device=device)
    go = graph.group_orig.to(device)
    out = torch.full((num_original_nodes,), -1, dtype=torch.long, device=device)
    out[go] = torch.arange(go.numel(), device=device)
    return out


def create_split_masks(graph, num_original_nodes, seed):
    """Masks over the model's output rows from a split of the original nodes. The split
    depends only on the seed and the number of nodes, so it is the same with and without
    LVNs (paired comparisons)."""
    rows = original_to_row(graph, num_original_nodes)
    num_rows = graph.group_orig.numel() if "group_orig" in graph else num_original_nodes
    masks = []
    for idx in split_indices(num_original_nodes, seed):
        mask = torch.zeros(num_rows, dtype=torch.bool, device=graph.edge_index.device)
        mask[rows[torch.as_tensor(idx, device=rows.device)]] = True
        masks.append(mask)
    return tuple(masks)


class NodeLevelTrainer:
    def __init__(self, args):
        self.args = args
        self.loss_func = DATASET_TO_LOSS[args.dataset]
        dataset = load_dataset(
            args.dataset, args.data_path, transform=T.Compose([T.ToDevice(args.device), T.NormalizeFeatures()])
        )
        self.in_features = dataset.num_features
        self.out_features = dataset.num_classes
        self.tree_neighbors_inp_size = None
        graph = dataset[0]
        self.num_original_nodes = graph.num_nodes
        y = graph.y
        if args.add_virtual_nodes:
            graph = build_lvn_transform(args)(graph)
            graph.y = y[graph.group_orig]  # one label per output row (group)
        self.graph = graph
        logger.info("Graph after transformations: %s", graph)

    def train(self, model, run_id=0):
        args = self.args
        graph = self.graph
        seed = args.split_seed + run_id if args.split_seed >= 0 else int(np.random.randint(2**31 - 1))
        train_mask, val_mask, test_mask = create_split_masks(graph, self.num_original_nodes, seed)
        masks = {"train": train_mask, "val": val_mask, "test": test_mask}

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_scheduler_factor, patience=args.lr_scheduler_patience)
        emb_params = list(model.lvn_embedding.parameters()) if model.lvn_embedding is not None else []
        if args.freeze_virtual_emb_after == 0:
            for p in emb_params:
                p.requires_grad_(False)

        stopper = EarlyStopping(args.patience, args.early_stopping_start)
        monitor = "train" if args.stopping_criterion == "train" else "val"
        progress = tqdm(range(args.epochs))
        for epoch in progress:
            model.train()
            pred = model(graph, pool=False)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            loss = self.loss_func(pred[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)
            if epoch + 1 == args.freeze_virtual_emb_after:
                for p in emb_params:
                    p.requires_grad_(False)

            if epoch % args.eval_freq == 0:
                ev = evaluate_masks(model, graph, {monitor: masks[monitor]})
                if stopper.step(ev[f"{monitor}_eval"], model, epoch, args.eval_freq):
                    break
                progress.set_postfix_str(f"run {run_id}: loss {loss.item():.3f}, {monitor} acc {ev[f'{monitor}_eval']:.2f}")

        model.load_state_dict(stopper.best_state)
        ev = evaluate_masks(model, graph, masks)
        logger.info(
            f"Run {run_id}: train {ev['train_eval']:.2f}, val {ev['val_eval']:.2f}, test {ev['test_eval']:.2f}"
        )
        return {"train_accuracy": ev["train_eval"], "val_accuracy": ev["val_eval"], "test_accuracy": ev["test_eval"]}
