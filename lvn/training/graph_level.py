import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import DATASET_TO_LOSS, load_dataset
from ..evaluation import evaluate_loaders
from ..logging import logger
from ..transforms import build_lvn_transform
from .common import EarlyStopping


class GraphLevelTrainer:
    def __init__(self, args):
        self.args = args
        self.loss_func = DATASET_TO_LOSS[args.dataset]
        transform = build_lvn_transform(args) if args.add_virtual_nodes else None
        self.dataset = load_dataset(args.dataset, args.data_path, transform=transform, tree_depth=args.tree_depth)
        self.in_features = self.dataset[0].x.shape[-1]
        self.out_features = len(torch.unique(torch.cat([g.y.view(-1) for g in self.dataset])))
        self.tree_neighbors_inp_size = None
        if args.dataset == "tree-neighbors":
            from ..data.neighborsmatch import DictionaryLookupDataset

            self.tree_neighbors_inp_size, _ = DictionaryLookupDataset(depth=args.tree_depth).get_dims()
            self.in_features = args.hidden_size
            self.out_features += 1
        self.splits = {}

    def train(self, model, run_id=0):
        args = self.args
        # Split seed + run id: every configuration is trained and evaluated on the same splits.
        generator = torch.Generator().manual_seed(args.split_seed + run_id) if args.split_seed >= 0 else None
        train_set, val_set, test_set = random_split(self.dataset, [0.8, 0.1, 0.1], generator=generator)
        self.splits[run_id] = {"train": train_set.indices, "val": val_set.indices, "test": test_set.indices}
        loaders = {
            "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(val_set, batch_size=args.batch_size),
            "test": DataLoader(test_set, batch_size=args.batch_size),
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_scheduler_factor, patience=args.lr_scheduler_patience)
        # LVN-embedding ablation: freeze the embeddings after k optimiser steps (0: fixed random).
        emb_params = list(model.lvn_embedding.parameters()) if model.lvn_embedding is not None else []
        if args.freeze_virtual_emb_after == 0:
            for p in emb_params:
                p.requires_grad_(False)
        self.emb_init = model.lvn_embedding.emb.weight.detach().clone() if emb_params else None

        stopper = EarlyStopping(args.patience, args.early_stopping_start)
        monitor = "train" if args.stopping_criterion == "train" else "val"
        step = 0
        progress = tqdm(range(args.epochs))
        for epoch in progress:
            model.train()
            total_loss = 0.0
            for batch in loaders["train"]:
                batch = batch.to(args.device)
                pred = model(batch, pool="root_mask" not in batch).squeeze(1)
                loss = self.loss_func(pred, batch.y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.detach()
                step += 1
                if step == args.freeze_virtual_emb_after:
                    for p in emb_params:
                        p.requires_grad_(False)
            scheduler.step(total_loss)

            if epoch % args.eval_freq == 0:
                ev = evaluate_loaders(model, {monitor: loaders[monitor]}, self.loss_func)
                metric = ev[f"{monitor}_eval"]
                if args.stopping_criterion == "train" and args.stop_at_perfect_train and metric >= 100.0:
                    stopper.step(metric, model, epoch)
                    break
                if stopper.step(metric, model, epoch, args.eval_freq):
                    break
                progress.set_postfix_str(f"run {run_id}: loss {float(total_loss):.3f}, {monitor} acc {metric:.2f}")

        model.load_state_dict(stopper.best_state)
        ev = evaluate_loaders(model, loaders, self.loss_func)
        logger.info(
            f"Run {run_id}: train {ev['train_eval']:.2f}, val {ev['val_eval']:.2f}, test {ev['test_eval']:.2f}"
        )
        return {"train_accuracy": ev["train_eval"], "val_accuracy": ev["val_eval"], "test_accuracy": ev["test_eval"]}
