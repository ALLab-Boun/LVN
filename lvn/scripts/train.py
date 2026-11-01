from collections import defaultdict
import json
import os

import numpy as np
import torch

from ..data import DATASET_TO_TYPE
from ..logging import logger
from ..model import GNN
from ..training import GraphLevelTrainer, NodeLevelTrainer
from .train_args import parse_arguments


def run_train(args):
    logger.info("Arguments: " + ", ".join(f"{k}={v}" for k, v in vars(args).items()))
    trainer = GraphLevelTrainer(args) if DATASET_TO_TYPE[args.dataset] == "graph" else NodeLevelTrainer(args)
    args.in_features = trainer.in_features
    args.out_features = trainer.out_features
    args.tree_neighbors_inp_size = trainer.tree_neighbors_inp_size

    evals = defaultdict(list)
    for run_id in range(args.n_runs):
        model = GNN(args).to(args.device)
        if run_id == 0:
            logger.info(model)
        for key, value in trainer.train(model, run_id).items():
            evals[key].append(value)

    results = {"all_evals": dict(evals)}
    for key, values in evals.items():
        results[f"{key}_mean"] = float(np.mean(values))
        results[f"{key}_ci"] = float(1.96 * np.std(values) / np.sqrt(len(values)))  # 95% CI
        logger.info(f"{key}: {results[f'{key}_mean']:.2f} +- {results[f'{key}_ci']:.2f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "eval.json"), "w") as f:
            json.dump({**vars(args), **results}, f, indent=2)
        if getattr(trainer, "splits", None):
            torch.save(trainer.splits, os.path.join(args.output_dir, "splits.pt"))
    return results


def main():
    run_train(parse_arguments())


if __name__ == "__main__":
    main()
