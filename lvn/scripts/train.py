from multiprocessing import set_start_method


from collections import defaultdict
from datetime import datetime
import os
import json

import numpy as np
import wandb
import torch

from ..model import MODEL_TO_CLS
from ..data import DATASET_TO_TYPE
from ..training import GraphLevelTrainer, NodeLevelTrainer
from ..logging import logger
from ..training.util import timer
from .train_args import parse_arguments


@timer
def run_train(args):
    if args.wandb:
        wandb.init(
            project="oversquashing",
            entity="gnn-study-group",
        )

    logger.info(
        "Arguments:\n"
        + "\n".join(f"{arg}: {value}" for arg, value in vars(args).items())
    )

    best_model = None
    best_run = 0
    best_test_acc = float("-inf")

    if DATASET_TO_TYPE[args.dataset] == "graph":
        trainer = GraphLevelTrainer(args)
    else:
        trainer = NodeLevelTrainer(args)

    args.in_features = trainer.in_features
    args.out_features = trainer.out_features
    args.edge_types = trainer.edge_types
    if args.dataset == "tree-neighbors":
        args.tree_neighbors_inp_size = trainer.tree_neighbors_inp_size

    models = {}
    all_evals = defaultdict(lambda: [])
    for run_id in range(args.n_runs):
        model = MODEL_TO_CLS[args.model](args).to(args.device)
        if run_id == 0:
            logger.info(model)

        eval_results = trainer.train(model, run_id)
        for key in eval_results:
            all_evals[key].append(eval_results[key])

        if eval_results["test_accuracy"] > best_test_acc:
            best_model = model
            best_run = run_id

        models[run_id] = model.state_dict()

    final_results = {}
    final_results["all_evals"] = all_evals
    for key in all_evals:
        mean = np.mean(all_evals[key])
        ci = 1.96 * np.std(all_evals[key]) / np.sqrt(args.n_runs)
        final_results[f"{key}_mean"] = mean
        final_results[f"{key}_ci"] = ci

        logger.info(f"{key}: {mean:.3f} +- {ci:.3f}")

    if args.wandb:
        wandb.log(final_results)
        wandb.finish(0)

    if args.save:
        output_name = args.output_name
        if output_name is None:
            output_name = ""
        output_dir = os.path.join(
            "output/",
            output_name,
            args.dataset + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M"),
        )
        os.makedirs(output_dir, exist_ok=True)
        args_dict = vars(args)
        combined_results = {**args_dict, **final_results}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            json.dump(combined_results, f, indent=4)

        torch.save(best_model.state_dict(), os.path.join(output_dir, "model.pt"))
        torch.save(models, os.path.join(output_dir, "models.pt"))
        if args.virtual_emb:
            torch.save(
                trainer.virtual_emb_history[best_run],
                os.path.join(output_dir, "virtual_emb.pt"),
            )

        torch.save(
            trainer.split_history,
            os.path.join(output_dir, "splits.pt"),
        )

        logger.info(f"Saved results to: {output_dir}!")

    return final_results


def main():
    set_start_method("spawn")
    args = parse_arguments()
    run_train(args)


if __name__ == "__main__":
    main()
