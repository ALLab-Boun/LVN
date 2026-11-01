"""Run the configurations of a config file (configs/tu.json, configs/node.json, configs/tree.json)
and compare with the reported results.

    reproduce configs/tu.json --only mutag --device cuda --output-dir output/tu
"""
from argparse import ArgumentParser
import json
import os

import numpy as np

from .train import run_train
from .train_args import parse_arguments


def to_cli(args):
    cli = []
    for key, value in args.items():
        cli += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    return cli


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--only", default=None, help="Run only entries whose name contains this string")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-path", default="./datasets/")
    p.add_argument("--output-dir", default="output")
    a = p.parse_args()

    summary = []
    for entry in json.load(open(a.config)):
        if a.only and a.only not in entry["name"]:
            continue
        seeds = entry.get("split_seeds", [entry["args"].get("split-seed", 0)])
        metric = "train_accuracy" if "reported_train_accuracy" in entry else "test_accuracy"
        values = []
        for seed in seeds:
            run_args = {**entry["args"], "split-seed": seed, "device": a.device, "data-path": a.data_path,
                        "output-dir": os.path.join(a.output_dir, f"{entry['name']}_seed{seed}")}
            results = run_train(parse_arguments(to_cli(run_args)))
            values += results["all_evals"][metric]
        reported = entry.get("reported_test_accuracy", entry.get("reported_train_accuracy"))
        summary.append((entry["name"], metric, float(np.mean(values)),
                        float(1.96 * np.std(values) / np.sqrt(len(values))), reported))

    print(f"\n{'configuration':40s} {'metric':15s} {'obtained':>15s}   reported")
    for name, metric, mean, ci, reported in summary:
        print(f"{name:40s} {metric:15s} {mean:7.2f} +- {ci:5.2f}   {reported}")


if __name__ == "__main__":
    main()
