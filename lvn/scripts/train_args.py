from argparse import ArgumentParser, ArgumentTypeError


from ..model import MODEL_TO_CLS
from ..data import DATASET_TO_CLS
from ..transforms import CRITERION_TO_FUNC


def check_tuple(value):
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError()
        return tuple(int(part) for part in parts)
    except ValueError:
        raise ArgumentTypeError("Must be a tuple of two integers separated by a comma.")


def parse_arguments(args=None):
    parser = ArgumentParser(description="GCN Model Training")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of training runs. (default: 1)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs. (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weigth decay rate. (default: 1e-5)",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1,
        help="Run eval every n epoch. (default: 1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience. (default: 20)",
    )
    parser.add_argument(
        "--stopping-criterion",
        type=str,
        default="validation",
        help="Stopping criterion. train or validation. (default: validation)",
    )
    parser.add_argument(
        "--early-stopping-start",
        type=int,
        default=50,
        help="Epoch to start checking for early stopping. (default: 50)",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=10,
        help="Patience for learning rate scheduler. (default: 10)",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.1,
        help="Factor for learning rate scheduler. (default: 0.1)",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Number of features in hidden layers (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of GCN layers to stack (default: 4)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate for dropout layers (default: 0.5)",
    )
    parser.add_argument(
        "--dropout-linear",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to apply dropout after input linear layer (default: False).",
    )
    parser.add_argument(
        "--layer-norm",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to use layer norm (default: False).",
    )
    parser.add_argument(
        "--residual",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to use residual connections (default: False).",
    )
    parser.add_argument(
        "--last-layer-ff",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to use a feedforward layer after the last GCN layer (default: False).",
    )

    parser.add_argument(
        "--add-virtual-nodes",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to add virtual nodes (default: False).",
    )
    parser.add_argument(
        "--num-selected-nodes", type=int, help="Number of central nodes to select."
    )
    parser.add_argument(
        "--workers-per-node",
        type=check_tuple,
        default=(2, 2),
        help="Workers per node and replication factor (default: (2,2)).",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="degree",
        choices=CRITERION_TO_FUNC,
        help="Centrality criterion to add virtual nodes.",
    )
    parser.add_argument(
        "--virtual-directed",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to apply the directed virtual node transform (default: false)",
    )
    parser.add_argument(
        "--virtual-dropout-rate",
        type=float,
        default=0.0,
        help="Within-group dropout rate for virtual nodes (default: 0.0)",
    )
    parser.add_argument(
        "--feature",
        default=None,
        choices=["ones", "degree"],
        help="Feature to include.",
    )
    parser.add_argument(
        "--featureless",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to use featureless graphs (default: False).",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="./datasets/",
        help="Path to the directory containing the dataset (default: ./datasets/)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Path to store the results (default: ./output/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mutag",
        choices=DATASET_TO_CLS.keys(),
        help="Dataset name (default: mutag)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=MODEL_TO_CLS.keys(),
        help="Model name (default: gcn)",
    )
    parser.add_argument(
        "--conv-type",
        type=str,
        default="gcn",
        choices=["gcn", "gin"],
        help="Model name (default: gcn)",
    )
    parser.add_argument(
        "--initial-ff",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to add an initial FF layer (default: False).",
    )
    parser.add_argument(
        "--virtual-emb",
        type=lambda x: x.lower() == "true",
        default=False,
        choices=[True, False],
        help="Whether to add a virtual node embedding layer (default: False).",
    )
    parser.add_argument(
        "--virtual-emb-type", type=str, default="shared", choices=["shared", "separate"]
    )
    parser.add_argument(
        "--virtual-emb-method",
        type=str,
        default="add",
        choices=["add", "replace", "add_one"],
        help="Virtual emb method (default: add).",
    )
    parser.add_argument(
        "--virtual-reg-alpha",
        type=float,
        default=0.0,
        help="Scaling factor for virtual embedding regularization.",
    )
    parser.add_argument(
        "--virtual-group-pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "sort", "add"],
        help="Virtual group pooling method (default: mean).",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot-grads", action="store_true")

    return parser.parse_args(args)
