import argparse
import random


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if value.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_config():
    parser = argparse.ArgumentParser(
        description="Federated Averaging Experiments")
    parser.add_argument("--method", type=str, default="fedavg")
    parser.add_argument("--n_client", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=1.0)
    parser.add_argument("--dirichlet", type=float, default=0.5)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--n_client_epoch", type=int, default=5)
    parser.add_argument("--s", type=int, default=10)

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--seed", type=int, default=5)

    parser.add_argument("--ours_n_sample", type=int, default=1)
    parser.add_argument("--lambd", type=float, default=0.0)
    parser.add_argument("--topk", type=float, default=0.01)

    parser.add_argument("--enable_sparse_masking", action="store_true", default=False)
    parser.add_argument("--sparsity_rate", type=float, default=0.0)
    parser.add_argument("--sparsity_min_density", type=float, default=0.0)
    parser.add_argument(
        "--sparsity_compression",
        type=str,
        default="CSR",
        choices=["CSR", "CSC", "BSR"],
    )
    parser.add_argument("--wandb_enabled", type=str2bool, default=True)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args
