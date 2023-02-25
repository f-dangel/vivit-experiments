"""Repeatedly time eigenpair computation with power iteration. Write times to .json."""

import sys
from os import makedirs, path
from timeit import repeat
from typing import Dict, List, Tuple

import torch
from power_iteration import compute_ggn_evecs_power
from shared import (  # noqa: F401
    fill_batch_norm_running_stats,
    layerwise_group,
    one_group,
    paramwise_group,
)
from torch import Tensor, cuda

from exp.utils.deepobs import (  # noqa: F401
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
    set_seeds,
)
from exp.utils.models import has_batchnorm
from exp.utils.path import write_to_json
from exp.utils.vivit import make_top_k_criterion

# IO
HERE_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.join(HERE_DIR, "results", "time", "evecs", "tol")


def get_output_file(
    architecture: str,
    device: str,
    param_groups: str,
    computations: str,
    N: str,
    K: str,
    tol: str,
    max_iter: str,
) -> str:
    """Return path of the output file."""
    return path.join(
        DATA_DIR,
        f"{architecture}_{device}_{param_groups}_{computations}_power_N_{N}_K_{K}"
        + f"_tol_{tol}_max_iter_{max_iter}.json",
    )


if __name__ == "__main__":
    # Fetch arguments from command line, then run
    N, device, architecture_fn, param_groups_fn, K, tol, max_iter = sys.argv[1:]
    # e.g.: python run_time_evecs_power_tol.py 1 cpu cifar10_3c3d one_group 3 0.001 100
    computations_fn = "full_batch_exact"

    DATA_FILE = get_output_file(
        architecture_fn, device, param_groups_fn, computations_fn, N, K, tol, max_iter
    )
    makedirs(path.dirname(DATA_FILE), exist_ok=True)

    if path.isfile(DATA_FILE):
        print(f"Skipping file (already exists): {DATA_FILE}")
        sys.exit(0)

    N = int(N)
    K = int(K)
    tol = float(tol)
    max_iter = int(max_iter)
    device = torch.device(device)
    REPEATS = 20

    thismodule = sys.modules[__name__]
    architecture_fn = getattr(thismodule, architecture_fn)
    param_groups_fn = getattr(thismodule, param_groups_fn)

    set_seeds(0)

    # setup
    model, loss_func, X, y = architecture_fn(N)

    model = model.to(device)
    loss_func = loss_func.to(device)
    X = X.to(device)
    y = y.to(device)

    if has_batchnorm(model):
        if X.shape[0] > 1:
            fill_batch_norm_running_stats(model, X)

        model = model.eval()

    param_groups = param_groups_fn(model, make_top_k_criterion(K))
    for group in param_groups:
        group["k"] = K
        group["max_iter"] = max_iter
        group["tol"] = tol

    def benchmark() -> (
        Tuple[Dict[int, Tensor], Dict[int, List[Tensor]], Dict[int, List[int]]]
    ):
        """Compute top-K eigenpairs of the GGN via power iteration.

        Returns:
            Eigenvalues and eigenvectors for each group.
        """
        set_seeds(0)

        evals, evecs, num_iters = compute_ggn_evecs_power(
            model, loss_func, X, y, param_groups, return_num_iters=True
        )
        if "cuda" in str(device):
            cuda.synchronize()

        return evals, evecs, num_iters

    times = repeat(benchmark, repeat=REPEATS, number=1)
    # benchmark() is deterministic -> get output of any of the benchmarked runs
    evals, _, num_iters = benchmark()
    evals = {key: val.detach().cpu().numpy() for key, val in evals.items()}

    DATA = {
        "times": times,
        "evals": evals,
        "num_iters": num_iters,
        "k": K,
        "max_iter": max_iter,
        "tol": tol,
    }
    write_to_json(DATA_FILE, DATA)
