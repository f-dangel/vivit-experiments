"""Repeatedly time eigenpair computation with power iteration. Write times to .json."""

import sys
from os import makedirs, path
from timeit import repeat
from typing import Dict, List, Tuple

import torch
from backpack import extend
from shared import (  # noqa: F401
    fill_batch_norm_running_stats,
    layerwise_group,
    one_group,
    paramwise_group,
)
from shared_evecs import (  # noqa: F401
    compute_ggn_evecs,
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
)
from torch import Tensor, cuda

from exp.exp10_performance_ggn_evals.gram_power_iteration import power_iteration
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
from vivit.linalg.eigh import gram_eigendecomposition

# IO
HERE_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = path.join(HERE_DIR, "results", "time", "evecs", "gram_tol")


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
        f"{architecture}_{device}_{param_groups}_{computations}_gram_power_N_{N}_K_{K}"
        + f"_tol_{tol}_max_iter_{max_iter}.json",
    )


if __name__ == "__main__":
    # Fetch arguments from command line, then run
    (
        N,
        device,
        architecture_fn,
        param_groups_fn,
        computations_fn,
        K,
        tol,
        max_iter,
    ) = sys.argv[1:]
    # e.g.: python run_time_evecs_gram_power_tol.py 1 cpu cifar10_3c3d one_group
    # full_batch_exact 3 0.001 100

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
    computations_fn = getattr(thismodule, computations_fn)

    set_seeds(0)

    # setup
    model, loss_func, X, y = architecture_fn(N)

    model = extend(model.to(device))
    loss_func = extend(loss_func.to(device))
    X = X.to(device)
    y = y.to(device)

    if has_batchnorm(model):
        if X.shape[0] > 1:
            fill_batch_norm_running_stats(model, X)

        model = model.eval()

    param_groups = param_groups_fn(model, make_top_k_criterion(K))
    computations = computations_fn(N)

    def power(tensor):
        """Method that will be used to eigen-decompose the Gram matrix."""
        evals, evecs = power_iteration(tensor, K, max_iter=max_iter, tol=tol)
        evecs = evecs.transpose(0, 1)  # same convention as symeig
        return evals, evecs

    def benchmark() -> (
        Tuple[Dict[int, Tensor], Dict[int, List[Tensor]], Dict[int, List[int]]]
    ):
        """Compute top-K eigenpairs of the GGN via power iteration on the Gram matrix.

        Returns:
            Eigenvalues and eigenvectors for each group.
        """
        set_seeds(0)

        with gram_eigendecomposition(power):
            evals, evecs = compute_ggn_evecs(
                model, loss_func, X, y, param_groups, computations
            )

            if "cuda" in str(device):
                cuda.synchronize()

        return evals, evecs

    times = repeat(benchmark, repeat=REPEATS, number=1)
    # benchmark() is deterministic -> get output of any of the benchmarked runs
    evals, _ = benchmark()
    evals = {key: val.detach().cpu().numpy() for key, val in evals.items()}

    DATA = {
        "times": times,
        "evals": evals,
        "k": K,
        "max_iter": max_iter,
        "tol": tol,
    }
    write_to_json(DATA_FILE, DATA)
