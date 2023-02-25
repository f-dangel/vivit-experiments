"""Repeatedly time eigenpair computation with ViViT. Write times to .json."""

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
DATA_DIR = path.join(HERE_DIR, "results", "time", "evecs")


def get_output_file(
    architecture: str, device: str, param_groups: str, computations: str, N: str, K: str
) -> str:
    """Return path of the output file."""
    return path.join(
        DATA_DIR,
        f"{architecture}_{device}_{param_groups}_{computations}_N_{N}_K_{K}.json",
    )


if __name__ == "__main__":
    # Fetch arguments from command line, then run
    N, device, architecture_fn, param_groups_fn, computations_fn, K = sys.argv[1:]
    # example: python run_time_evecs.py 1 cpu cifar10_3c3d one_group full_batch_exact 3

    DATA_FILE = get_output_file(
        architecture_fn, device, param_groups_fn, computations_fn, N, K
    )
    makedirs(path.dirname(DATA_FILE), exist_ok=True)

    N = int(N)
    K = int(K)
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

    def benchmark() -> Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]]:
        """Compute top-K eigenpairs of the GGN.

        Returns:
            Eigenvalues and eigenvectors for each group.
        """
        set_seeds(0)

        evals, evecs = compute_ggn_evecs(
            model, loss_func, X, y, param_groups, computations
        )
        if "cuda" in str(device):
            cuda.synchronize()

        return evals, evecs

    times = repeat(benchmark, repeat=REPEATS, number=1)

    DATA = {"times": times}
    write_to_json(DATA_FILE, DATA)
