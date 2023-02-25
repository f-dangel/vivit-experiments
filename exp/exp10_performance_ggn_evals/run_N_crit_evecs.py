"""Compute GGN eigen-pairs."""

import sys

import torch
from shared import layerwise_group, one_group, paramwise_group  # noqa: F401
from shared_evecs import (  # noqa: F401
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
    run_ggn_evecs,
)

from exp.utils.deepobs import (  # noqa: F401
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
    get_deepobs_architecture,
    set_seeds,
)

if __name__ == "__main__":
    # Fetch arguments from command line, then run
    N, device, architecture_fn, param_groups_fn, computations_fn = sys.argv[1:]
    # example: python run_N_crit_evecs.py 1 cpu cifar10_3c3d one_group full_batch_exact

    N = int(N)
    device = torch.device(device)

    thismodule = sys.modules[__name__]
    architecture_fn = getattr(thismodule, architecture_fn)
    param_groups_fn = getattr(thismodule, param_groups_fn)
    computations_fn = getattr(thismodule, computations_fn)

    set_seeds(0)
    run_ggn_evecs(architecture_fn, param_groups_fn, computations_fn, N, device)
