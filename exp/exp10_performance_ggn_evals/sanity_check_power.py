"""Sanity check: Compare ViViT and power iteration."""

import sys

import torch
from backpack import extend
from power_iteration import compute_ggn_evecs_power
from shared import (  # noqa: F401
    fill_batch_norm_running_stats,
    layerwise_group,
    one_group,
    paramwise_group,
)
from shared_evecs import compute_ggn_evecs, full_batch_exact  # noqa: F401
from torch import allclose

from exp.utils.deepobs import (  # noqa: F401
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
    set_seeds,
)
from exp.utils.models import has_batchnorm
from exp.utils.vivit import make_top_k_criterion

if __name__ == "__main__":
    # Fetch arguments from command line, then run
    N, device, architecture_fn, param_groups_fn, K = sys.argv[1:]
    # example: python sanity_check_power.py 1 cpu cifar10_3c3d one_group 3
    computations_fn = "full_batch_exact"

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
    computations = computations_fn(N)

    if has_batchnorm(model):
        if X.shape[0] > 1:
            fill_batch_norm_running_stats(model, X)

        model = model.eval()

    param_groups = param_groups_fn(model, make_top_k_criterion(K))
    for group in param_groups:
        group["k"] = K
        group["max_iter"] = 1000
        group["tol"] = 1e-5

    set_seeds(0)
    print("Running power iteration")
    evals_power, evecs_power = compute_ggn_evecs_power(
        model, loss_func, X, y, param_groups
    )
    set_seeds(0)
    print("Running ViViT")
    evals_vivit, evecs_vivit = compute_ggn_evecs(
        model, loss_func, X, y, param_groups, computations
    )

    all_match = True

    for group in param_groups:
        group_id = id(group)
        print("Group", group_id)

        group_evals_power = evals_power[group_id]
        group_evals_vivit = evals_vivit[group_id]

        print("\tPower:", group_evals_power)
        print("\tViViT:", group_evals_vivit)

        atol, rtol = 1e-4, 5e-3
        evals_close = allclose(
            group_evals_power, group_evals_vivit, atol=atol, rtol=rtol
        )
        print(f"\tEvals close? {evals_close}")

        if not evals_close:
            all_match = False

        group_evecs_power = evecs_power[group_id]
        group_evecs_vivit = evecs_vivit[group_id]
        atol, rtol = 5e-3, 5e-2
        evecs_close = all(
            allclose(evec_power.abs(), evec_vivit.abs(), atol=atol, rtol=rtol)
            for evec_power, evec_vivit in zip(group_evecs_power, group_evecs_vivit)
        )
        print(f"\tEvecs close? {evecs_close}")

        if not evecs_close:
            all_match = False

    if not all_match:
        raise ValueError("Sanity check failed.")
    else:
        print("Sanity check passed.")
