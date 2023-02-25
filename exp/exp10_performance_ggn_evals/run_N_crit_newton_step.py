"""Compute a damped Newton step."""

import sys

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from backpack.extensions import SqrtGGNExact, SqrtGGNMC
from shared import (  # noqa: F401
    criterion,
    fill_batch_norm_running_stats,
    layerwise_group,
    one_group,
    paramwise_group,
)

from exp.utils.deepobs import (  # noqa: F401
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
    get_deepobs_architecture,
)
from exp.utils.models import has_batchnorm
from vivit.optim import BaseComputations, ConstantDamping


def compute_newton_step(model, loss_func, X, y, param_groups, computations):
    """Compute the Newton step."""
    loss = loss_func(model(X), y)

    savefield = "_newton_step"
    damping = ConstantDamping(1.0)

    with backpack(
        *computations.get_extensions(param_groups),
        extension_hook=computations.get_extension_hook(
            param_groups,
            damping,
            savefield,
            keep_gram_mat=False,
            keep_gram_evals=False,
            keep_gram_evecs=False,
            keep_gammas=False,
            keep_lambdas=False,
            keep_batch_size=False,
            keep_coefficients=False,
            keep_newton_step=False,
            keep_backpack_buffers=False,
        ),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    newton_step = [
        [getattr(p, savefield) for p in group["params"]] for group in param_groups
    ]
    return newton_step


def full_batch_exact(N):
    """Build computations to use the full batch and exact GGN."""
    return BaseComputations(
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        subsampling_first=None,
        subsampling_directions=None,
        subsampling_second=None,
        verbose=False,
    )


def full_batch_mc(N):
    """Build computations to use the full batch and the GGN-MC."""
    return BaseComputations(
        extension_cls_directions=SqrtGGNMC,
        extension_cls_second=SqrtGGNMC,
        subsampling_first=None,
        subsampling_directions=None,
        subsampling_second=None,
        verbose=False,
    )


def frac_batch_exact(N):
    """Build computations to use a fraction of the mini-batch and the exact GGN.

    Evaluate first-order directional derivatives on the full batch.
    Evaluate second-order directional derivatives on a fraction of the batch.
    """
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    frac = 8
    return BaseComputations(
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        subsampling_first=None,
        subsampling_directions=list(range(max(N // frac, 1))),
        subsampling_second=list(range(max(N // frac, 1))),
        verbose=False,
    )


def frac_batch_mc(N):
    """Build computations to use a fraction of the mini-batch and the GGN-MC.

    Evaluate first-order directional derivatives on the full batch.
    Evaluate second-order directional derivatives on a fraction of the batch.
    """
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    frac = 8
    return BaseComputations(
        extension_cls_directions=SqrtGGNMC,
        extension_cls_second=SqrtGGNMC,
        subsampling_first=None,
        subsampling_directions=list(range(max(N // frac, 1))),
        subsampling_second=list(range(max(N // frac, 1))),
        verbose=False,
    )


def run(architecture_fn, param_groups_fn, computations_fn, N, device):
    """Build model, data, and run Newton step computation."""
    model, loss_func, X, y = architecture_fn(N)

    model = extend(model.to(device))
    loss_func = extend(loss_func.to(device))
    X = X.to(device)
    y = y.to(device)

    if has_batchnorm(model):
        if X.shape[0] > 1:
            fill_batch_norm_running_stats(model, X)

        model = model.eval()

    param_groups = param_groups_fn(model, criterion)
    computations = computations_fn(N)

    newton_step = compute_newton_step(
        model, loss_func, X, y, param_groups, computations
    )
    print(newton_step)


if __name__ == "__main__":
    # Fetch arguments from command line, then run
    N, device, architecture_fn, param_groups_fn, computations_fn = sys.argv[1:]
    # example:
    # python run_N_crit_newton_step.py 1 cpu cifar10_3c3d one_group full_batch_exact

    N = int(N)
    device = torch.device(device)

    thismodule = sys.modules[__name__]
    architecture_fn = getattr(thismodule, architecture_fn)
    param_groups_fn = getattr(thismodule, param_groups_fn)
    computations_fn = getattr(thismodule, computations_fn)

    torch.manual_seed(0)
    run(architecture_fn, param_groups_fn, computations_fn, N, device)
