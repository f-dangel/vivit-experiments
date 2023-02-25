"""Shared functionality among eigenvector sub-experiments."""

from typing import Any, Callable, Dict, List, Tuple

from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from shared import fill_batch_norm_running_stats
from torch import Tensor, device
from torch.nn import Module

from exp.utils.models import has_batchnorm
from vivit.linalg.eigh import EighComputation


def top_eval(evals: Tensor) -> List[int]:
    """Criterion to compute the largest eigenvalue.

    Args:
        evals: Eigenvalues.

        Returns:
            Indices of largest eigenvalue.
    """
    return [len(evals) - 1]


def compute_ggn_evecs(
    model: Module,
    loss_func: Module,
    X: Tensor,
    y: Tensor,
    param_groups: List[Dict[str, Any]],
    computations: EighComputation,
) -> Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]]:
    """Compute the GGN eigenvalues and eigenvectors."""
    loss = loss_func(model(X), y)

    with backpack(
        computations.get_extension(),
        extension_hook=computations.get_extension_hook(
            param_groups, keep_batch_size=False, keep_backpack_buffers=False
        ),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    return computations._evals, computations._evecs


def run_ggn_evecs(
    architecture_fn: Callable[[int], Tuple[Module, Module, Tensor, Tensor]],
    param_groups_fn: Callable[
        [Module, Callable[[Tensor], List[int]]], List[Dict[str, Any]]
    ],
    computations_fn: Callable[[int], EighComputation],
    N: int,
    device: device,
) -> Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]]:
    """Build model, data, and compute the GGN top eigen-pair."""
    model, loss_func, X, y = architecture_fn(N)

    model = extend(model.to(device))
    loss_func = extend(loss_func.to(device))
    X = X.to(device)
    y = y.to(device)

    if has_batchnorm(model):
        if X.shape[0] > 1:
            fill_batch_norm_running_stats(model, X)

        model = model.eval()

    param_groups = param_groups_fn(model, top_eval)
    computations = computations_fn(N)

    evals, evecs = compute_ggn_evecs(model, loss_func, X, y, param_groups, computations)
    print(evals)
    # print(evecs)

    return evals, evecs


def full_batch_exact(N: int) -> EighComputation:
    """Build computations to use the full batch and exact GGN."""
    return EighComputation(subsampling=None, mc_samples=0, verbose=False)


def full_batch_mc(N: int) -> EighComputation:
    """Build computations to use the full batch and GGN-MC."""
    return EighComputation(subsampling=None, mc_samples=1, verbose=False)


def frac_batch_exact(N: int) -> EighComputation:
    """Build computations to use a fraction of the mini-batch and the exact GGN."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    return EighComputation(
        subsampling=list(range(max(N // 8, 1))), mc_samples=0, verbose=False
    )


def frac_batch_mc(N: int) -> EighComputation:
    """Build computations to use a fraction of the mini-batch and the GGN-MC."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    return EighComputation(
        subsampling=list(range(max(N // 8, 1))), mc_samples=1, verbose=False
    )
