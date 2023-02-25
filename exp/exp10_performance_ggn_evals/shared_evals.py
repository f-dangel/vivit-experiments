"""Shared functionality among eigenvalue sub-experiments."""

from typing import Any, Callable, Dict, List, Tuple

from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from shared import criterion, fill_batch_norm_running_stats
from torch import Tensor, device
from torch.nn import Module

from exp.utils.models import has_batchnorm
from vivit.linalg.eigvalsh import EigvalshComputation


def compute_ggn_gram_evals(
    model: Module,
    loss_func: Module,
    X: Tensor,
    y: Tensor,
    param_groups: List[Dict[str, Any]],
    computations: EigvalshComputation,
) -> Dict[int, Tensor]:
    """Compute the GGN eigenvalues."""
    loss = loss_func(model(X), y)

    with backpack(
        computations.get_extension(),
        extension_hook=computations.get_extension_hook(
            param_groups, keep_batch_size=False, keep_backpack_buffers=False
        ),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    return computations._evals


def run_ggn_gram_evals(
    architecture_fn: Callable[[int], Tuple[Module, Module, Tensor, Tensor]],
    param_groups_fn: Callable[
        [Module, Callable[[Tensor], List[int]]], List[Dict[str, Any]]
    ],
    computations_fn: Callable[[int], EigvalshComputation],
    N: int,
    device: device,
):
    """Build model, data, and run GGN spectral computations."""
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

    gram_evals = compute_ggn_gram_evals(
        model, loss_func, X, y, param_groups, computations
    )
    print(gram_evals)
    return gram_evals


def compute_num_trainable_params(
    architecture_fn: Callable[[int], Tuple[Module, Any, Any, Any]]
) -> int:
    """Evaluate the number of trainable model parameters."""
    N_dummy = 1
    model, _, _, _ = architecture_fn(N_dummy)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def full_batch_exact(N: int) -> EigvalshComputation:
    """Build computations to use the full batch and exact GGN."""
    return EigvalshComputation(subsampling=None, mc_samples=0, verbose=False)


def full_batch_mc(N: int) -> EigvalshComputation:
    """Build computations to use the full batch and GGN-MC."""
    return EigvalshComputation(subsampling=None, mc_samples=1, verbose=False)


def frac_batch_exact(N: int) -> EigvalshComputation:
    """Build computations to use a fraction of the mini-batch and the exact GGN."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    return EigvalshComputation(
        subsampling=list(range(max(N // 8, 1))), mc_samples=0, verbose=False
    )


def frac_batch_mc(N: int) -> EigvalshComputation:
    """Build computations to use a fraction of the mini-batch and the GGN-MC."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    return EigvalshComputation(
        subsampling=list(range(max(N // 8, 1))), mc_samples=1, verbose=False
    )
