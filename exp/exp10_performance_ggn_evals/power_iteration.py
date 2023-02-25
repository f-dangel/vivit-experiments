"""Power iteration for the GGN via matrix-free multiplication used as comparison.

The implementation is basically an adapted version of the PyHessian library
(https://github.com/amirgholami/PyHessian) and uses the same hyperparameters
(convergence criterion etc). Instead of storing eigenvectors as nested
parameter lists, we stack them per parameter, which reduces ``for``-loop calls
in the orthonormalization step and is more consistent with the format used by
the ViViT implementation.
"""

from typing import Dict, List, Tuple, Union
from warnings import warn

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import Tensor, cat, einsum, rand_like
from torch.nn import Module


def compute_ggn_evecs_power(
    model: Module,
    loss_func: Module,
    X: Tensor,
    y: Tensor,
    param_groups: List[Dict],
    return_num_iters: bool = False,
) -> Union[
    Tuple[Dict[int, Tensor], Dict[int, List[Tensor]]],
    Tuple[Dict[int, Tensor], Dict[int, List[Tensor]], Dict[int, int]],
]:
    """Compute the top-K eigenpair of the GGN via power iteration for each group.

    Args:
        model: Neural network.
        loss_func: Loss function.
        X: Input to the neural network.
        y: Ground truth.
        param_groups: Parameter groups that specify the block-diagonal approximation
            and power iteration hyperparameters. Each group must contain the keys
            ``'params'``, ``'k'``, ``'max_iter'``, and ``'tol'``.
        return_num_iters: Also return the number of power iterations.
            Default: ``False``.

    Returns:
        Eigenvalues and associated eigenvectors for each group. Keys are group ids.
        Eigenvalues are sorted in ascending order. Format of eigenvectors is
        ``[[k, *p1.shape], [k, *p2.shape], ...]``.
    """
    outputs = model(X)
    loss = loss_func(outputs, y)

    evals, evecs = {}, {}

    if return_num_iters:
        num_iters = {}

    for group in param_groups:
        parameters = group["params"]
        k = group["k"]
        max_iter = group["max_iter"]
        tol = group["tol"]

        group_id = id(group)
        if return_num_iters:
            (
                evals[group_id],
                evecs[group_id],
                num_iters[group_id],
            ) = compute_group_ggn_evecs_power(
                loss, outputs, parameters, k, max_iter, tol, return_num_iters=True
            )
        else:
            evals[group_id], evecs[group_id] = compute_group_ggn_evecs_power(
                loss, outputs, parameters, k, max_iter, tol
            )

    if return_num_iters:
        return evals, evecs, num_iters
    else:
        return evals, evecs


def compute_group_ggn_evecs_power(
    loss: Tensor,
    outputs: Tensor,
    parameters: List[Tensor],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-3,
    return_num_iters: bool = False,
) -> Union[Tuple[Tensor, List[Tensor]], Tuple[Tensor, List[Tensor], int]]:
    """Compute the top-K eigenpair of the GGN w.r.t. parameters by power iteration.

    Args:
        loss: Differentiable scalar loss value.
        outputs: Differentiable model output.
        parameters: Parameters used in the GGN.
        k: Number of top eigenpairs to compute.
        max_iter: Maximum number of iterations used per eigenpair. Default ``100``.
        tol: Relative tolerance to detect convergence. Default ``1e-3``.
        return_num_iters: Also return the number of power iterations.
            Default: ``False``.

    Returns:
        Eigenvalues and associated eigenvectors. Eigenvalues are sorted in ascending
        order. Format of eigenvectors is ``[[k, *p1.shape], [k, *p2.shape], ...]``.
    """
    num_evals = 0

    num_iters = 0
    evecs, evals = None, None

    while num_evals < k:
        eigval = Tensor([float("inf")]).to(loss.device)
        v_list = [rand_like(p) for p in parameters]
        normalize(v_list)

        has_converged = False

        for _ in range(max_iter):
            if evecs is not None:
                orthonormal(v_list, evecs)

            v_list, new_eigval = iteration(loss, outputs, parameters, v_list)
            num_iters += 1

            if converged(eigval, new_eigval, tol):
                has_converged = True
                eigval = new_eigval
                break

            eigval = new_eigval

        if not has_converged:
            warn(f"Exceeded maximum number of {max_iter} iterations")

        if evals is None:
            evals = eigval.unsqueeze(0)
        else:
            evals = cat([eigval.unsqueeze(0), evals])

        if evecs is None:
            evecs = [v.unsqueeze(0) for v in v_list]
        else:
            evecs = [cat([v.unsqueeze(0), e]) for v, e in zip(v_list, evecs)]

        num_evals += 1

    if return_num_iters:
        return evals, evecs, num_iters
    else:
        return evals, evecs


def orthonormal(v_list: List[Tensor], others_list: List[Tensor]):
    """Orthonormalize a vector with respect to a set of other vectors.

    Args:
        v_list: Vector in parameter format to be ortho-normalized.
        others_list: Stacked vectors in parameter format.
    """
    weights = sum(
        einsum("k...,...->k", others, v) for others, v in zip(others_list, v_list)
    )
    for v, others in zip(v_list, others_list):
        v -= einsum("k,k...->...", weights, others)

    normalize(v_list)


def normalize(v_list: List[Tensor], epsilon: float = 1e-6):
    """Normalize a vector in parameter format.

    Args:
        v_list: Vector in parameter format.
        epsilon: Small shift added when dividing by the norm. Default ``1e-6``.
    """
    norm = sum((v**2).sum() for v in v_list).sqrt()
    for v in v_list:
        # PyHessian: https://github.com/amirgholami/PyHessian/blob/8a61d6cea978cf610def8fe40f01c60d18a2653d/pyhessian/utils.py#L57 # noqa: B950
        v /= norm + epsilon


def iteration(
    loss: Tensor, outputs: Tensor, parameters: List[Tensor], v_list: List[Tensor]
) -> Tuple[List[Tensor], Tensor]:
    """Perform one power iteration with the GGN.

    Args:
        loss: Differentiable scalar loss value.
        outputs: Differentiable model output.
        parameters: Parameters used in the GGN.
        v_list: Current normalized eigenvector estimate in parameter format.

    Returns:
        Updated normalized eigenvector estimate and eigenvalue estimate
    """
    new_v_list = list(ggn_vector_product_from_plist(loss, outputs, parameters, v_list))
    new_eigval = sum((v * new_v).sum() for v, new_v in zip(v_list, new_v_list))
    normalize(new_v_list)

    return new_v_list, new_eigval


def converged(old: Tensor, new: Tensor, tol: float) -> bool:
    """Determine if power iteration has converged (same criterion as PyHessian).

    Args:
        old: Eigenvalue estimate of previous iteration.
        new: Current eigenvalue estimate.
        tol: Relative tolerance that must be satisfied for convergence.

    Returns:
        Whether the power iteration has converged up to the specified tolerance.
    """
    # https://github.com/amirgholami/PyHessian/blob/0f7e0f63a0f132998608013351ba19955fc9d861/pyhessian/hessian.py#L149-L150 # noqa: B950
    return (old - new).abs() / (old.abs() + 1e-6) < tol
