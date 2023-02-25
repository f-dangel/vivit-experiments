"""Power iteration for the Gram matrix"""

from typing import Tuple
from warnings import warn

from torch import Tensor, cat, einsum, rand


def power_iteration(
    tensor: Tensor,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[Tensor, Tensor]:
    """Compute the top-K eigenpair of tensor.

    Args:
        loss: Differentiable scalar loss value.
        outputs: Differentiable model output.
        parameters: Parameters used in the GGN.
        k: Number of top eigenpairs to compute.
        max_iter: Maximum number of iterations used per eigenpair. Default ``100``.
        tol: Relative tolerance to detect convergence. Default ``1e-3``.

    Returns:
        Eigenvalues and associated eigenvectors. Eigenvalues are sorted in ascending
        order. Format of eigenvectors is ``[k, *tensor.shape[1]]``.
    """
    evecs, evals = None, None

    for _ in range(k):
        eigval = Tensor([float("inf")]).to(tensor.device)
        v = rand(tensor.shape[1]).to(tensor.device)
        normalize(v)

        has_converged = False

        for _ in range(max_iter):
            if evecs is not None:
                orthonormal(v, evecs)

            v, new_eigval = iteration(tensor, v)

            if converged(eigval, new_eigval, tol):
                has_converged = True
                eigval = new_eigval
                break

            eigval = new_eigval

        if not has_converged:
            warn(f"Exceeded maximum number of {max_iter} iterations")

        evals = (
            eigval.unsqueeze(0) if evals is None else cat([eigval.unsqueeze(0), evals])
        )

        evecs = v.unsqueeze(0) if evecs is None else cat([v.unsqueeze(0), evecs])

    return evals, evecs


def orthonormal(v: Tensor, others: Tensor):
    """Orthonormalize a vector with respect to a set of other vectors."""
    weights = einsum("k...,...->k", others, v)
    v -= einsum("k,k...->...", weights, others)

    normalize(v)


def normalize(v: Tensor, epsilon: float = 1e-6):
    """Normalize a vector.

    Args:
        v: Vector.
        epsilon: Small shift added when dividing by the norm. Default ``1e-6``.
    """
    norm = (v**2).sum().sqrt()
    # PyHessian: https://github.com/amirgholami/PyHessian/blob/8a61d6cea978cf610def8fe40f01c60d18a2653d/pyhessian/utils.py#L57 # noqa: B950
    v /= norm + epsilon


def iteration(tensor: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """Perform one power iteration with the GGN.

    Returns:
        Updated normalized eigenvector estimate and eigenvalue estimate
    """
    new_v = tensor @ v
    new_eigval = (v * new_v).sum()
    normalize(new_v)

    return new_v, new_eigval


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
