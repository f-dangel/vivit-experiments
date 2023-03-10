"""Spectral analysis methods for SciPy linear operators."""

from typing import Tuple

from numpy import exp, inner, linspace, log, ndarray, pi, sqrt, zeros, zeros_like
from numpy.linalg import norm
from numpy.random import randn
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh


def fast_lanczos(
    A: LinearOperator, ncv: int, use_eigh_tridiagonal: bool = False
) -> Tuple[ndarray, ndarray]:
    """Lanczos iterations for large-scale problems (no reorthogonalization step).

    Algorithm 2 of papyan2020traces.

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors.
        use_eigh_tridiagonal: Whether to use eigh_tridiagonal to eigen-decompose the
            tri-diagonal matrix. Default: ``False``. Setting this value to ``True``
            results in faster eigen-decomposition, but is less stable.

    Returns:
        Eigenvalues and eigenvectors of the tri-diagonal matrix built up during
        Lanczos iterations. ``evecs[:, i]`` is normalized eigenvector of ``evals[i]``.
    """
    alphas, betas = zeros(ncv), zeros(ncv - 1)

    dim = A.shape[1]
    v, v_prev = None, None

    for m in range(ncv):
        if m == 0:
            v = randn(dim)
            v /= norm(v)
            v_next = A @ v

        else:
            v_next = A @ v - betas[m - 1] * v_prev

        alphas[m] = inner(v_next, v)
        v_next -= alphas[m] * v

        last = m == ncv - 1
        if not last:
            betas[m] = norm(v_next)
            v_next /= betas[m]
            v_prev = v
            v = v_next

    if use_eigh_tridiagonal:
        evals, evecs = eigh_tridiagonal(alphas, betas)
    else:
        T = diags([betas, alphas, betas], offsets=[-1, 0, 1]).todense()
        evals, evecs = eigh(T)

    return evals, evecs


def approximate_boundaries(A: LinearOperator, tol: float = 1e-2) -> Tuple[float, float]:
    """Approximate λₘᵢₙ(A) and λₘₐₓ(A) using SciPy's ``eigsh``.

    Args:
        A: Symmetric linear operator.
        tol: Relative accuracy used by ``eigsh``. ``0`` implies machine precision.
            Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.

    Returns:
        Estimates of λₘᵢₙ and λₘₐₓ.
    """
    eval_min, eval_max = eigsh(A, k=2, which="BE", tol=tol, return_eigenvectors=False)

    return eval_min, eval_max


def approximate_boundaries_abs(
    A: LinearOperator, tol: float = 1e-2
) -> Tuple[float, float]:
    """Approximate λₘᵢₙ(|A|) and λₘₐₓ(|A|) using SciPy's ``eigsh``.

    Args:
        A: Symmetric linear operator.
        tol: Relative accuracy used by ``eigsh``. ``0`` implies machine precision.
            Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.

    Returns:
        Estimates of λₘᵢₙ and λₘₐₓ of |A|.
    """
    (eval_max,) = eigsh(A, k=1, which="LM", tol=tol, return_eigenvectors=False)
    (eval_min,) = eigsh(A, k=1, which="SM", tol=tol, return_eigenvectors=False)

    return abs(eval_min), abs(eval_max)


def lanczos_approximate_spectrum(
    A: LinearOperator,
    ncv: int,
    num_points: int = 1024,
    num_repeats: int = 1,
    kappa: float = 3.0,
    boundaries: Tuple[float, float] = None,
    margin: float = 0.05,
    boundaries_tol: float = 1e-2,
) -> Tuple[ndarray, ndarray]:
    """Approximate the spectral density p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of A ∈ Rᵈˣᵈ.

    Internally rescales the operator spectrum to the interval [-1; 1] such that
    the width ``kappa`` of the Gaussian bumps used to approximate the delta peaks
    need not be tweaked.

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        num_points: Resolution.
        num_repeats: Number of Lanczos quadratures to average the density over.
            Default: ``1``. Taken from papyan2020traces, Section D.2.
        kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
            be greater than 1. Default: ``3``. Taken from papyan2020traces, Section D.2.
        boundaries: Estimates of the minimum and maximum eigenvalues of ``A``. If left
            unspecified, they will be estimated internally.
        margin: Relative margin added around the spectral boundary. Default: ``0.05``.
            Taken from papyan2020traces, Section D.2.
        boundaries_tol: (Only relevant if ``boundaries`` are not specified). Relative
            accuracy used to estimate the spectral boundary. ``0`` implies machine
            precision. Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.

    Returns:
        Grid points λ and approximated spectral density p(λ) of A.
    """
    if boundaries is None:
        boundaries = approximate_boundaries(A, tol=boundaries_tol)
    eval_min, eval_max = boundaries

    _width = eval_max - eval_min
    _padding = margin * _width

    eval_min, eval_max = eval_min - _padding, eval_max + _padding

    # use normalized operator ``(A - c I) / d`` whose spectrum lies in [-1; 1]
    c = (eval_max + eval_min) / 2
    d = (eval_max - eval_min) / 2

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, endpoint=True)
    grid_out = grid_norm * d + c
    density = zeros_like(grid_norm)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))

    for _ in range(num_repeats):
        evals, evecs = fast_lanczos(A, ncv)
        nodes = (evals - c) / d

        # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
        grid = grid_norm.reshape((1, num_points)).repeat(ncv, axis=0)
        nodes = nodes.reshape((ncv, 1)).repeat(num_points, axis=1)
        weights = (evecs[0, :] ** 2 / d).reshape((ncv, 1)).repeat(num_points, axis=1)

        density += (weights * _gaussian(grid, nodes, sigma)).sum(0) / num_repeats

    return grid_out, density


def _gaussian(x: ndarray, mu: ndarray, sigma: float) -> ndarray:
    """Normal distribution pdf.

    Args:
        x: Position to evaluate.
        mu: Mean values.
        sigma: Standard deviation.

    Returns:
        Values of normal distribution.
    """
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(2 * pi))


def lanczos_approximate_log_spectrum(
    A: LinearOperator,
    ncv: int,
    num_points: int = 1024,
    num_repeats: int = 1,
    kappa: float = 1.04,
    boundaries: Tuple[float, float] = None,
    margin: float = 0.05,
    boundaries_tol: float = 1e-2,
    epsilon: float = 1e-5,
) -> Tuple[ndarray, ndarray]:
    """Approximate the spectral density p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of log(|A| + εI) ∈ Rᵈˣᵈ.

    Here, log denotes the natural logarithm (i.e. base e).

    Internally rescales the operator spectrum to the interval [-1; 1] such that
    the width ``kappa`` of the Gaussian bumps used to approximate the delta peaks
    need not be tweaked.

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        num_points: Resolution.
        num_repeats: Number of Lanczos quadratures to average the density over.
            Default: ``1``. Taken from papyan2020traces, Section D.2.
        kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
            be greater than 1. Default: ``1.04``. Obtained by tweaking the parameter
            while reproducing Figure 15b of from papyan2020traces.
        boundaries: Estimates of the minimum and maximum eigenvalues of ``|A|``. If left
            unspecified, they will be estimated internally.
        margin: Relative margin added around the spectral boundary. Default: ``0.05``.
            Taken from papyan2020traces, Section D.2.
        boundaries_tol: (Only relevant if ``boundaries`` are not specified). Relative
            accuracy used to estimate the spectral boundary. ``0`` implies machine
            precision. Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        boundaries_tol: (Only relevant if ``boundaries`` are not specified). Relative
            accuracy used to estimate the spectral boundary. ``0`` implies machine
            precision. Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        epsilon: Shift to increase numerical stability. Default: ``1e-5``. Taken from
            papyan2020traces, Section D.2.

    Returns:
        Grid points λ and approximated spectral density p(λ) of log(|A| + εI).
    """
    if boundaries is None:
        boundaries = approximate_boundaries_abs(A, tol=boundaries_tol)

    log_eval_min, log_eval_max = (log(boundary + epsilon) for boundary in boundaries)

    _width = log_eval_max - log_eval_min
    _padding = margin * _width

    log_eval_min, log_eval_max = log_eval_min - _padding, log_eval_max + _padding

    # use normalized operator ``(log(|A| + εI) - c I) / d`` with spectrum in [-1; 1]
    c = (log_eval_max + log_eval_min) / 2
    d = (log_eval_max - log_eval_min) / 2

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, endpoint=True)
    grid_out = exp(grid_norm * d + c)
    density = zeros_like(grid_norm)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))

    for _ in range(num_repeats):
        evals, evecs = fast_lanczos(A, ncv)
        abs_evals = abs(evals) + epsilon
        log_evals = log(abs_evals)
        nodes = (log_evals - c) / d

        # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
        grid = grid_norm.reshape((1, num_points)).repeat(ncv, axis=0)
        nodes = nodes.reshape((ncv, 1)).repeat(num_points, axis=1)
        weights = (evecs[0, :] ** 2).reshape((ncv, 1)).repeat(num_points, axis=1)

        density += (weights * _gaussian(grid, nodes, sigma)).sum(0) / num_repeats

    density /= d * grid_out

    return grid_out, density
