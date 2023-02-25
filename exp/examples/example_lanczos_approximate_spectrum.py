"""Reproduce figure 15 of papyan2020traces to test Lanczos spectral estimation."""

import matplotlib.pyplot as plt
from numpy import e, exp, linspace, log, logspace, matmul, median, ndarray, zeros
from numpy.linalg import eigh
from numpy.random import pareto, randn, seed
from scipy.sparse.linalg import LinearOperator, eigsh

from vivit.hessianfree.lanczos import (
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)
from vivit.hessianfree.utils import LowRank


def create_matrix(dim: int) -> ndarray:
    """Draw a matrix from the matrix distribution used in papyan2020traces, Figure 15a.

    Args:
        dim: Matrix dimension.

    Returns:
        A sample from the matrix distribution.k
    """
    X = zeros((dim, dim))
    X[0, 0] = 5
    X[1, 1] = 4
    X[2, 2] = 3

    Z = randn(dim, dim)

    return X + 1 / dim * matmul(Z, Z.transpose())


def create_matrix_log_spectrum(dim: int) -> ndarray:
    """Draw a matrix from the matrix distribution used in papyan2020traces, Figure 15b.

    Args:
        dim: Matrix dimension.

    Returns:
        A sample from the matrix distribution.
    """
    Z = pareto(a=1, size=(dim, 2 * dim))

    return 1 / (2 * dim) * Z @ Z.transpose()


class DenseLinearOperator(LinearOperator):
    """Linear operator created from a dense matrix."""

    def __init__(self, A: ndarray):
        """Store dense operator for multiplication.

        Args:
            A: Dense matrix.
        """
        super().__init__(shape=A.shape, dtype=A.dtype)
        self._A = A

    def _matvec(self, x: ndarray) -> ndarray:
        """Apply the linear operator to a vector.

        Args:
            x: Vector.

        Returns:
            Result of linear operator applied to the vector.
        """
        return matmul(self._A, x)


def figure15a():
    """Create Figure 15a in papyan2020traces."""
    seed(0)

    # hyperparameters
    dim = 2000
    num_points = 1024
    num_bins = 100

    # spectral density hyperparameters
    ncv = 128
    num_repeats = 10
    kappa = 3
    margin = 0.05

    Y = create_matrix(dim)
    Y_linop = DenseLinearOperator(Y)

    ###################################################################################
    # no rank-deflation

    print("Computing the full spectrum")
    evals, _ = eigh(Y)

    print("Approximating density")
    grid, density = lanczos_approximate_spectrum(
        Y_linop,
        ncv,
        num_points=num_points,
        num_repeats=num_repeats,
        kappa=kappa,
        boundaries=(evals[0], evals[-1]),
        margin=margin,
    )

    print("Plotting results")
    plt.figure()

    left, right = grid[0], grid[-1]
    bins = linspace(left, right, num_bins, endpoint=True)
    plt.hist(evals, bins=bins, log=True, density=True, label="Exact")

    plt.plot(grid, density, label="Approximate")

    plt.xlabel("Eigenvalue")
    plt.ylabel("Spectral density")
    plt.ylim(bottom=1e-5, top=1e1)
    plt.legend()
    plt.show()
    plt.close()

    ###################################################################################
    # project out top-k eigenvalues

    k = 3
    print(f"Computing top-{k} eigenvalues of normalized operator")
    evals_top, evecs_top = eigsh(Y_linop, k=k, which="LA")
    Y_top = LowRank(evals_top, evecs_top)

    print(f"Approximating density with eliminated top-{k} eigenspace")
    grid, density_no_top = lanczos_approximate_spectrum(
        Y_linop - Y_top,
        ncv,
        num_points=num_points,
        num_repeats=num_repeats,
        kappa=kappa,
        boundaries=(evals[0], evals[-1]),
        margin=0.05,
    )

    print("Plotting results")
    plt.figure()

    bins = linspace(left, right, num_bins, endpoint=True)
    plt.hist(evals, bins=bins, log=True, density=True, label="Exact")

    plt.plot(grid, density_no_top, label=f"Approximate (no top {k})")

    plt.plot(
        evals_top,
        len(evals_top) * [median(density_no_top)],
        linestyle="",
        marker="o",
        label=f"Top {k}",
    )

    plt.xlabel("Eigenvalue")
    plt.ylabel("Spectral density")
    plt.ylim(bottom=1e-5, top=1e1)
    plt.legend()
    plt.show()
    plt.close()


def figure15b():
    """Create Figure 15b in papyan2020traces."""
    seed(0)

    # hyperparameters
    dim = 500
    num_points = 1024
    num_bins = 100
    epsilon = 1e-5
    margin = 0.05

    # spectral density hyperparameters
    ncv = 256
    num_repeats = 10
    kappa = 1.04

    Y = create_matrix_log_spectrum(dim)
    Y_linop = DenseLinearOperator(Y)

    print("Computing the full log-spectrum")
    evals, _ = eigh(Y)

    print("Approximating log-spectrum")
    grid, density = lanczos_approximate_log_spectrum(
        Y_linop,
        ncv,
        num_points=num_points,
        num_repeats=num_repeats,
        kappa=kappa,
        boundaries=(min(abs(evals)), max(abs(evals))),
        margin=margin,
        epsilon=epsilon,
    )

    print("Plotting results")
    plt.figure()

    log_evals = log(abs(evals) + epsilon)
    log_eval_min, log_eval_max = log_evals.min(), log_evals.max()

    log_width = log_eval_max - log_eval_min
    log_left = log_eval_min - margin * log_width
    log_right = log_eval_max + margin * log_width

    plt.semilogx()
    bins = logspace(log_left, log_right, num=num_bins, endpoint=True, base=e)
    plt.hist(exp(log_evals), bins=bins, log=True, density=True, label="Exact")

    plt.plot(grid, density, label="Approximate")

    plt.xlabel("Eigenvalue")
    plt.ylabel("Spectral density")
    plt.ylim(bottom=1e-14, top=1e-2)
    plt.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":
    figure15a()
    figure15b()
