"""Utility functions for the ``vivit`` library."""

from typing import Callable, List

from torch import Tensor


def make_top_k_criterion(k: int) -> Callable[[Tensor], List[int]]:
    """Create a filter function that keeps the top-k eigenvalues.

    Use all eigenvalues if number of eigenvalues is smaller than k.

    Args:
        k: Number of top eigenvalues.

    Returns:
        Filter function.
    """

    def top_k(evals: Tensor) -> List[int]:
        """Filter the top-k eigenvalues.

        Use all eigenvalues if number of eigenvalues is smaller than k.

        Args:
            evals: Sorted eigenvalues (ascending order).

        Returns:
            Indices of k largest eigenvalue.
        """
        keep = min(evals.numel(), k)

        return [len(evals) - keep + idx for idx in range(k)]

    return top_k
