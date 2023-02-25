"""Shared functionality among sub-experiments."""

from typing import Any, Callable, Dict, List

from backpack import disable
from torch import Tensor, no_grad
from torch.nn import Module

from exp.utils.models import _recursive_submodules

eigenvalue_cutoff = 1e-4


def criterion(evals, must_exceed=eigenvalue_cutoff):
    """Filter out eigenvalues close to zero.

    Args:
        evals (torch.Tensor): Eigenvalues.
        must_exceed (float, optional): Minimum value for eigenvalue to be kept.

        Returns:
            [int]: Indices of non-zero eigenvalues.
    """
    return [idx for idx, ev in enumerate(evals) if ev > must_exceed]


def one_group(model, criterion):
    """Build parameter group containing all parameters in one group."""
    return [{"params": list(model.parameters()), "criterion": criterion}]


def paramwise_group(model, criterion):
    """Build parameter group containing one parameter per group."""
    return [{"params": [p], "criterion": criterion} for p in model.parameters()]


def layerwise_group(
    model: Module, criterion: Callable[[Tensor], List[int]]
) -> List[Dict[str, Any]]:
    """Group parameters of a layer."""
    param_groups = []

    for module in _recursive_submodules(model):
        trainable_parameters = [
            p for (_, p) in module.named_parameters() if p.requires_grad
        ]
        trainable_names = [n for (n, p) in module.named_parameters() if p.requires_grad]

        if len(trainable_parameters) > 0:
            print(f"Constructing group with {trainable_names}")
            param_groups.append(
                {"params": trainable_parameters, "criterion": criterion}
            )

    # check
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    ids = sum(([id(p) for p in group["params"]] for group in param_groups), start=[])
    assert len(trainable_ids) == len(ids)
    assert set(ids) == trainable_ids

    return param_groups


def fill_batch_norm_running_stats(
    model: Module, X: Tensor, num_forward_passes: int = 5
):
    """Perform some forward passes to initialize the running statistics of BatchNorm.

    Args:
        model: Neural network.
        X: Input to the neural network.
        num_forward_passes: Number of forward passes to perform. Default; ``5``.
    """
    with no_grad(), disable():
        for _ in range(num_forward_passes):
            model(X)
