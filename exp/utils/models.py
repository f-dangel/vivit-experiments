"""Model definitions for experiments."""

from typing import Iterable

import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, Module


def simple_cnn():
    """A simplistic, yet deep, ReLU convolutional neural network for MNIST."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 20, 5, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(20, 50, 5, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 50, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 10),
    )


def has_batchnorm(model: Module) -> bool:
    """Determine if a model contains batch normalization layers.

    Args:
        model: Neural network consisting of ``nn.Module``s.

    Returns:
        Whether the model contains a batch normalization layer.
    """
    return any(
        isinstance(module, (BatchNorm1d, BatchNorm2d, BatchNorm3d))
        for module in _recursive_submodules(model)
    )


def _recursive_submodules(model: Module) -> Iterable[Module]:
    no_submodules = len(list(model.children())) == 0
    if no_submodules:
        yield model

    for module in model.children():
        has_children = len(list(module.children())) > 0

        if has_children:
            yield from _recursive_submodules(module)
        else:
            yield module
