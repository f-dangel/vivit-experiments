"""DeepOBS utility functions shared among experiments."""

import copy
import random
from typing import Tuple, Type
from warnings import warn

import numpy
from deepobs.pytorch import datasets, testproblems
from torch import Tensor, manual_seed
from torch.nn import Module
from torch.utils.data import DataLoader


def set_seeds(seed: int):
    """Set all DeepOBS-relevant random seeds.

    DeepOBS uses random numbers from multiple libraries. To make experiments
    reproducible, all random number generators must be seeded.

    Args:
        seed: Value that will be assigned to the random seed.
    """
    manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def get_deterministic_deepobs_train_loader(
    problem_cls: Type[testproblems.testproblem.TestProblem], N: int
) -> DataLoader:
    """Return a deterministic train loader (same data, maybe different order).

    Args:
        problem_cls: DeepOBS problem.
        N: Approximate batch size. It will be adapted if its value leads to dropping
            last mini-batches, which would imply non-deterministic data.

    Returns:
        Deterministic train loader for the test problem which uses batches of
        approximate specified size.

    Raises:
        NotImplementedError: For problems that are not supported.
    """
    if problem_cls in [testproblems.cifar10_3c3d, testproblems.cifar10_resnet32]:
        dataset = datasets.cifar10
    elif problem_cls == testproblems.cifar100_allcnnc:
        dataset = datasets.cifar100
    elif problem_cls == testproblems.fmnist_2c2d:
        dataset = datasets.fmnist
    else:
        raise NotImplementedError
    return _get_deterministic_train_loader(dataset, N)


def _get_deterministic_train_loader(
    deepobs_dataset: Type[datasets.DataSet], N: int
) -> DataLoader:
    """Return deterministic train loader for DeepOBS dataset `deepobs_dataset`.

    Turns off data augmentation and avoids drop_last by adapting the batch size.

     Args:
        deepobs_dataset: DeepOBS dataset instance
        N: Approximate batch size. Will be adapted if the passed value implies
            drop_last.

    Returns:
        Deterministic train loader for DeepOBS dataset.
    """
    # determine size of train set
    try:
        dataset = deepobs_dataset(1, data_augmentation=False)
    except TypeError:  # No data_augmentation avaiable (e.g. fmnist)
        dataset = deepobs_dataset(1)
    train_loader, _ = dataset._make_train_and_valid_dataloader()
    train_size = len(train_loader)

    # adapt N if it would lead to dropped last mini-batch
    if train_size % N != 0:
        N_input = N
        N = _get_divisor(train_size, close_to=N)
        warn(
            f"N_data={train_size}, N={N_input} implies drop_last={train_size%N_input}."
            + f" Using N={N} instead."
        )

    # create the actual train set
    try:
        dataset = deepobs_dataset(N, data_augmentation=False)
    except TypeError:  # No data_augmentation avaiable (e.g. fmnist)
        dataset = deepobs_dataset(N)

    train_loader, _ = dataset._make_train_and_valid_dataloader()
    return train_loader


def _get_divisor(N: int, close_to: int = None) -> int:
    """Find a divisor of ``N`` less than or equal to ``close_to``."""
    divisor = N // 2 if close_to is None else close_to
    while N % divisor != 0:
        divisor -= 1
    return divisor


def get_num_classes(problem_cls: Type[testproblems.testproblem.TestProblem]) -> int:
    if problem_cls in [
        testproblems.cifar10_3c3d,
        testproblems.cifar10_resnet32,
        testproblems.fmnist_2c2d,
    ]:
        return 10
    elif problem_cls in [testproblems.cifar100_allcnnc]:
        return 100
    else:
        raise NotImplementedError(f"Number of classes unknwon for {problem_cls}")


def get_deepobs_dataloader(problem_cls, eval_batchsize):
    """Get training dataloader of DeepOBS problems."""
    problem = problem_cls(batch_size=eval_batchsize)
    problem.set_up()

    # Initialize the testproblem instance to train mode and return dataloader
    problem.train_init_op()
    return problem.data._train_dataloader


def get_deepobs_architecture(
    problem_cls: Type[testproblems.testproblem.TestProblem], N: int
) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss function, and data of DeepOBS problems."""
    problem = problem_cls(batch_size=N)
    problem.set_up()
    problem.train_init_op()

    model = copy.deepcopy(problem.net)
    loss_func = copy.deepcopy(problem.loss_function(reduction="mean"))
    X, y = problem._get_next_batch()

    return model, loss_func, X.clone(), y.clone()


def cifar10_3c3d(N: int) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss_function, and data for CIFAR10-3c3d."""
    return get_deepobs_architecture(testproblems.cifar10_3c3d, N)


def fmnist_2c2d(N: int) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss_function, and data for F-MNIST-2c2d."""
    return get_deepobs_architecture(testproblems.fmnist_2c2d, N)


def cifar100_allcnnc(N: int) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss_function, and data for CIFAR100-All-CNNC."""
    return get_deepobs_architecture(testproblems.cifar100_allcnnc, N)


def cifar10_resnet32(N: int) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss_function, and data for CIFAR10-ResNet32."""
    return get_deepobs_architecture(testproblems.cifar10_resnet32, N)


def cifar10_resnet56(N: int) -> Tuple[Module, Module, Tensor, Tensor]:
    """Get model, loss_function, and data for CIFAR10-ResNet32."""
    return get_deepobs_architecture(testproblems.cifar10_resnet56, N)


def net_cifar10_3c3d() -> Tuple[Module, Module, Tensor, Tensor]:
    """Return the architecture of DeepOBS' cifar10_3c3d problem.

    Returns:
        Neural network of DeepOBS' cifar10_3c3d problem
    """
    model, _, _, _ = cifar10_3c3d(1)
    return model
