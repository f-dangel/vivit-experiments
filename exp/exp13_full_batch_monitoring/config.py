"""This file contains the configurations for the experiment, i.e. optimization scenarios
with different DeepOBS architectures, optimizers and hyperparameters. 
"""

import os

import numpy as np
import torch
from deepobs.config import DEFAULT_TEST_PROBLEMS_SETTINGS
from deepobs.pytorch.testproblems import (
    cifar10_3c3d,
    cifar10_resnet32,
    cifar100_allcnnc,
    fmnist_2c2d,
)

from exp.utils.deepobs import get_deepobs_dataloader, get_num_classes
from exp.utils.deepobs_runner import CheckpointRunner

HEREDIR = os.path.dirname(os.path.abspath(__file__))
DEEPOBS_OUTPUT = os.path.join(HEREDIR, "results", "deepobs_log")
CHECKPOINTS_OUTPUT = os.path.join(HEREDIR, "results", "checkpoints")
EVAL_OUTPUT = os.path.join(HEREDIR, "results", "eval")
PLOT_OUTPUT = os.path.join(HEREDIR, "results", "plots")


def complete_config(config):
    """Add batch size, number of epochs, number of classes and the checkpoints (if not
    specified by user) to the configuration dict.
    """

    # Find deepobs default parameters and plug into `config`
    problem_cls = config["problem_cls"]
    deepobs_config = DEFAULT_TEST_PROBLEMS_SETTINGS[problem_cls.__name__]

    config["batch_size"] = deepobs_config["batch_size"]
    config["num_epochs"] = deepobs_config["num_epochs"]
    config["num_classes"] = get_num_classes(problem_cls)

    # Add checkpoints if not specified by user (e.g. `[(0, 0), (0, 100), (56, 0)]`) and
    # also convert into decimal numbers
    if "checkpoints" not in config.keys():
        config = _add_auto_checkpoints(config)
    config["num_checkpoints"] = len(config["checkpoints"])
    config = _add_decimal_checkpoints(config)

    return config


def _add_auto_checkpoints(config, num_checkpoints=100):
    """Add automated grid of checkpoints to the configuration dict. `checkpoints` is a
    list of tuples: The first number is the epoch, the second number corresponds to the
    batch index within this epoch.
    """

    # Shifted log-space
    cp_decimal = np.geomspace(1, config["num_epochs"], num_checkpoints) - 1

    # Split decimal epoch into integer epoch and integer batch index
    cp_epoch, cp_batch = divmod(cp_decimal, 1)
    cp_epoch = cp_epoch.astype(dtype=np.int32)
    num_batches = len(
        get_deepobs_dataloader(config["problem_cls"], config["batch_size"])
    )
    cp_batch = np.floor(cp_batch * num_batches).astype(dtype=np.int32)

    # Create tuples, remove duplicates
    cp_with_duplicates = list(zip(cp_epoch, cp_batch))
    checkpoints = []
    [checkpoints.append(cp) for cp in cp_with_duplicates if cp not in checkpoints]

    config["checkpoints"] = checkpoints
    return config


def _add_decimal_checkpoints(config):
    """Add `"decimal_checkpoints"` to the configuration dict. These checkpoints are
    single decimal numbers.
    NOTE: This may not be completely accurate since it depends on whether the
    `drop_last` argument was set to `True` or `False` when the data loader was created.
    """

    num_batches = len(
        get_deepobs_dataloader(config["problem_cls"], config["batch_size"])
    )

    decimal_checkpoints = []
    for epoch, batch in config["checkpoints"]:
        decimal_checkpoints.append(epoch + batch / num_batches)

    config["decimal_checkpoints"] = decimal_checkpoints
    return config


def run_config(config):
    """Train a DeepOBS problem, save model at checkpoints while training."""

    torch.manual_seed(0)

    # Set up the `CheckpointRunner`-instance and initialize the checkpoints
    runner = CheckpointRunner(
        optimizer_class=config["optimizer_cls"],
        hyperparameter_names=config["hyperparams"],
        checkpoint_dir=CHECKPOINTS_OUTPUT,
    )
    runner.set_checkpoints(config["checkpoints"])

    # Run the testproblem
    runner.run(
        testproblem=config["problem_cls"].__name__,
        output_dir=DEEPOBS_OUTPUT,
        l2_reg=0.0,
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        skip_if_exists=True,
    )


# ======================================================================================
# Configurations
# NOTE: You can also specify custom checkpoints in the dictionaries below, e.g.
# `"checkpoints": [(0, 0), (0, 100), (56, 0)]`. The first number is the epoch, the
# second number corresponds to the batch index within this epoch.
# ======================================================================================
fmnist_2c2d_sgd = complete_config(
    {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.02069138081114788},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": fmnist_2c2d,
    }
)

fmnist_2c2d_adam = complete_config(
    {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.00012742749857031334}},
        "problem_cls": fmnist_2c2d,
    }
)

cifar10_3c3d_sgd = complete_config(
    {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.00379269019073225},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": cifar10_3c3d,
    }
)

cifar10_3c3d_adam = complete_config(
    {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.00029763514416313193}},
        "problem_cls": cifar10_3c3d,
    }
)

cifar100_allcnnc_sgd = complete_config(
    {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.04832930238571752},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": cifar100_allcnnc,
    }
)

cifar100_allcnnc_adam = complete_config(
    {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.0006951927961775605}},
        "problem_cls": cifar100_allcnnc,
    }
)

# For details on the hyperparameters for this case, see ``exp15_resnet_gridsearch``
cifar10_resnet32_sgd = complete_config(
    {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.06309573444801936},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": cifar10_resnet32,
    }
)

# For details on the hyperparameters for this case, see ``exp15_resnet_gridsearch``
cifar10_resnet32_adam = complete_config(
    {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.002511886431509582}},
        "problem_cls": cifar10_resnet32,
    }
)


CONFIGURATIONS = [
    fmnist_2c2d_sgd,
    fmnist_2c2d_adam,
    cifar10_3c3d_sgd,
    cifar10_3c3d_adam,
    cifar10_resnet32_sgd,
    cifar10_resnet32_adam,
    cifar100_allcnnc_sgd,
    cifar100_allcnnc_adam,
]


# ======================================================================================
# Converters
# Convert a config dict to a config string and vice versa.
# ======================================================================================
def config_to_config_str(config):
    """For a given configuration dict, return the configuration string. This is used
    by e.g. `run_eval.py` to turn the configuration dict into a command line argument.
    """
    problem_str = config["problem_cls"].__name__
    optimizer_str = config["optimizer_cls"].__name__.lower()

    return f"{problem_str}_{optimizer_str}"


def config_str_to_config(config_str):
    """For a given configuration string, return the actual configuration dict. This is
    used by e.g. `eval.py` to map the command line argument (generated by
    `config_to_config_str`) to the actual configuration dict.
    """

    mapping = {
        "fmnist_2c2d_sgd": fmnist_2c2d_sgd,
        "fmnist_2c2d_adam": fmnist_2c2d_adam,
        "cifar10_3c3d_sgd": cifar10_3c3d_sgd,
        "cifar10_3c3d_adam": cifar10_3c3d_adam,
        "cifar10_resnet32_sgd": cifar10_resnet32_sgd,
        "cifar10_resnet32_adam": cifar10_resnet32_adam,
        "cifar100_allcnnc_sgd": cifar100_allcnnc_sgd,
        "cifar100_allcnnc_adam": cifar100_allcnnc_adam,
    }
    try:
        return mapping[config_str]
    except KeyError as exc:
        raise ValueError(
            f"Config string {config_str} cannot be mapped to config dict"
        ) from exc
