"""DeepOBS grid search for the optimal learning rate using SGD with momentum on the
resnet32 architecture.
"""

import numpy as np
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch
from torch.optim import SGD

if __name__ == "__main__":
    print("Using SGD with momentum")

    # Define optimizer and search grid
    optimizer_class = SGD
    hyperparams = {
        "lr": {"type": float},
        "momentum": {"type": float},
        "nesterov": {"type": bool},
    }
    grid = {
        "lr": list(np.logspace(-5, 2, 36)),
        "momentum": [0.9],
        "nesterov": [False],
    }

    # Run grid search
    tuner = GridSearch(
        optimizer_class,
        hyperparams,
        grid,
        runner=StandardRunner,
        ressources=36,
    )
    tuner.tune("cifar10_resnet32", rerun_best_setting=False)
