"""DeepOBS grid search for the optimal learning rate using Adam on the resnet32 
architecture.
"""

import numpy as np
from deepobs.pytorch.runners import StandardRunner
from deepobs.tuner import GridSearch
from torch.optim import Adam

if __name__ == "__main__":
    print("Using Adam")

    # Define optimizer and search grid
    optimizer_class = Adam
    hyperparams = {"lr": {"type": float}}
    grid = {"lr": list(np.logspace(-5, 2, 36))}

    # Run grid search
    tuner = GridSearch(
        optimizer_class,
        hyperparams,
        grid,
        runner=StandardRunner,
        ressources=36,
    )
    tuner.tune("cifar10_resnet32", rerun_best_setting=False)
