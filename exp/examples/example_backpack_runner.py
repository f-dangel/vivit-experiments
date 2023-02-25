"""Demonstrate usage of a DeepOBS runner with BackPACK-compatible problem.

Runs SGD with same hyperparameters and two different runners:

1. DeepOBS standard runner (no closure)
2. BackpackRunner (uses closure)

Finally, make sure that the training metrics are identical.
"""

import os

from deepobs.pytorch.runners import StandardRunner
from torch.optim import SGD

from exp.utils.deepobs_runner import BackpackRunner

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)

hyperparam_names = {"lr": {"type": float, "default": 1.0}}
hyperparams = {"lr": 0.1}

outputs = []

for runner_cls in [BackpackRunner, StandardRunner]:
    runner = runner_cls(SGD, hyperparam_names)
    outputs.append(
        runner.run(
            testproblem="quadratic_deep",
            hyperparams=hyperparams,
            num_epochs=2,
            l2_reg=0,
            print_train_iter=True,
            train_log_interval=1,
            debug=False,
            output_dir=os.path.join(HEREDIR, "results"),
        )
    )

ref = outputs.pop()
while outputs:
    assert ref == outputs.pop(), "Outputs vary between runners."
