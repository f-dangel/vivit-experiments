"""Extract the optimal learning rate for SGD with momentum and Adam from the results."""

import os

from deepobs.analyzer.analyze import (
    get_performance_dictionary,
    plot_hyperparameter_sensitivity,
)
from matplotlib import pyplot as plt

METRIC = "test_accuracies"
MODE = "best"  # The best test accuracy among all training epochs
# MODE = "final"  # The test accuracy at the end of training
SHOW_FIG = True

# Paths
HERE = os.path.abspath(__file__)
HERE_DIR = os.path.dirname(HERE)
TESTPROBLEM_DIR = os.path.join(HERE_DIR, "results", "cifar10_resnet32")
SGD_DIR = os.path.join(TESTPROBLEM_DIR, "SGD")
ADAM_DIR = os.path.join(TESTPROBLEM_DIR, "Adam")

# Plot performance over learning rate for both optimizers
fig, ax = plt.subplots(1, 1)
plot_hyperparameter_sensitivity(
    TESTPROBLEM_DIR,
    hyperparam="lr",
    metric=METRIC,
    mode=MODE,
    xscale="log",
    plot_std=False,
    show=False,
    ax=ax,
)

# Get best hypterparameters
print("===== ANALYSIS =====")
print(f"METRIC = {METRIC}")
print(f"MODE = {MODE}")
for opt_name, opt_dir, ls in zip(["SGD", "Adam"], [SGD_DIR, ADAM_DIR], ["--", "-"]):
    # Extract info from performance dict
    performance_dic = get_performance_dictionary(opt_dir, metric=METRIC, mode=MODE)
    performance = performance_dic["Performance"]
    lr = performance_dic["Hyperparameters"]["lr"]

    print(f"{opt_name:7}: lr = {lr:23}, performance = {performance:.3f}")

    # Also plot in figure
    ax.plot(
        [0, lr],
        [performance, performance],
        ls=ls,
        color="r",
        label=f"lr* ({opt_name})",
    )
    ax.plot([lr, lr], [0, performance], ls=ls, color="r")

# Save and show performance plot
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(TESTPROBLEM_DIR, f"performance_MODE_{MODE}.pdf"))
if SHOW_FIG:
    plt.show()
