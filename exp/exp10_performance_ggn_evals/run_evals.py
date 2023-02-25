"""Compute and save the GGN spectrum under different approximations."""

import os

from shared import layerwise_group, one_group
from shared_evals import (
    compute_num_trainable_params,
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
    run_ggn_gram_evals,
)
from torch import cat

from exp.utils.deepobs import (
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
    set_seeds,
)
from exp.utils.path import write_to_json

# IO
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE_DIR, "results", "evals")

# Define settings
computations_cases = [full_batch_exact, full_batch_mc, frac_batch_exact, frac_batch_mc]
param_groups_cases = [one_group, layerwise_group]
architecture_cases = [
    cifar10_3c3d,
    fmnist_2c2d,
    cifar100_allcnnc,
    cifar10_resnet32,
    cifar10_resnet56,
]
device = "cpu"
num_params_key = "D"
batch_size_key = "N"


def get_output_file(architecture, device, param_groups):
    return os.path.join(
        DATA_DIR, f"{architecture.__name__}_{device}_{param_groups.__name__}.json"
    )


def get_batch_size(architecture):
    if architecture == cifar100_allcnnc:
        N = 64
    else:
        N = 128

    return N


if __name__ == "__main__":
    # compute eigenvalues, write results to file
    for architecture in architecture_cases:
        N = get_batch_size(architecture)
        D = compute_num_trainable_params(architecture)

        for param_groups in param_groups_cases:
            DATA_FILE = get_output_file(architecture, device, param_groups)
            if os.path.exists(DATA_FILE):
                print(
                    "[exp10] Skipping computation. "
                    + f"Output file already exists: {DATA_FILE}"
                )
                continue
            else:
                os.makedirs(DATA_DIR, exist_ok=True)
            DATA = {num_params_key: D, batch_size_key: N}

            for computations in computations_cases:
                set_seeds(0)
                group_evals = run_ggn_gram_evals(
                    architecture, param_groups, computations, N, device
                )
                evals = cat(list(group_evals.values())).cpu().numpy()
                DATA[computations.__name__] = evals

            write_to_json(DATA_FILE, DATA)
