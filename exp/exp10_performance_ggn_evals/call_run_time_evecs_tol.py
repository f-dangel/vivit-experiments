"""Execute time benchmarks in a separate python session for each config."""

import os
from typing import Iterable, Tuple

from run_time_evecs_tol import __name__ as SCRIPT
from run_time_evecs_tol import get_output_file
from shared import one_group  # , layerwise_group
from shared_call import run
from shared_evecs import (
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
)

from exp.utils.deepobs import cifar10_3c3d

# Define settings
computations_cases = [
    full_batch_exact.__name__,
    full_batch_mc.__name__,
    frac_batch_exact.__name__,
    frac_batch_mc.__name__,
]
param_groups_cases = [
    one_group.__name__,
    # layerwise_group.__name__,
]
architecture_cases = [
    cifar10_3c3d.__name__,
    # fmnist_2c2d.__name__,
    # cifar100_allcnnc.__name__,
    # cifar10_resnet32.__name__,
    # cifar10_resnet56.__name__,
]
device_cases = [
    "cuda",
    # "cpu",
]

batch_sizes = {
    cifar10_3c3d.__name__: "128",
    # fmnist_2c2d.__name__: "128",
    # cifar100_allcnnc.__name__: "64",
    # cifar10_resnet32.__name__: "128",
    # cifar10_resnet56.__name__: "128",
}
K = 10


def configurations() -> Iterable[Tuple[str, str, str, str, str, str]]:
    """Yield all configurations."""
    for architecture in architecture_cases:
        N = batch_sizes[architecture]
        for device in device_cases:
            for param_groups in param_groups_cases:
                for computations in computations_cases:
                    yield N, architecture, device, param_groups, computations, str(K)


if __name__ == "__main__":
    # Launch eigenpair run time benchmark which creates an output file
    for N, architecture, device, param_groups, computations, k in configurations():
        DATA_FILE = get_output_file(
            architecture, device, param_groups, computations, N, k
        )
        if os.path.exists(DATA_FILE):
            print(
                "[exp10] Skipping computation. "
                + f"Output file already exists: {DATA_FILE}"
            )
            continue

        print(
            f"\narchitecture = {architecture}\n"
            + f"param_groups = {param_groups}\n"
            + f"computations = {computations}\n"
            + f"device       = {device}\n"
            + f"N            = {N}\n"
            + f"K            = {k}\n"
        )

        cmd = [
            "python",
            f"{SCRIPT}.py",
            N,
            device,
            architecture,
            param_groups,
            computations,
            k,
        ]

        run(cmd)
