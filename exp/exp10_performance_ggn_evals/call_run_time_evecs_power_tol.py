"""Execute power iteration benchmarks in separate python sessions."""

import os
from typing import Iterable, Tuple

from run_time_evecs_power_tol import __name__ as SCRIPT
from run_time_evecs_power_tol import get_output_file
from shared import one_group  # , layerwise_group
from shared_call import run
from torch import logspace

from exp.utils.deepobs import cifar10_3c3d

# Define settings
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

TOLS = [float(eps) for eps in logspace(-5, -1, 21)]
# large enough to not stop due to exceeded allowed iteration
MAXITERS = 1_000_000


def configurations() -> Iterable[Tuple[str, str, str, str, str, str, str, str]]:
    """Yield all configurations."""
    computations_fn = "full_batch_exact"

    for architecture in architecture_cases:
        N = batch_sizes[architecture]
        for device in device_cases:
            for param_groups in param_groups_cases:
                for tol in TOLS:
                    yield N, architecture, device, param_groups, computations_fn, str(
                        K
                    ), str(tol), str(MAXITERS)


if __name__ == "__main__":
    # Launch eigenpair run time benchmark which creates an output file
    computations_fn = "full_batch_exact"
    for (
        N,
        architecture,
        device,
        param_groups,
        computations,
        k,
        tol,
        max_iters,
    ) in configurations():
        DATA_FILE = get_output_file(
            architecture, device, param_groups, computations, N, k, tol, max_iters
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
            + f"tol          = {tol}\n"
            + f"max_iters    = {max_iters}\n"
        )

        cmd = [
            "python",
            f"{SCRIPT}.py",
            N,
            device,
            architecture,
            param_groups,
            k,
            tol,
            max_iters,
        ]

        run(cmd)
