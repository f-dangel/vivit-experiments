"""Determine critical batch size for Newton step computations.

Call every setting in a fresh python session for independence across runs.
"""

import functools
import os
from typing import Iterable, Tuple

from run_N_crit_newton_step import __name__ as SCRIPT
from run_N_crit_newton_step import (
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
)
from shared import layerwise_group, one_group
from shared_call import bisect, run_batch_size

from exp.utils.deepobs import (
    cifar10_3c3d,
    cifar10_resnet32,
    cifar10_resnet56,
    cifar100_allcnnc,
    fmnist_2c2d,
)
from exp.utils.path import write_to_json

# IO
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE_DIR, "results", "N_crit", "newton_step")
SCRIPT = f"{SCRIPT}.py"

# Define settings
computations_cases = [
    full_batch_exact.__name__,
    full_batch_mc.__name__,
    frac_batch_exact.__name__,
    frac_batch_mc.__name__,
]
param_groups_cases = [
    one_group.__name__,
    layerwise_group.__name__,
]
architecture_cases = [
    cifar10_3c3d.__name__,
    fmnist_2c2d.__name__,
    cifar100_allcnnc.__name__,
    cifar10_resnet32.__name__,
    cifar10_resnet56.__name__,
]
device_cases = ["cuda", "cpu"]

# search space
N_min, N_max = 1, 32768

# N_crit large such that Gram matrix diagonalization takes extremely long
EXCLUDE = [
    (cifar10_resnet32.__name__, "cpu", one_group.__name__),
    (cifar10_resnet32.__name__, "cpu", layerwise_group.__name__),
    (cifar10_resnet56.__name__, "cpu", one_group.__name__),
    (cifar10_resnet56.__name__, "cpu", layerwise_group.__name__),
]


def get_output_file(architecture: str, device: str, param_groups: str) -> str:
    """Return path of the output file."""
    return os.path.join(DATA_DIR, f"{architecture}_{device}_{param_groups}.json")


def configurations() -> Iterable[Tuple[str, str, str]]:
    """Yield all configurations."""
    for architecture in architecture_cases:
        for device in device_cases:
            for param_groups in param_groups_cases:
                if (architecture, device, param_groups) not in EXCLUDE:
                    yield architecture, device, param_groups


# find N_crit, write results to file
if __name__ == "__main__":
    for architecture, device, param_groups in configurations():
        DATA_FILE = get_output_file(architecture, device, param_groups)
        if os.path.exists(DATA_FILE):
            print(
                "[exp10] Skipping computation. "
                + f"Output file already exists: {DATA_FILE}"
            )
            continue
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
        DATA = {}

        for computations in computations_cases:
            run_func = functools.partial(
                run_batch_size,
                script=SCRIPT,
                device=device,
                architecture=architecture,
                param_groups=param_groups,
                computations=computations,
                show_full_stdout=False,
                show_full_stderr=False,
            )
            N_crit = bisect(run_func, N_min, N_max)
            DATA[computations] = N_crit
            print(f"overflow @ N > {N_crit}")

        write_to_json(DATA_FILE, DATA)
