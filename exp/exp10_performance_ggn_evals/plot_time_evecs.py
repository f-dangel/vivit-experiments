"""Create run time plots of eigen-pair computation."""

from os import path
from typing import Dict, Iterable, List, Tuple

from call_run_time_evecs import K_MAX, batch_sizes
from matplotlib import pyplot as plt
from run_time_evecs import get_output_file as get_output_file_vivit
from run_time_evecs_power import get_output_file as get_output_file_power
from shared_plot import COLORS

from exp.exp10_performance_ggn_evals.call_run_time_evecs import configurations_no_k
from exp.utils.path import read_from_json
from exp.utils.plot import TikzExport

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
FIG_DIR = path.join(HEREDIR, "fig", "time", "evecs")

###############################################################################
#                                 Data loading                                #
###############################################################################


def load(
    architecture: str,
    device: str,
    param_groups: str,
    computations: str,
    N: str,
    K: str,
    which: str,
) -> Dict[str, List[float]]:
    """Load run time results power iteration runs."""
    get_output_file = {"vivit": get_output_file_vivit, "power": get_output_file_power}[
        which
    ]
    output_file = get_output_file(
        architecture, device, param_groups, computations, N, K
    )
    return read_from_json(output_file)


def load_k_range(
    architecture: str,
    device: str,
    param_groups: str,
    computations: str,
    N: str,
    K_range: Iterable[int],
    which: str,
    allow_missing: bool,
) -> Iterable[Tuple[int, Dict[str, List[float]]]]:
    """Yield run time results for a range of top eigen-pairs."""
    for K in K_range:
        try:
            yield K, load(
                architecture, device, param_groups, computations, N, str(K), which
            )
        except FileNotFoundError as e:
            if not allow_missing:
                raise e


def load_min_times(
    architecture: str,
    device: str,
    param_groups: str,
    computations: str,
    N: str,
    K_range: Iterable[int],
    which: str,
    allow_missing: bool = False,
) -> Tuple[List[int], List[float]]:
    """Select best run time for a range of top eigen-pairs."""
    k_values = []
    min_times = []

    for k, k_data in load_k_range(
        architecture,
        device,
        param_groups,
        computations,
        N,
        K_range,
        which,
        allow_missing,
    ):
        k_values.append(k)
        min_times.append(min(k_data["times"]))

    return k_values, min_times


###############################################################################
#                               Plotting styles                               #
###############################################################################
MARKERS = {
    "power iteration": "p",
    "full_batch_exact": "o",
    "frac_batch_exact": "D",
    "full_batch_mc": "s",
    "frac_batch_mc": "v",
}
LINESTYLES = {
    "power iteration": "dashed",
    "full_batch_exact": "dashed",
    "frac_batch_exact": "dashed",
    "full_batch_mc": "dashed",
    "frac_batch_mc": "dashed",
}

LABELS = {
    "power iteration": "power iteration",
    "full_batch_exact": "mb, exact",
    "frac_batch_exact": "sub, exact",
    "full_batch_mc": "mb, mc",
    "frac_batch_mc": "sub, mc",
}

###############################################################################
#                               Plot one setting                              #
###############################################################################
ORDER = [
    ("full_batch_exact", "power"),
    ("full_batch_exact", "vivit"),
    ("frac_batch_exact", "vivit"),
    ("full_batch_mc", "vivit"),
    ("frac_batch_mc", "vivit"),
]


def plot(
    architecture: str, device: str, param_groups: str, allow_missing: bool = False
):
    K_RANGE = range(1, K_MAX + 1)
    N = batch_sizes[architecture]

    plt.figure()
    plt.title(f"{architecture}, N={N}, {device}, {param_groups}")
    plt.xlabel("top eigenpairs ($k$)")
    plt.ylabel("time [s]")
    plt.semilogy()

    for computations, which in ORDER:
        k_values, min_times = load_min_times(
            architecture,
            device,
            param_groups,
            computations,
            N,
            K_RANGE,
            which,
            allow_missing,
        )

        no_data = len(k_values) == 0 and len(min_times) == 0
        if no_data:
            continue

        computations = "power iteration" if which == "power" else computations
        label = LABELS[computations]

        plt.plot(
            k_values,
            min_times,
            label=label,
            color=COLORS[computations],
            marker=MARKERS[computations],
            ls=LINESTYLES[computations],
        )

    plt.legend()


def get_fig_savepath(
    architecture: str, device: str, param_groups: str, extension: str = ".tex"
) -> str:
    """Return save path of a figure."""
    return path.join(FIG_DIR, f"{architecture}_{device}_{param_groups}{extension}")


def configurations() -> Iterable[Tuple[str, str, str]]:
    """Yield all plotting configurations."""
    settings = {
        (architecture, device, param_groups)
        for _, architecture, device, param_groups, _ in configurations_no_k()
    }

    for architecture, device, param_groups in settings:
        yield architecture, device, param_groups


if __name__ == "__main__":
    for architecture, device, param_groups in configurations():
        plot(architecture, device, param_groups, allow_missing=True)

        savepath = get_fig_savepath(architecture, device, param_groups, extension="")
        TikzExport().save_fig(savepath, tex_preview=False)

        plt.close("all")
