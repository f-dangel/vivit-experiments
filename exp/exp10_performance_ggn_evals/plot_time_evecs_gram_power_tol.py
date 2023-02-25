"""Create run time plots of eigen-pair computation."""

from os import path
from typing import Dict, Iterable, List, Tuple

from call_run_time_evecs_gram_power_tol import K_MAX, batch_sizes, configurations_no_k
from matplotlib import pyplot as plt
from run_time_evecs_gram_power_tol import get_output_file as get_output_file_gram_power
from run_time_evecs_power import get_output_file as get_output_file_power
from shared_plot import COLORS

from exp.utils.path import read_from_json
from exp.utils.plot import TikzExport

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
FIG_DIR = path.join(HEREDIR, "fig", "time", "evecs", "gram_tol")

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
    tol: str = None,
    max_iter: str = None,
) -> Dict[str, List[float]]:
    """Load run time results power iteration runs."""
    assert which in {"gram_power", "power"}

    if which == "gram_power":
        output_file = get_output_file_gram_power(
            architecture, device, param_groups, computations, N, K, tol, max_iter
        )
    else:
        output_file = get_output_file_power(
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
    tol: str = None,
    max_iter: str = None,
) -> Iterable[Tuple[int, Dict[str, List[float]]]]:
    """Yield run time results for a range of top eigen-pairs."""
    for K in K_range:
        try:
            yield K, load(
                architecture,
                device,
                param_groups,
                computations,
                N,
                str(K),
                which,
                tol,
                max_iter,
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
    tol: str = None,
    max_iter: str = None,
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
        tol,
        max_iter,
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
    "power iteration": "GGN power iteration",
    "full_batch_exact": "(mb, exact) + power",
    "frac_batch_exact": "(sub, exact) + power",
    "full_batch_mc": "(mb, mc) + power",
    "frac_batch_mc": "(sub, mc) + power",
}

###############################################################################
#                               Plot one setting                              #
###############################################################################
ORDER = [
    ("full_batch_exact", "power"),
    ("full_batch_exact", "gram_power"),
    ("frac_batch_exact", "gram_power"),
    ("full_batch_mc", "gram_power"),
    ("frac_batch_mc", "gram_power"),
]


def plot(
    architecture: str,
    device: str,
    param_groups: str,
    allow_missing: bool = False,
    tol: str = None,
    max_iter: str = None,
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
            tol,
            max_iter,
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
    architecture: str,
    device: str,
    param_groups: str,
    tol: str,
    max_iter: str,
    extension: str = ".tex",
) -> str:
    """Return save path of a figure."""
    return path.join(
        FIG_DIR,
        f"{architecture}_{device}_{param_groups}_tol_{tol.replace('.', '_')}"
        + f"_max_iter_{max_iter}{extension}",
    )


def configurations() -> Iterable[Tuple[str, str, str, str, str]]:
    """Yield all plotting configurations."""
    yield from {
        (architecture, device, param_groups, tol, max_iter)
        for (
            _,
            device,
            architecture,
            param_groups,
            _,
            tol,
            max_iter,
        ) in configurations_no_k()
    }


if __name__ == "__main__":
    for architecture, device, param_groups, tol, max_iter in configurations():
        plot(
            architecture,
            device,
            param_groups,
            allow_missing=False,
            tol=tol,
            max_iter=max_iter,
        )
        savepath = get_fig_savepath(
            architecture, device, param_groups, tol, max_iter, extension=""
        )
        TikzExport().save_fig(savepath, tex_preview=False)
        plt.close("all")
