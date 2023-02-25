"""Create run time plots of eigen-pair computation."""

from os import path
from typing import Dict, List

from matplotlib import pyplot as plt
from plot_utils_tol import add_tue_defaults
from run_time_evecs_power_tol import get_output_file as get_output_file_power
from run_time_evecs_tol import get_output_file as get_output_file_vivit
from shared_plot import COLORS
from tueplots.figsizes import neurips2022

from exp.exp10_performance_ggn_evals.call_run_time_evecs_power_tol import (
    configurations as configurations_power,
)
from exp.exp10_performance_ggn_evals.call_run_time_evecs_tol import (
    configurations as configurations_vivit,
)
from exp.utils.path import read_from_json
from exp.utils.plot import TikzExport

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
FIG_DIR = path.join(HEREDIR, "fig", "time", "evecs", "tol")

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
    max_iters: str = None,
) -> Dict[str, List[float]]:
    """Load run times and eigenvalues."""
    if which == "vivit":
        output_file = get_output_file_vivit(
            architecture, device, param_groups, computations, N, K
        )
    elif which == "power":
        output_file = get_output_file_power(
            architecture, device, param_groups, computations, N, K, tol, max_iters
        )
    else:
        raise ValueError

    return read_from_json(output_file)


def get_fig_savepath(
    architecture: str, device: str, param_groups: str, extension: str = ".tex"
) -> str:
    """Return save path of a figure."""
    return path.join(FIG_DIR, f"{architecture}_{device}_{param_groups}_tol{extension}")


def accuracy(evals_ground, evals_ref):
    return 1 - (abs(evals_ground - evals_ref) / abs(evals_ref)).mean()


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


if __name__ == "__main__":  # noqa: C901
    plots = set()

    for N, architecture, device, param_groups, _, k in configurations_vivit():
        if (N, architecture, device, param_groups, k) not in plots:
            key = (N, architecture, device, param_groups, k)
            plots.add(key)

    for plot_config in plots:
        print(plot_config)

        # get ground truth
        true_evals = None
        true_acc = None

        for (
            N,
            architecture,
            device,
            param_groups,
            computations,
            k,
        ) in configurations_vivit():
            key = (N, architecture, device, param_groups, k)
            if key == plot_config and computations == "full_batch_exact":
                data = load(
                    architecture, device, param_groups, computations, N, k, "vivit"
                )
                assert len(list(data["evals"].items())) == 1
                _, true_evals = list(data["evals"].items())[0]
                true_acc = accuracy(true_evals, true_evals)

        # access the ViViT result, keys are ViViTs approximations
        vivit_times = {}
        vivit_evals = {}
        vivit_accs = {}

        for (
            N,
            architecture,
            device,
            param_groups,
            computations,
            k,
        ) in configurations_vivit():
            key = (N, architecture, device, param_groups, k)
            if key == plot_config:
                data = load(
                    architecture, device, param_groups, computations, N, k, "vivit"
                )
                vivit_times[computations] = min(data["times"])
                assert len(list(data["evals"].items())) == 1
                _, evals = list(data["evals"].items())[0]
                vivit_evals[computations] = evals
                vivit_accs[computations] = accuracy(true_evals, evals)

        # access the power results, keys are power iteration tolerances
        power_times = {}
        power_evals = {}
        power_accs = {}

        for (
            N,
            architecture,
            device,
            param_groups,
            computations,
            k,
            tol,
            max_iters,
        ) in configurations_power():
            key = (N, architecture, device, param_groups, k)
            if key == plot_config:
                data = load(
                    architecture,
                    device,
                    param_groups,
                    computations,
                    N,
                    k,
                    "power",
                    tol,
                    max_iters,
                )
                power_times[tol] = min(data["times"])
                assert len(list(data["evals"].items())) == 1
                _, evals = list(data["evals"].items())[0]
                power_evals[tol] = evals
                power_accs[tol] = accuracy(true_evals, evals)

        print("\nGround truth")
        print(true_evals)

        print("\nViViT")
        print(vivit_times)
        print(vivit_accs)
        print(vivit_evals)

        print("\nPower")
        print(power_times)
        print(power_accs)
        print(power_evals)

        tue_config = neurips2022(nrows=1, ncols=1, rel_width=1.0)
        tue_config = add_tue_defaults(tue_config)
        with plt.rc_context(tue_config):
            plt.figure()
            plt.xlabel("time [s]")
            plt.ylabel("mean accuracy")
            plt.grid()

            # Title
            assert plot_config == ("128", "cifar10_3c3d", "cuda", "one_group", "10")
            plt.title("CIFAR-10, 3C3D network, $k=10$, $N=128$ on GPU")

            tmin = min(list(vivit_times.values()) + list(power_times.values()))
            tmax = max(list(vivit_times.values()) + list(power_times.values()))
            plt.hlines(1, tmin, tmax, linestyles="dotted", color="black")

            for computation in vivit_evals.keys():
                label = LABELS[computation]
                plt.plot(
                    vivit_times[computation],
                    vivit_accs[computation],
                    label=label,
                    color=COLORS[computation],
                    marker=MARKERS[computation],
                    ls=LINESTYLES[computation],
                )

            for idx, tol in enumerate(power_evals.keys()):
                label = LABELS["power iteration"] if idx == 0 else None

                default_tol = "0.0010000000474974513"  # default tolerance 1e-3
                marker = "*" if tol == default_tol else MARKERS["power iteration"]
                plt.plot(
                    power_times[tol],
                    power_accs[tol],
                    label=label,
                    color=COLORS["power iteration"],
                    marker=marker,
                    ls=LINESTYLES["power iteration"],
                )

        plt.legend()

        savepath = get_fig_savepath(architecture, device, param_groups, extension="")
        TikzExport().save_fig(savepath, tex_preview=False)

        plt.savefig(path.join(HEREDIR, "performance.pdf"))
        plt.close("all")
