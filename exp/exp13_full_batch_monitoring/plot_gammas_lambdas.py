"""This is a plotting script for one particular configuration specified via command line
argument. It can be called like this:
`python plot_gammas_lambdas.py --config_str fmnist_2c2d_sgd`
`run_plot_gammas_lambdas.py` automatically calls this script for all configurations.

This script consists of 3 steps:
1) Parse the command line argument and map to actual configuration.
2) Compute and store the plotting data (if neccessary). We compute the directional
   derivatives ({𝛾ₙₖ} and {𝜆ₙₖ}) using ViViT (using its approximations) and compute the
   SNRs of these quantities.
3) Compute and store the figure (if neccessary). This figure shows the SNRs of the
   quantities computed in 2).
"""

import os
from math import isnan

import numpy as np
import torch
from config import config_str_to_config
from deepobs.pytorch import testproblems
from eval import load_checkpoint
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plot_shared import get_xticks_labels
from run_plot_gammas_lambdas import get_plot_savedir
from torch import cuda, device
from tueplots import figsizes, fonts, fontsizes
from utils_shared import (
    check_cases,
    dump_json,
    eval_gammas_lambdas_pi,
    get_case_label,
    get_config_str_parser,
    get_deepobs_dataloader,
    load_json,
)

from exp.utils.plot import TikzExport


def get_plot_savepath(file_name, extension=".pdf"):
    """Get savepath for some result of the evaluation named `file_name`."""
    savedir = get_plot_savedir()
    return os.path.join(savedir, f"{file_name}{extension}")


DEVICE = device("cuda" if cuda.is_available() else "cpu")
VERBOSE = True
RECOMPUTE_DATA = False
RECOMPUTE_FIG = True
CHECK_DETERMINISTIC = True


# ======================================================================================
# Define case
# ======================================================================================
NOF_REPS = 3  # Number of samples/repetitions per case
BATCH_SIZE = 128
CASE = {
    "batch_size": BATCH_SIZE,
    "subsampling": None,
    "mc_samples": 0,
    "method": "PI",
}


def get_cases(config):
    all_cases = [CASE]

    # Add label and check all cases
    for case in all_cases:
        case["label"] = get_case_label(case)
    check_cases(all_cases)

    return all_cases


# ======================================================================================
# Computation of plotting data
# ======================================================================================
def eval_config(config, cases, json_path):
    """For the given configuration, evaluate all cases given by ``cases``. Store
    the results at ``json_path``. ``results`` is basically a copy of ``cases``, where
    each case has additionals keys for the results.
    """

    if VERBOSE:
        print("\nWorking on config = \n", config)

    problem_cls = config["problem_cls"]
    optimizer_cls = config["optimizer_cls"]
    num_classes = config["num_classes"]

    # Get data with batch size ``BATCH_SIZE`` for approximations (NOTE: all cases use
    # this batch size)
    torch.manual_seed(0)
    batch_data_list = list(get_deepobs_dataloader(problem_cls, BATCH_SIZE))[0:NOF_REPS]

    results = []
    for case in cases:
        gammas_snr = np.zeros((NOF_REPS, len(config["checkpoints"]), num_classes))
        lambdas_snr = np.zeros_like(gammas_snr)
        lambdas_mean = np.zeros_like(lambdas_snr)

        for checkpoint_idx, checkpoint in enumerate(config["checkpoints"]):
            if VERBOSE:
                case_label = case["label"]
                print(
                    f"Working on case {case_label} checkpoint {checkpoint}",
                    flush=True,
                )

            # Load checkpoint data (i.e. model and loss-function)
            checkpoint_data = load_checkpoint(problem_cls, optimizer_cls, checkpoint)
            if checkpoint_data is None:
                print("No checkpoint data was found. Skipping computations.")
                continue
            model = checkpoint_data.pop("model")
            loss_function = checkpoint_data.pop("loss_func")  # Must use ``mean``

            # Compute gammas and lambdas for each batch
            for batch_idx in range(NOF_REPS):
                gamma_nk, lambda_nk = eval_gammas_lambdas_pi(
                    model,
                    loss_function,
                    batch_data_list[batch_idx],
                    num_classes,
                    DEVICE,
                    check_deterministic=CHECK_DETERMINISTIC,
                    verbose=True,
                )

                gamma_nk = gamma_nk.cpu().numpy()
                lambda_nk = lambda_nk.cpu().numpy()

                gammas_snr[batch_idx, checkpoint_idx, :] = eval_SNR(gamma_nk)
                lambdas_snr[batch_idx, checkpoint_idx, :] = eval_SNR(lambda_nk)
                lambdas_mean[batch_idx, checkpoint_idx, :] = np.mean(lambda_nk, axis=0)

        # Store results in case dict
        case["gammas_snr"] = gammas_snr.tolist()
        case["lambdas_snr"] = lambdas_snr.tolist()
        case["lambdas_mean"] = lambdas_mean.tolist()
        results.append(case)

    dump_json(results, json_path)
    return results


# ======================================================================================
# Plotting
# ======================================================================================
def eval_SNR(input):
    """Compute the signal-to-noise-ratio (SNR) by taking the squared sample mean of
    ``input`` and divide it by the sample variance. The input's first dimension is
    assumed to be the (batch) dimension.
    """

    BATCH_DIM_IDX = 0
    signal2 = np.mean(input, axis=BATCH_DIM_IDX) ** 2
    noise = np.var(input, axis=BATCH_DIM_IDX, ddof=1)
    return np.divide(signal2, noise)


def plot(config, plot_data, figpath_1, figpath_2, plot_every):
    """Plot all cases in one figure."""

    assert len(plot_data) == 1, "We consider only the standard case"
    case = plot_data[0]

    # We consider only one batch
    # Dimensions: NOF_REPS, len(config["checkpoints"]), num_classes
    BATCH_IDX = 0
    gammas_snr = np.array(case["gammas_snr"])[BATCH_IDX]
    lambdas_snr = np.array(case["lambdas_snr"])[BATCH_IDX]
    lambdas_mean = np.array(case["lambdas_mean"])[BATCH_IDX]

    # Map input to color
    def get_color(input_vec):
        """Map every entry of the input vector to a color, where ``input_min`` is mapped
        to ``colors[0]`` and ``input_max`` is mapped to ``colors[1]``. We use linear
        interpolation in between.
        """
        colors = ["#000000", "#eb8775"]  # black & light red

        # Interploate linearly between the two colors
        cm = LinearSegmentedColormap.from_list("Custom", colors)
        input_len = len(input_vec)
        linear_colors = [cm(i / (input_len - 1)) for i in range(input_len)]

        # Arange the colors such that the smallest entry in ``input_vec`` is mapped to
        # ``linear_colors[0]`` and the largest to ``linear_colors[-1]``
        color_idx = np.argsort(np.argsort(input_vec))
        return [linear_colors[i] for i in color_idx]

    # Determine alpha
    alpha = 0.9
    problem_cls = config["problem_cls"]
    if problem_cls in [testproblems.cifar100_allcnnc]:
        alpha = 0.3

    # tueplot settings and context
    icml_size = figsizes.icml2022(column="half")
    font_config = fonts.icml2022_tex()
    font_sizes = fontsizes.icml2022()
    with plt.rc_context({**icml_size, **font_config, **font_sizes}):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        g_snr, l_snr, checkpoints, colors = [], [], [], []

        for checkpoint_idx in range(0, len(config["checkpoints"]), plot_every):
            cp = config["decimal_checkpoints"][checkpoint_idx] + 1

            g_snr += gammas_snr[checkpoint_idx].tolist()
            l_snr += lambdas_snr[checkpoint_idx].tolist()
            checkpoints += (cp * np.ones_like(gammas_snr[checkpoint_idx])).tolist()
            colors += get_color(lambdas_mean[checkpoint_idx])

        keep_g = [idx for idx, snr in enumerate(g_snr) if not isnan(snr)]
        ax1.scatter(
            [checkpoints[idx] for idx in keep_g],
            [g_snr[idx] for idx in keep_g],
            marker="o",
            c=[colors[idx] for idx in keep_g],
            s=1.0,
            alpha=alpha,
        )

        keep_l = [idx for idx, snr in enumerate(l_snr) if not isnan(snr)]
        ax2.scatter(
            [checkpoints[idx] for idx in keep_l],
            [l_snr[idx] for idx in keep_l],
            marker="o",
            c=[colors[idx] for idx in keep_l],
            s=1.0,
            alpha=alpha,
        )

        ax1.set_ylabel(r"SNR[$\gamma_{nk}$] (log scale)")
        ax2.set_ylabel(r"SNR[$\lambda_{nk}$] (log scale)")
        for ax in [ax1, ax2]:
            ax.set_xlabel("epoch (log scale)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(  # Remove minor ticks
                axis="y", which="minor", left=False, right=False, labelleft=False
            )

            # Ticks (shift epochs back, see above) and grid
            xticks, xticklabels = get_xticks_labels(config["num_epochs"])
            xticks = (np.array(xticks) + 1).astype(int).tolist()
            ax.tick_params(  # Remove minor ticks
                axis="x", which="minor", bottom=False, top=False, labelbottom=False
            )
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            # Additional settings
            ax.grid(which="major", ls="dashed", lw=0.4, dashes=(7, 7))
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_visible(True)
                ax.spines[axis].set_color("k")
                ax.spines[axis].set_linewidth(0.4)
            ax.xaxis.set_tick_params(width=0.5, direction="in", length=5)
            ax.yaxis.set_tick_params(width=0.5, direction="in", length=5)
            [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
            [t.set_color("k") for t in ax.yaxis.get_ticklabels()]

        fig1.savefig(figpath_1)

        TikzExport().save_fig(
            figpath_1.replace(".pdf", ""),
            png_preview=True,
            tex_preview=False,
            fig=fig1,
        )
        fig2.savefig(figpath_2)
        TikzExport().save_fig(
            figpath_2.replace(".pdf", ""),
            png_preview=True,
            tex_preview=False,
            fig=fig2,
        )
        plt.close(fig1)
        plt.close(fig2)


# ======================================================================================
# Main-function: Coordinate the computation of the plotting data and the figure
# ======================================================================================
if __name__ == "__main__":
    # Parse command line argument
    parser = get_config_str_parser()
    args = parser.parse_args()
    config_str = args.config_str
    print(f"\nconfig_str = {config_str}")
    config = config_str_to_config(config_str)

    # Set up and check cases
    cases = get_cases(config)
    if VERBOSE:
        print(f"\ncases (total: {len(cases)}):")
        for case in cases:
            print(case)

    # Compute plotting data if necessary
    json_path = get_plot_savepath(config_str + "_plot_data", extension=".json")
    if VERBOSE:
        print(f"\nChecking for json file at {json_path}")

    if not os.path.exists(json_path) or RECOMPUTE_DATA:
        print("Computing plotting data.")
        plot_data = eval_config(config, cases, json_path)
    else:
        print(f"Skipping computation. Using existing file {json_path}")
        plot_data = load_json(json_path)

    # Compute figure if necessary
    figpath_1 = get_plot_savepath(config_str + "_gammas", extension=".pdf")
    figpath_2 = get_plot_savepath(config_str + "_lambdas", extension=".pdf")
    if VERBOSE:
        print(f"\nChecking for figures at {figpath_1} and {figpath_2}.")

    if not os.path.exists(figpath_1) or not os.path.exists(figpath_2) or RECOMPUTE_FIG:
        print("Computing figure.")
        plot_every = 3 if "cifar100_allcnnc" in config_str else 1
        plot(config, plot_data, figpath_1, figpath_2, plot_every)
    else:
        print(f"Skipping computation. Using existing files {figpath_1} and {figpath_2}")
