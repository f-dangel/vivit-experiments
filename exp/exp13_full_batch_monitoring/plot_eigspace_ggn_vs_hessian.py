"""In this script, we compare the full-batch Hessian and full-batch GGN in terms of the
overlap of their top-C eigenspaces and the corresponding eigenvalues. It is called like
this: `python plot_eigspace_ggn_vs_hessian.py`.
"""
import os

import numpy as np
import torch
from config import CONFIGURATIONS, PLOT_OUTPUT, config_to_config_str
from matplotlib import pyplot as plt
from plot_shared import get_xticks_labels
from tueplots import figsizes, fonts, fontsizes
from utils_shared import (
    dump_json,
    load_eval_result,
    load_json,
    subspaces_overlap,
    tensor_to_list,
)

from exp.utils.plot import TikzExport

PLOT_SUBDIR = os.path.join(PLOT_OUTPUT, "eigspace_ggn_vs_hessian")
VERBOSE = True
RECOMPUTE_DATA = False
RECOMPUTE_FIG = True


def get_plot_savedir():
    """Return sub-directory where results of the plotting script are saved."""
    savedir = PLOT_SUBDIR
    os.makedirs(savedir, exist_ok=True)
    return savedir


def get_plot_savepath(file_name, extension=".pdf"):
    """Get savepath for some result of the evaluation named `file_name`."""
    savedir = get_plot_savedir()
    return os.path.join(savedir, f"{file_name}{extension}")


# ======================================================================================
# Computation of plotting data
# ======================================================================================
def eval_config(config, json_path):
    """For the given configuration, go through all checkpoints and compute the overlap
    between the full-batch eigenspaces. Also extract the full-batch eigenvalues of the
    GGN and Hessian. Store everything in a dict at ``json_path``."""

    problem_cls = config["problem_cls"]
    optimizer_cls = config["optimizer_cls"]

    overlaps = []
    G_evals = []
    H_evals = []

    # Gather results for all checkpoints
    for checkpoint in config["checkpoints"]:
        # GGN
        file_name = "topC_fb_GGN"
        topC_fb_GGN = load_eval_result(
            problem_cls, optimizer_cls, checkpoint, file_name
        )

        if topC_fb_GGN is None:
            print("Result not found. Skipping this checkpoint.")
            continue
        G_evecs = topC_fb_GGN["G_evecs"]
        G_evals.append(tensor_to_list(topC_fb_GGN["G_evals"]))

        # Hessian
        file_name = "topC_fb_Hessian"
        topC_fb_Hessain = load_eval_result(
            problem_cls, optimizer_cls, checkpoint, file_name
        )

        if topC_fb_Hessain is None:
            print("Result not found. Skipping this checkpoint.")
            continue
        H_evecs = topC_fb_Hessain["H_evecs"]
        H_evals.append(tensor_to_list(topC_fb_Hessain["H_evals"]))

        # Compute overlap of eigenspaces
        overlaps.append(subspaces_overlap(G_evecs, H_evecs, config["num_classes"]))

    # Store results in a json file
    plot_data = {"overlaps": overlaps, "G_evals": G_evals, "H_evals": H_evals}
    dump_json(plot_data, json_path)
    return plot_data


# ======================================================================================
# Plotting
# ======================================================================================
def plot_1(config, plot_data, fig_path):
    """Plot overlaps between eigenspaces of GGN and Hessian."""

    # Extract results
    overlaps = plot_data["overlaps"]

    if not len(overlaps) == len(config["decimal_checkpoints"]):
        print("Not enough results were found. Skipping this config.")
        return

    col_overlap = "#257d59"

    # Eigenspace overlap: tueplot settings and context
    icml_size = figsizes.icml2022(column="half")
    font_config = fonts.icml2022_tex()
    font_sizes = fontsizes.icml2022()
    with plt.rc_context({**icml_size, **font_config, **font_sizes}):
        fig, ax = plt.subplots()

        ax.axhline(y=1.0, color="k", ls="--", lw=0.5)
        # ax.axhline(y=0.0, color="k", ls="--", lw=0.5)
        ax.plot(
            np.array(config["decimal_checkpoints"]) + 1,  # shifted! (log-plot)
            overlaps,
            "o",
            ms=1.0,
            color=col_overlap,
        )

        ax.set_xlabel("epoch (log scale)")
        ax.set_ylabel("overlap")
        ax.set_xscale("log")
        # ax.set_ylim([-0.02, 1.02])

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

        fig.savefig(fig_path)
        TikzExport().save_fig(
            fig_path.replace(".pdf", ""), png_preview=True, tex_preview=False
        )
        plt.close(fig)


def plot_2(config, plot_data, fig_path):
    """Plot eigenvalues of GGN and Hessian."""

    # Extract results
    G_evals = plot_data["G_evals"]
    H_evals = plot_data["H_evals"]

    if not len(G_evals) == len(config["decimal_checkpoints"]):
        print("Not enough results were found. Skipping this config.")
        return

    col_G = "#eb8775"
    col_H = "#71c5e3"

    # Eigenvalues: tueplot settings and context
    icml_size = figsizes.icml2022()
    font_config = fonts.icml2022_tex()
    font_sizes = fontsizes.icml2022()
    with plt.rc_context({**icml_size, **font_config, **font_sizes}):
        fig, ax = plt.subplots()

        G_label = "GGN"
        H_label = "Hessian"
        for decimal_cp, G_ev, H_ev in zip(
            config["decimal_checkpoints"], G_evals, H_evals
        ):
            epoch_vec = (decimal_cp + 1) * torch.ones(len(G_ev))
            ax.plot(epoch_vec, G_ev, "o", ms=2.0, color=col_G, label=G_label)
            ax.plot(epoch_vec, H_ev, "x", ms=2.0, color=col_H, label=H_label)
            G_label, H_label = None, None

        ax.set_xscale("log")
        ax.set_xlabel("epoch (log scale)")
        ax.set_ylabel("eigenvalue")

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

        leg = ax.legend()
        leg.get_frame().set_linewidth(0.5)

        fig.savefig(fig_path)
        TikzExport().save_fig(
            fig_path.replace(".pdf", ""), png_preview=True, tex_preview=False
        )
        plt.close(fig)


# ======================================================================================
# Main-function: Coordinate the computation of the plotting data and the figures
# ======================================================================================
if __name__ == "__main__":
    for config in CONFIGURATIONS:
        config_title = config_to_config_str(config)
        print(f"\nConfig {config_title}")

        # Compute plotting data if necessary
        json_path = get_plot_savepath(config_title + "_plot_data", extension=".json")
        if VERBOSE:
            print(f"\nChecking for json file at {json_path}")

        if not os.path.exists(json_path) or RECOMPUTE_DATA:
            print("Computing plotting data.")
            plot_data = eval_config(config, json_path)
        else:
            print(f"Skipping computation. Using existing file {json_path}")
            plot_data = load_json(json_path)

        # Compute figure 1 if necessary
        fig_path = get_plot_savepath(config_title + "_eigenspace", extension=".pdf")
        if VERBOSE:
            print(f"\nChecking for figure at {fig_path}")

        if not os.path.exists(fig_path) or RECOMPUTE_FIG:
            print("Computing figure.")
            plot_1(config, plot_data, fig_path)
        else:
            print(f"Skipping computation. Using existing file {fig_path}")

        # Compute figure 2 if necessary
        fig_path = get_plot_savepath(config_title + "_eigenvals", extension=".pdf")
        if VERBOSE:
            print(f"\nChecking for figure at {fig_path}")

        if not os.path.exists(fig_path) or RECOMPUTE_FIG:
            print("Computing figure.")
            plot_2(config, plot_data, fig_path)
        else:
            print(f"Skipping computation. Using existing file {fig_path}")
