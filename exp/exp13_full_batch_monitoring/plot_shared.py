"""This file implements functionality that is shared among all plotting scripts."""

from warnings import warn

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from tueplots import figsizes, fonts, fontsizes
from utils_shared import get_case_label

from exp.utils.plot import TikzExport


def get_config_title(config):
    """Turn configuration into a title (for a matplotlib figure)"""
    problem_str = config["problem_cls"].__name__
    optimizer_str = config["optimizer_cls"].__name__

    return f"{problem_str} ({optimizer_str})"


def plot_stacked_bar(ax, labels, data):
    """A stacked bar plot with x-labels `labels` for the C x nof_checkpoints numpy array
    `data`. One column in `data` corresponds to one bar."""

    def color_fader(c1, c2, mix=0):
        """Fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)"""
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    c1 = "#e0d1d1"
    c2 = "#851010"
    num_rows = len(data[:, 0])

    # Plot the first bar for all checkpoints
    ax.bar(labels, data[0, :], color=c1)
    bottom_data = data[0, :]

    # Subsequent bars (update bottom_data)
    for row_idx in range(1, data.shape[0]):
        ax.bar(
            labels,
            data[row_idx, :],
            bottom=bottom_data,
            color=color_fader(c1, c2, row_idx / (num_rows - 1)),
        )
        bottom_data += data[row_idx, :]


def get_xticks_labels(nof_epochs):
    """Get x-ticks and x-ticklabels for training with ``nof_epochs`` epochs. Note that
    ``labels_at`` must be a subset of ``xticks``.
    """

    if nof_epochs == 100:  # conv nets on cifar10, fmnist
        xticks_1 = np.arange(0, 10)
        xticks_2 = np.arange(10, 101, 10)
        xticks = np.concatenate((xticks_1, xticks_2)).tolist()
        labels_at = [0, 1, 5, 10, 20, 50, 100]
    elif nof_epochs == 180:  # resnet on cifar10
        xticks_1 = np.arange(0, 10)
        xticks_2 = np.arange(10, 100, 10)
        xticks_3 = np.arange(100, 181, 20)
        xticks = np.concatenate((xticks_1, xticks_2, xticks_3)).tolist()
        labels_at = [0, 1, 5, 10, 20, 50, 100, 180]
    elif nof_epochs == 350:  # conv net on cifar100
        xticks_1 = np.arange(0, 10)
        xticks_2 = np.arange(0, 100, 10)
        xticks_3 = np.arange(100, 351, 50)
        xticks = np.concatenate((xticks_1, xticks_2, xticks_3)).tolist()
        labels_at = [0, 1, 5, 10, 50, 100, 200, 350]
    else:
        raise NotImplementedError("")

    xticklabels = ["" for _ in range(len(xticks))]
    for num in labels_at:
        index = xticks.index(num)
        xticklabels[index] = f"{num}"

    return xticks, xticklabels


def get_case_color_marker(case):
    """Get color and marker based on case."""

    black_o = ("#000000", "o")
    teal_D = ("#469990", "D")
    orange_s = ("#de9f16", "s")
    purple_v = ("#802f99", "v")

    bs = case["batch_size"]
    sub = case["subsampling"]
    mc = case["mc_samples"]

    if sub is None and mc == 0:  # only bs
        mapping = {2: purple_v, 8: orange_s, 32: teal_D, 128: black_o}
        try:
            return mapping[bs]
        except KeyError:
            warn(f"Could not map bs={bs} to color-marker-pair. Returning (black, o)")
            return black_o

    if sub is not None and mc == 0:  # only bs & sub
        return teal_D
    if sub is None and mc != 0:  # only bs & mc
        return orange_s
    if sub is not None and mc != 0:  # bs, sub & mc
        return purple_v


def plot_overlap_cases(config, cases, fig_path, legend=True, nof_legend_cols=1):
    """Plot a list of cases ``cases``. Save figure at fig_path."""

    # tueplot settings and context
    icml_size = figsizes.icml2022(column="half")
    font_config = fonts.icml2022_tex()
    font_sizes = fontsizes.icml2022()
    with plt.rc_context({**icml_size, **font_config, **font_sizes}):
        fig, ax = plt.subplots()
        ax.axhline(y=1.0, color="k", ls="--", lw=0.5)
        ax.axhline(y=0.0, color="k", ls="--", lw=0.5)

        for case in cases:
            overlaps = np.array(case["overlaps"])
            nof_reps = overlaps.shape[0]
            label = get_case_label(case)  # update case label
            color, marker = get_case_color_marker(case)
            for batch_idx in range(nof_reps):
                ax.plot(
                    np.array(config["decimal_checkpoints"]) + 1,  # shifted! (log-plot)
                    overlaps[batch_idx, :],
                    marker,
                    color=color,
                    ms=1.0,
                    label=label,
                    alpha=0.6,
                )
                label = None

        ax.set_xlabel("epoch (log scale)")
        ax.set_ylabel("overlap")
        ax.set_xscale("log")

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

        if legend:
            leg = ax.legend(ncol=nof_legend_cols)
            leg.get_frame().set_linewidth(0.5)

        fig.savefig(fig_path)
        TikzExport().save_fig(
            fig_path.replace(".pdf", ""), png_preview=True, tex_preview=False
        )
        plt.close(fig)
