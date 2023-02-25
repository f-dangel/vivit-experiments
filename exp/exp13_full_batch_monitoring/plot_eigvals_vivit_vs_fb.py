"""This is a plotting script for one particular configuration specified via command
line argument. It can be called like this:
`python plot_eigvals_vivit_vs_fb.py --config_str fmnist_2c2d_sgd`
`run_plot_eigvals_vivit_vs_fb.py` automatically calls this script for all
configurations.

This script consists of 3 steps:
1) Parse the command line argument and map to actual configuration.
2) Compute and store the plotting data (if neccessary). We compute the eigenvalues using
   ViViT (using its approximations) and compare this to the full-batch directional
   derivative.
3) Compute and store the figure (if neccessary). This figure shows a metric for the
   error between the quantities computed in 2).
"""

import os

import numpy as np
import torch
from config import _add_decimal_checkpoints, config_str_to_config
from deepobs.pytorch import testproblems
from eval import load_checkpoint
from matplotlib import pyplot as plt
from plot_shared import get_case_color_marker, get_xticks_labels
from run_plot_eigvals_vivit_vs_fb import get_plot_savedir
from torch import cuda, device
from tueplots import figsizes, fonts, fontsizes
from utils_shared import (
    check_cases,
    directional_derivatives,
    dump_json,
    eval_eigspace_pi,
    eval_eigspace_vivit,
    get_case_label,
    get_config_str_parser,
    get_deepobs_dataloader,
    load_json,
    tensor_to_list,
)

from exp.utils.deepobs import get_deterministic_deepobs_train_loader
from exp.utils.plot import TikzExport
from vivit.hessianfree import GGNLinearOperator


def get_plot_savepath(file_name, extension=".pdf"):
    """Get savepath for some result of the evaluation named `file_name`."""
    savedir = get_plot_savedir()
    return os.path.join(savedir, f"{file_name}{extension}")


DEVICE = device("cuda" if cuda.is_available() else "cpu")
VERBOSE = True
RECOMPUTE_DATA = False
RECOMPUTE_FIG = True
CHECK_DETERMINISTIC_BATCH = True
CHECK_DETERMINISTIC_FB = False


# ======================================================================================
# Define cases
# ======================================================================================
BATCH_SIZE = 128
STD_CASES = [
    {
        "batch_size": BATCH_SIZE,
        "subsampling": None,
        "mc_samples": 0,
        "method": "PI",
    },
    {
        "batch_size": BATCH_SIZE,
        "subsampling": list(range(16)),
        "mc_samples": 0,
        "method": "ViViT",
    },
]


def get_cases(config):
    """Extend the standard cases ``STD_CASES`` (used by all configurations) by some
    configuration-specific case(s). Check that the given method is reasonable by
    calling ``check_cases`` and add the case label.
    """

    problem_cls = config["problem_cls"]
    if problem_cls in [
        testproblems.cifar10_3c3d,
        testproblems.fmnist_2c2d,
        testproblems.cifar10_resnet32,
    ]:
        mc_samples = 1
    elif problem_cls in [testproblems.cifar100_allcnnc]:
        mc_samples = 10

    extra_cases = [
        {
            "batch_size": BATCH_SIZE,
            "subsampling": None,
            "mc_samples": mc_samples,
            "method": "ViViT",
        },
        {
            "batch_size": BATCH_SIZE,
            "subsampling": list(range(16)),
            "mc_samples": mc_samples,
            "method": "ViViT",
        },
    ]

    all_cases = STD_CASES + extra_cases

    # Add label and check all cases
    for case in all_cases:
        case["label"] = get_case_label(case)
    check_cases(all_cases)

    return all_cases


def get_nof_reps(config):
    """Choose number of repetitions per case based on ``config``."""

    problem_cls = config["problem_cls"]
    if problem_cls in [
        testproblems.cifar10_3c3d,
        testproblems.fmnist_2c2d,
        testproblems.cifar10_resnet32,
    ]:
        return 3
    elif problem_cls in [testproblems.cifar100_allcnnc]:
        return 1
    else:
        raise NotImplementedError(f"Problem class {problem_cls} not supported.")


def get_fewer_checkpoints(config):
    """Choose grid of checkpoints based on ``config``."""

    problem_cls = config["problem_cls"]
    checkpoints = config["checkpoints"]
    if problem_cls in [testproblems.cifar10_3c3d, testproblems.fmnist_2c2d]:
        return checkpoints
    elif problem_cls in [testproblems.cifar10_resnet32]:
        return checkpoints[::2] + [checkpoints[-1]]
    elif problem_cls in [testproblems.cifar100_allcnnc]:
        return checkpoints[::4] + [checkpoints[-1]]
    else:
        raise NotImplementedError(f"Problem class {problem_cls} not supported.")


# ======================================================================================
# Computation of plotting data
# ======================================================================================
def eval_eigspace(case, model, loss_function, batch_data, top_C):
    """Evaluate the GGN's eigenspace either using ViViT or using the power iteration
    implemented by ``GGNLinearOperator``.
    """

    # Check inputs
    _, labels = batch_data
    if not len(labels) == case["batch_size"]:
        raise ValueError(
            "Provided data does not match the batchsize specified by case."
        )

    method = case["method"]
    if method == "ViViT":
        evals, evecs = eval_eigspace_vivit(
            case,
            model,
            loss_function,
            batch_data,
            top_C,
            device=DEVICE,
            verbose=VERBOSE,
        )
    elif method == "PI":
        evals, evecs = eval_eigspace_pi(
            model,
            loss_function,
            batch_data,
            top_C,
            device=DEVICE,
            check_deterministic=CHECK_DETERMINISTIC_BATCH,
            verbose=VERBOSE,
        )
    else:
        raise ValueError(f"Unknown computing method {method}")
    return evals, evecs


def eval_config(config, cases, json_path, nof_reps):
    """For the given configuration, evaluate all cases given by ``cases``. Store
    the results at ``json_path``. ``results`` is basically a copy of ``cases``, where
    each case has additionals keys for the results.
    """

    if VERBOSE:
        print("\nWorking on config = \n", config)

    problem_cls = config["problem_cls"]
    optimizer_cls = config["optimizer_cls"]
    num_classes = config["num_classes"]

    # Get deterministic training set data loader for full-batch GGN
    torch.manual_seed(0)
    training_dataloader = get_deterministic_deepobs_train_loader(
        problem_cls, BATCH_SIZE
    )

    # Get data with batch size ``BATCH_SIZE`` for approximations (NOTE: all cases use
    # this batch size)
    torch.manual_seed(0)
    batch_data_list = list(get_deepobs_dataloader(problem_cls, BATCH_SIZE))[0:nof_reps]

    results = []
    for case in cases:
        eigvals_vivit = torch.zeros(nof_reps, len(config["checkpoints"]), num_classes)
        eigvals_fb = torch.zeros_like(eigvals_vivit)

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

            # Reference: Full-batch GGN (evaluated on the entire training set)
            fb_ggn = GGNLinearOperator(
                model,
                loss_function,
                training_dataloader,
                DEVICE,
                progressbar=False,
                check_deterministic=CHECK_DETERMINISTIC_FB,
            )

            for batch_idx in range(nof_reps):
                # Evaluate the eigenvalues and -vectors on a new batch
                evals_approx, evecs_approx = eval_eigspace(
                    case, model, loss_function, batch_data_list[batch_idx], num_classes
                )
                eigvals_vivit[batch_idx, checkpoint_idx, :] = evals_approx

                # Evaluate the directional derivatives on the entire data set
                eigvals_fb[batch_idx, checkpoint_idx, :] = directional_derivatives(
                    fb_ggn, evecs_approx, DEVICE
                )

        # Store results in case dict
        case["eigvals_vivit"] = tensor_to_list(eigvals_vivit)
        case["eigvals_fb"] = tensor_to_list(eigvals_fb)
        results.append(case)

    dump_json(results, json_path)
    return results


# ======================================================================================
# Plotting
# ======================================================================================
def rel_error(vec_approx, vec_exact):
    """Compute the relative error for two arrays of same shape."""
    assert vec_approx.shape == vec_exact.shape, "Arrays must have the same shape"
    err = np.abs(vec_approx - vec_exact)
    return np.divide(err, vec_exact, where=(vec_exact != 0))


def av_rel_error(vec_approx, vec_exact):
    """Compute the average relative error for two arrays of same shape."""
    assert vec_approx.shape == vec_exact.shape, "Arrays must have the same shape"
    return np.mean(rel_error(vec_approx, vec_exact))


def plot_case(config, case, ax):
    color, marker = get_case_color_marker(case)
    label = get_case_label(case)  # HOTFIX: Update case label

    eigvals_vivit = np.array(case["eigvals_vivit"])
    eigvals_fb = np.array(case["eigvals_fb"])

    for batch_idx in range(eigvals_vivit.shape[0]):
        checkpoints = []
        av_rel_errors = []

        for checkpoint_idx in range(eigvals_vivit.shape[1]):
            evals_vivit = eigvals_vivit[batch_idx, checkpoint_idx, :]
            evals_fb = eigvals_fb[batch_idx, checkpoint_idx, :]

            checkpoints.append(config["decimal_checkpoints"][checkpoint_idx] + 1)
            av_rel_errors.append(av_rel_error(evals_vivit, evals_fb))

        # Plot average relative error
        ax.plot(
            checkpoints,
            av_rel_errors,
            marker,
            color=color,
            ms=1.0,
            label=label if batch_idx == 0 else None,
            alpha=0.6,
        )


def plot(config, plot_data, fig_path):
    """Plot all cases in one figure."""

    # tueplot settings and context
    icml_size = figsizes.icml2022(column="half")
    font_config = fonts.icml2022_tex()
    font_sizes = fontsizes.icml2022()
    with plt.rc_context({**icml_size, **font_config, **font_sizes}):
        fig, ax = plt.subplots()

        for case in plot_data:
            plot_case(config, case, ax)

        ax.set_xlabel("epoch (log scale)")
        ax.set_ylabel("av. rel. error (log scale)")
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

        ncol = 2
        if config["problem_cls"] in [testproblems.cifar100_allcnnc]:
            ncol = 1
        leg = ax.legend(ncol=ncol)
        leg.get_frame().set_linewidth(0.5)

        fig.savefig(fig_path)
        TikzExport().save_fig(
            fig_path.replace(".pdf", ""), png_preview=True, tex_preview=False
        )
        plt.close(fig)


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

    # Reduce number of checkpoints and number of repetitions
    nof_reps = get_nof_reps(config)
    fewer_checkpoints = get_fewer_checkpoints(config)
    config["checkpoints"] = fewer_checkpoints
    config = _add_decimal_checkpoints(config)  # update decimal checkpoints
    print(f"\nUsing nof_reps = {nof_reps} and {len(fewer_checkpoints)} checkpoints.")

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
        plot_data = eval_config(config, cases, json_path, nof_reps)
    else:
        print(f"Skipping computation. Using existing file {json_path}")
        plot_data = load_json(json_path)

    # Compute figure if necessary
    fig_path = get_plot_savepath(config_str + "_plot", extension=".pdf")
    if VERBOSE:
        print(f"\nChecking for figure at {fig_path}")

    if not os.path.exists(fig_path) or RECOMPUTE_FIG:
        print("Computing figure.")
        plot(config, plot_data, fig_path)
    else:
        print(f"Skipping computation. Using existing file {fig_path}")
