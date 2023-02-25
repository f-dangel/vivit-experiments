"""This is a plotting script for one particular configuration specified via command line
argument. It can be called like this:
`python plot_eigspace_vivit_vs_mb.py --config_str fmnist_2c2d_sgd`
`run_plot_eigspace_vivit_vs_mb.py` automatically calls this script for all 
configurations. 

This script consists of 3 steps:
1) Parse the command line argument and map to actual configuration.
2) Compute and store the plotting data (if neccessary). We compute the overlap between
   the mini-batch (fb) eigenspace (as ground truth) and different approximations (e.g. 
   an MC approximation).
3) Compute and store the figure (if neccessary). This figure shows the overlap between
   the eigenspaces computed in 2).
"""

import os

import torch
from config import config_str_to_config
from deepobs.pytorch import testproblems
from eval import load_checkpoint
from plot_shared import plot_overlap_cases
from run_plot_eigspace_vivit_vs_mb import get_plot_savedir
from torch import cuda, device
from utils_shared import (
    check_cases,
    dump_json,
    eval_eigspace_pi,
    eval_eigspace_vivit,
    get_case_label,
    get_config_str_parser,
    get_deepobs_dataloader,
    load_json,
    subspaces_overlap,
    tensor_to_list,
)


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
# Define cases
# ======================================================================================
NOF_REPS = 5  # Number of samples/repetitions per case
BATCH_SIZE = 128
REF_CASE = {
    "batch_size": BATCH_SIZE,
    "subsampling": None,
    "mc_samples": 0,
    "method": "PI",
}
STD_CASES = [
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
        extra_cases = [
            {
                "batch_size": BATCH_SIZE,
                "subsampling": None,
                "mc_samples": 1,
                "method": "ViViT",
            },
            {
                "batch_size": BATCH_SIZE,
                "subsampling": list(range(16)),
                "mc_samples": 1,
                "method": "ViViT",
            },
        ]
    elif problem_cls in [testproblems.cifar100_allcnnc]:
        extra_cases = [
            {
                "batch_size": BATCH_SIZE,
                "subsampling": None,
                "mc_samples": 10,
                "method": "ViViT",
            },
            {
                "batch_size": BATCH_SIZE,
                "subsampling": list(range(16)),
                "mc_samples": 10,
                "method": "ViViT",
            },
        ]

    all_cases = STD_CASES + extra_cases

    # Add label and check all cases
    for case in all_cases:
        case["label"] = get_case_label(case)
    check_cases(all_cases)
    check_cases([REF_CASE])

    return all_cases


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
        _, evecs = eval_eigspace_vivit(
            case,
            model,
            loss_function,
            batch_data,
            top_C,
            device=DEVICE,
            verbose=VERBOSE,
        )
    elif method == "PI":
        _, evecs = eval_eigspace_pi(
            model,
            loss_function,
            batch_data,
            top_C,
            device=DEVICE,
            check_deterministic=CHECK_DETERMINISTIC,
            verbose=VERBOSE,
        )
    else:
        raise ValueError(f"Unknown computing method {method}")
    return evecs


def eval_config(config, cases, json_path):
    """For the given configuration, evaluate all cases given by ``cases``. Store
    the results at ``json_path``. ``results`` is basically a copy of ``cases``, where
    each case has the additional key ``overlaps``.
    """

    if VERBOSE:
        print("\nWorking on config = \n", config)

    # Get data with batch size ``BATCH_SIZE``
    problem_cls = config["problem_cls"]
    torch.manual_seed(0)
    data_list = list(get_deepobs_dataloader(problem_cls, BATCH_SIZE))[0:NOF_REPS]

    optimizer_cls = config["optimizer_cls"]
    num_classes = config["num_classes"]

    overlaps = torch.zeros(len(cases), NOF_REPS, len(config["checkpoints"]))
    for checkpoint_idx, checkpoint in enumerate(config["checkpoints"]):
        for batch_idx in range(NOF_REPS):
            # Reload checkpoint data (i.e. model and loss-function)
            checkpoint_data = load_checkpoint(problem_cls, optimizer_cls, checkpoint)
            if checkpoint_data is None:
                print("No checkpoint data was found. Skipping computations.")
                continue
            model = checkpoint_data.pop("model")
            loss_function = checkpoint_data.pop("loss_func")  # Must use ``mean``

            # Compute groud truth for this checkpoint
            if VERBOSE:
                print("\nWorking on reference case \n", REF_CASE)
            evecs_ref = eval_eigspace(
                REF_CASE, model, loss_function, data_list[batch_idx], num_classes
            )

            for case_idx, case in enumerate(cases):
                if VERBOSE:
                    print("\nWorking on case \n", case)

                # Evaluate the eigenspace on a new batch
                evecs_approx = eval_eigspace(
                    case, model, loss_function, data_list[batch_idx], num_classes
                )

                # Compute overlap
                overlap = subspaces_overlap(evecs_ref, evecs_approx, num_classes)
                if VERBOSE:
                    print_str = f"case_idx {case_idx} "
                    print_str += f"batch_idx {batch_idx} "
                    print_str += f"checkpoint_idx {checkpoint_idx}: "
                    print_str += f"overlap = {overlap}"
                    print(print_str)
                overlaps[case_idx, batch_idx, checkpoint_idx] = overlap

    # Re-organize data
    results = []
    for case_idx, case in enumerate(cases):
        case["overlaps"] = tensor_to_list(overlaps[case_idx, :, :])
        results.append(case)

    dump_json(results, json_path)
    return results


# ======================================================================================
# Plotting
# ======================================================================================
def plot(config, plot_data, fig_path):
    """Plotting for the first cases without ms sampling or curvature subsampling"""

    # Restrict cases
    cases = plot_data[0:4]
    plot_overlap_cases(config, cases, fig_path)


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
    fig_path = get_plot_savepath(config_str + "_plot", extension=".pdf")
    if VERBOSE:
        print(f"\nChecking for figure at {fig_path}")

    if not os.path.exists(fig_path) or RECOMPUTE_FIG:
        print("Computing figure.")
        plot(config, plot_data, fig_path)
    else:
        print(f"Skipping computation. Using existing file {fig_path}")
