"""This script performs the evaluation for the configuration and checkpoint specified 
by the two command line arguments `config_str` and `checkpoint`. An example call looks
like this: `python ./eval.py --config_str cifar10_resnet32_sgd --checkpoint 0 0`.

The evaluation includes the following computations:
1) full-batch gradient
2) top-C eigenvectors of the full-batch GGN 
3) top-C eigenvectors of the full-batch Hessian 

NOTE: All computed quantities for one particular configuration and one particular 
checkpoint are stored in the same folder determined by `get_eval_savedir`. 
"""

import argparse
import os
from warnings import warn

import dill
import torch
from config import CHECKPOINTS_OUTPUT, config_str_to_config
from run_eval import get_eval_savedir
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, TensorDataset

from exp.utils.deepobs import get_deterministic_deepobs_train_loader
from exp.utils.deepobs_runner import CheckpointRunner
from vivit.hessianfree import GGNLinearOperator, HessianLinearOperator

USE_ONE_BATCH = False  # For debugging: Dataloader only contains one batch
VERBOSE = True
CHECK_DETERMINISTIC = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# I/O: Utilities for loading checkpoints and saving results of the evaluation
# ======================================================================================
def load_checkpoint(problem_cls, optimizer_cls, checkpoint):
    """Load checkpointed model and loss function. Returns `None` if checkpoint is not
    found.
    """
    savepath = CheckpointRunner.get_checkpoint_savepath(
        checkpoint, optimizer_cls, problem_cls, CHECKPOINTS_OUTPUT, extension=".pt"
    )

    print(f"Loading checkpoint from {savepath}")
    try:
        return torch.load(savepath, pickle_module=dill)
    except OSError:
        warn(f"Checkpoint {checkpoint} not found at {savepath}. Returning `None`.")
        return None


def get_eval_savepath(
    problem_cls, optimizer_cls, checkpoint, file_name, extension=".pt"
):
    """Get savepath for some result of the evaluation named `file_name`."""
    savedir = get_eval_savedir(problem_cls, optimizer_cls, checkpoint)
    return os.path.join(savedir, f"{file_name}{extension}")


def numpy_to_torch32(numpy_array):
    """Convert numpy array into torch float32 tensor"""
    return (torch.from_numpy(numpy_array)).to(torch.float32)


# ======================================================================================
# Evaluation
# ======================================================================================
def eval_Hessian(model, loss_func, dataloader, num_classes, savepath):
    """Evaluate and store top-C eigenspace of the Hessian"""
    if VERBOSE:
        print(f"\neval_Hessian: Storing results at {savepath}")

    if os.path.exists(savepath):
        print(f"File {savepath} already exists. Skipping computation.")
    else:
        H = HessianLinearOperator(
            model,
            loss_func,
            dataloader,
            DEV,
            progressbar=False,
            check_deterministic=CHECK_DETERMINISTIC,
        )
        H_evals, H_evecs = eigsh(H, k=num_classes, which="LM")
        H_results = {
            "H_evals": numpy_to_torch32(H_evals),
            "H_evecs": numpy_to_torch32(H_evecs),
        }
        torch.save(H_results, savepath)


def eval_GGN(model, loss_func, dataloader, num_classes, savepath):
    """Evaluate and store top-C eigenspace of the GGN"""
    if VERBOSE:
        print(f"\neval_GGN: Storing results at {savepath}")

    if os.path.exists(savepath):
        print(f"File {savepath} already exists. Skipping computation.")
    else:
        G = GGNLinearOperator(
            model,
            loss_func,
            dataloader,
            DEV,
            progressbar=False,
            check_deterministic=CHECK_DETERMINISTIC,
        )
        G_evals, G_evecs = eigsh(G, k=num_classes, which="LM")
        G_results = {
            "G_evals": numpy_to_torch32(G_evals),
            "G_evecs": numpy_to_torch32(G_evecs),
        }
        torch.save(G_results, savepath)


def eval_gradient(model, loss_func, dataloader, savepath):
    """Evaluate and store gradient"""
    if VERBOSE:
        print(f"\neval_gradient: Storing results at {savepath}")

    if os.path.exists(savepath):
        print(f"File {savepath} already exists. Skipping computation.")
    else:
        H = HessianLinearOperator(
            model,
            loss_func,
            dataloader,
            DEV,
            progressbar=False,
            check_deterministic=CHECK_DETERMINISTIC,
        )
        grad, _ = H.gradient_and_loss()
        grad = (torch.nn.utils.parameters_to_vector(grad)).to(torch.float32)
        torch.save(grad, savepath)


def eval_checkpoint(config, checkpoint):
    """Perform all computations"""
    if VERBOSE:
        print("\n===== eval_checkpoint =====")
        print("\nconfig = \n", config)
        print("\ncheckpoint = ", checkpoint)
        print("\nDEV = ", DEV, "\n")

    problem_cls = config["problem_cls"]
    optimizer_cls = config["optimizer_cls"]
    eval_batchsize = config["batch_size"]
    num_classes = config["num_classes"]

    # Load checkpoint data, skip evaluation if checkpoint is not found
    checkpoint_data = load_checkpoint(problem_cls, optimizer_cls, checkpoint)
    if checkpoint_data is None:
        warn("No checkpoint data was found. Skipping computations.")
        return

    # Preparations
    model = checkpoint_data.pop("model")
    model.eval()  # Required by ResNet (BN layers)
    loss_func = checkpoint_data.pop("loss_func")

    torch.manual_seed(0)  # deterministic split into training/validation set
    dataloader = get_deterministic_deepobs_train_loader(problem_cls, eval_batchsize)

    # For debugging: Modify the dataloader such that it contains only the first batch
    if USE_ONE_BATCH:
        print("Warning: `eval_checkpoint` uses only the first batch of the dataloader")
        input, labels = next(iter(dataloader))
        dataset = TensorDataset(input, labels)
        dataloader = DataLoader(dataset, batch_size=len(labels))

    if not CHECK_DETERMINISTIC:
        warn("Deterministic behaviour of linear operators is not checked")

    # Hessian
    file_name = "topC_fb_Hessian"
    savepath = get_eval_savepath(problem_cls, optimizer_cls, checkpoint, file_name)
    eval_Hessian(model, loss_func, dataloader, num_classes, savepath)

    # GGN
    file_name = "topC_fb_GGN"
    savepath = get_eval_savepath(problem_cls, optimizer_cls, checkpoint, file_name)
    eval_GGN(model, loss_func, dataloader, num_classes, savepath)

    # Gradient
    file_name = "fb_gradient"
    savepath = get_eval_savepath(problem_cls, optimizer_cls, checkpoint, file_name)
    eval_gradient(model, loss_func, dataloader, savepath)


# ======================================================================================
# Parser for the two command line arguments `config_str` and `checkpoint`
# ======================================================================================
def get_args_parser():
    """Create a parser for the command line arguments"""
    parser_description = "Run optimizer on SLURM cluster."
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument(
        "--config_str",
        dest="config_str",
        action="store",
        type=str,
        help="The configuration as a string",
    )

    parser.add_argument(
        "--checkpoint",
        nargs=2,
        dest="checkpoint",
        action="store",
        type=int,
        help="The checkpoint (a tuple)",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    config = config_str_to_config(args.config_str)
    checkpoint = tuple(args.checkpoint)

    # Evaluate checkpoint
    eval_checkpoint(config, checkpoint)
