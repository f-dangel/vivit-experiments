"""This file implements utilities that are shared among all scripts."""

import argparse
import json
import warnings
from copy import deepcopy

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from deepobs.pytorch import datasets, testproblems
from eval import get_eval_savepath
from scipy.sparse.linalg import eigsh

from vivit.hessianfree import GGNLinearOperator
from vivit.linalg.eigh import EighComputation


def load_eval_result(
    problem_cls, optimizer_cls, checkpoint, file_name, extension=".pt"
):
    """Load a result of the evaluation named `file_name`."""
    savepath = get_eval_savepath(
        problem_cls, optimizer_cls, checkpoint, file_name, extension
    )
    try:
        return torch.load(savepath)
    except FileNotFoundError:
        print(f"File not found at {savepath}. Returning `None`.")
        return None


def subspaces_overlap(U, V, num_classes):
    """Compute the overlap of two eigenspaces of the same dimensionality.

    Mathematical background: We consider two spaces, spanned by a set of C orthonormal
    vectors S_U = span(u_1, ..., u_C) and S_V = span(v_1, ..., v_C). These spaces are
    represented by the (D x C)-matrices U = (u_1, ..., u_C) and V = (v_1, ..., v_C). The
    respective projection matrices are P_U = U @ U.T and P_V = V @ V.T.

    As in the paper "GRADIENT DESCENT HAPPENS IN A TINY SUBSPACE" (see
    https://arxiv.org/pdf/1812.04754.pdf), we define the overlap (a number in [0, 1])

    overlap(S_U, S_V) = tr(P_U @ P_V) / sqrt(tr(P_U) tr(P_V))

    The numerator and denominator can be computed efficiently by using the trace's
    cyclic property. It holds tr(P_U @ P_V) = tr(W.T @ W) with W := U.T @ V (note that
    this is a "small" C x C matrix). This is normalized by sqrt(tr(P_U) tr(P_V)) = C.
    """

    # Check that U, V have the correct shape
    assert U.shape == V.shape, "U and V don't have the same shape"
    _, C = U.shape
    assert C == num_classes, "U doesn't have `num_classes` columns"

    W = U.T @ V
    overlap = torch.trace(W.T @ W) / num_classes
    return overlap.item()


def gradient_overlap(grad, V, num_classes):
    """Compute the overlap of the gradient and an eigenspace.

    Mathematical background: We consider a space, spanned by a set of orthonormal
    vectors S_V = span(v_1, ..., v_C). This spaces is represented by the (D x C)-matrix
    V = (v_1, ..., v_C). The respective projection matrix are P_V = V @ V.T.

    As in the paper "GRADIENT DESCENT HAPPENS IN A TINY SUBSPACE" (see
    https://arxiv.org/pdf/1812.04754.pdf), we define the overlap (a number in [0, 1])

    overlap(grad, S_V) = ||P_V grad||^2 / ||grad||^2
    """

    # Check that grad, V have the correct shape
    D, C = V.shape
    assert C == num_classes, "V doesn't have `num_classes` columns"
    assert grad.numel() == D, f"grad does not have {D} entries"

    proj_grad = torch.matmul(V, torch.matmul(V.T, grad))
    overlap = torch.dot(proj_grad, proj_grad) / torch.dot(grad, grad)
    return overlap.item()


def gradient_detailed_overlap(grad, V, num_classes):
    """Similar to `gradient_overlap`, but here, we compute the overlap of the gradient
    with the individual eigenvectors. The gradiejt overlap can be written as

    overlap(grad, S_V) = ||P_V grad||^2 / ||grad||^2
                       = (sum_{c=1}^C (v_c.T grad)^2) / (grad.T grad)

    Here, we consider the individual projections (v_c.T grad)^2.
    """

    # Check that grad, V have the correct shape
    D, C = V.shape
    assert C == num_classes, "V doesn't have `num_classes` columns"
    assert grad.numel() == D, f"grad does not have {D} entries"

    proj_coeffs = torch.square(torch.matmul(V.T, grad))
    return proj_coeffs / (grad.T @ grad)


def get_config_str_parser():
    """Create a parser for the command line argument `config_str`"""
    parser_description = "Parser for the command line argument `config_str`."
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument(
        "--config_str",
        dest="config_str",
        action="store",
        type=str,
        help="The configuration as a string",
    )

    return parser


def tensor_to_list(T):
    """Convert a ``torch.Tensor`` into a list (to be able to dump it into json)"""
    return T.detach().cpu().numpy().tolist()


def dump_json(data, file_path):
    """Write `data` to a ``.json`` file. NOTE: The file gets overwritten if it already
    exists."""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_json(file_path):
    """Load data from json file."""
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def evec_list_to_mat(evecs):
    """``EighComputation`` returns a list of tensors with shapes
    ``[[E, *p1.shape], [E, *p2.shape], ...]``, where ``E`` is the number of
    eigenvectors. We convert this into a 2D-tensor with
    ``p1.numel() + p2.numel() + ...`` rows and ``E`` columns.
    """

    evecs_flat = torch.cat([e.flatten(start_dim=1) for e in evecs], dim=1)
    evecs_flat = torch.flip(evecs_flat.T, dims=(1,))
    return evecs_flat


def eval_eigspace_vivit(
    case, model, loss_function, batch_data, top_C, device, verbose=True
):
    """Evaluate the eigenspace for the given case ``case`` using ViViT. ``case`` needs
    to provide at least the keys ``"subsampling"`` and ``"mc_samples"``. Return
    eigenvalues and eigenvectors.
    """

    if verbose:
        print("Evaluating eigenspace with ViViT")

    def criterion(evals):
        """Filter criterion for eigenvalues. Only keep the largest eigenvalues."""

        if len(evals) < top_C:
            warn_msg = f"criterion: num_classes exceeds len(evals) = {len(evals)}. "
            warn_msg += f"Using top-{len(evals)} instead of top-{top_C}."
            warnings.warn(warn_msg)
            return list(range(len(evals)))

        _, indices = torch.topk(evals, top_C)
        return indices.tolist()

    # Copy and extend model
    model = extend(deepcopy(model).eval().to(device))
    loss_function = extend(deepcopy(loss_function).to(device))

    # Form one parameter group with trainable parameters
    parameters = [p for p in model.parameters() if p.requires_grad]
    group = {"params": parameters, "criterion": criterion}
    param_groups = [group]

    # Evaluate the eigenspace on the given batch
    inputs, labels = batch_data
    loss = loss_function(model(inputs.to(device)), labels.to(device))

    computation = EighComputation(
        subsampling=case["subsampling"],
        mc_samples=case["mc_samples"],
        verbose=False,
    )

    torch.manual_seed(0)  # In case MC sampling is used
    with backpack(
        computation.get_extension(),
        extension_hook=computation.get_extension_hook(param_groups),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    evals = computation._evals[id(group)]
    evecs = computation._evecs[id(group)]
    return torch.flip(evals, dims=(0,)), evec_list_to_mat(evecs).to(device)


def eval_eigspace_pi(
    model,
    loss_function,
    batch_data,
    top_C,
    device,
    check_deterministic=True,
    verbose=True,
):
    """Evaluate the eigenspace using the power iteration method implemented by
    ``GGNLinearOperator``. Return eigenvalues and eigenvectors.
    """

    if verbose:
        print("Evaluating eigenspace with ``GGNLinearOperator``'s power iteration")

    # Copy model
    model = deepcopy(model).eval().to(device)
    loss_function = deepcopy(loss_function).to(device)

    GGN_linop = GGNLinearOperator(
        model,
        loss_function,
        [batch_data],
        device,
        progressbar=False,
        check_deterministic=check_deterministic,
    )
    evals, evecs = eigsh(GGN_linop, k=top_C, which="LM")
    return torch.from_numpy(evals).to(device), torch.from_numpy(evecs).to(device)


def check_cases(cases):
    """Make sure that cases with method ``PI`` don't use curvature subsampling or MC
    samples (that is only supportet when using ``ViViT``).
    """
    for case in cases:
        # Check that computing method is known
        if not case["method"] in ["PI", "ViViT"]:
            raise ValueError("Unknown computing method in ``cases``")

        # Check that ``PI`` is not used with MC-sampling or curvature subsampling
        if case["method"] == "PI":
            assert (
                case["subsampling"] is None
            ), "Computing method ``PI`` doesn't support curvature subsampling"
            assert (
                case["mc_samples"] == 0
            ), "Computing method ``PI`` doesn't support mc_samples"


def get_case_label(case):
    """Determine label for case based on ``"batch_size"``, ``"subsampling"`` and
    ``"mc_samples"``
    """
    bs = case["batch_size"]
    sub = case["subsampling"]
    mc = case["mc_samples"]

    # Distinguish between mini-batch ``mb`` and curvature sub-sampling ``sub`` label
    if sub is None:
        label = f"mb {bs}, "
    else:
        label = f"sub {len(sub)}, "

    # Add label ``mc`` if Monte Carlo estimator is used, else ``exact``
    if mc != 0:
        label += f"mc {mc}"
    else:
        label += "exact"

    return label


def get_deepobs_dataloader(problem_cls, batch_size):
    """Get deterministic training dataloader of DeepOBS problems (but do not alter the
    batch size).
    """

    # Determine dataset class
    if problem_cls in [testproblems.cifar10_3c3d, testproblems.cifar10_resnet32]:
        dataset_cls = datasets.cifar10
    elif problem_cls == testproblems.cifar100_allcnnc:
        dataset_cls = datasets.cifar100
    elif problem_cls == testproblems.fmnist_2c2d:
        dataset_cls = datasets.fmnist
    else:
        raise NotImplementedError

    # Turn off data augmentation and return train loader
    try:
        dataset = dataset_cls(batch_size, data_augmentation=False)
    except TypeError:  # No data_augmentation avaiable (e.g. fmnist)
        dataset = dataset_cls(batch_size)

    torch.manual_seed(0)
    train_loader, _ = dataset._make_train_and_valid_dataloader()
    return train_loader


def directional_derivatives(lin_op, directions, device):
    """``lin_op`` represents a curvature matrix (either ``GGNLinearOperator`` or
    ``HessianLinearOperator`` from ``vivit.hessianfree``). ``directions`` is a
    ``D x nof_directions`` matrix, where ``nof_directions`` directions are stored
    column-wise. For every direction ``d``, we compute ``d.T @ lin_op_matrix @ d``.
    """

    nof_directions = directions.shape[1]

    derivs = torch.zeros(nof_directions)
    for d_idx in range(nof_directions):
        # Pick direction and convert to numpy
        d = directions[:, d_idx]
        d_np = torch.clone(d).cpu().numpy()

        # Compute directional derivative
        mat_vec_product = torch.from_numpy(lin_op.matvec(d_np)).to(device)
        derivs[d_idx] = torch.inner(mat_vec_product, d).item()

    return derivs


def eval_gammas_lambdas_pi(
    model,
    loss_function,
    batch_data,
    top_C,
    device,
    check_deterministic=True,
    verbose=True,
):
    """Evaluate the directional derivatives using the power iteration (``pi``). Return
    the  gammas ``𝛾[n, d]`` and lambdas ``λ[n, d]``.
    """

    if verbose:
        print(
            "Evaluating 𝛾[n, d], λ[n, d] with ``GGNLinearOperator``'s power iteration"
        )

    # Copy model
    model = deepcopy(model).eval().to(device)
    loss_function = deepcopy(loss_function).to(device)

    # Compute eigenvectors on mini-batch
    GGN_linop = GGNLinearOperator(
        model,
        loss_function,
        [batch_data],
        device,
        progressbar=False,
        check_deterministic=check_deterministic,
    )
    _, evecs = eigsh(GGN_linop, k=top_C, which="LM")
    evecs = torch.from_numpy(evecs).to(device)

    # Determine batch size (``N``)
    inputs, labels = batch_data
    N = len(labels)

    gamma_nk = torch.zeros(N, top_C)
    lambda_nk = torch.zeros(N, top_C)
    for n_idx in range(N):
        # Compute gradient and GGN on one sample
        sample = (
            torch.unsqueeze(inputs[n_idx], dim=0),
            torch.unsqueeze(labels[n_idx], dim=0),
        )
        sample_GGN_linop = GGNLinearOperator(
            model,
            loss_function,
            [sample],
            device,
            progressbar=False,
            check_deterministic=False,  # Just one sample --> there is no "order"
        )
        sample_grad, _ = sample_GGN_linop.gradient_and_loss()
        sample_grad = torch.nn.utils.parameters_to_vector(sample_grad).to(device)

        # Compute projections onto eigenvectors
        lambda_nk[n_idx, :] = directional_derivatives(sample_GGN_linop, evecs, device)
        for k_idx in range(top_C):
            gamma_nk[n_idx, k_idx] = torch.inner(sample_grad, evecs[:, k_idx])

    return gamma_nk.to(device), lambda_nk.to(device)
