"""Utility functions for examples."""

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.convert_parameters import vector_to_parameter_list
from torch.nn.utils.convert_parameters import parameters_to_vector


def classification_targets(size, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return torch.randint(size=size, low=0, high=num_classes)


def check_symeig(mat, evals, evecs, atol=1e-7, rtol=1e-5):
    """Check that EVecs are scaled with EVals by matrix multiplication.

    Args:
        mat (torch.Tensor): The matrix whose ``symeig`` is to be checked. To be
            more precise, any symmetric operator which implements ``matmul`` with
            a vector can be passed.
        evals (torch.Tensor): Eigenvalues of ``mat``.
        evecs (torch.Tensor): Eigenvectors of ``mat`` associated with ``evals``.
        atol (float): Absolute tolerance to detect wrong scaling of an EVec.
        rtol (float): Relative tolerance to detect wrong scaling of an EVec.

    Raises:
        ValueError: If the EVecs are not properly scaled up to numerical tolerance
            described by ``atol`` and ``rtol``.
    """
    for idx, ev in enumerate(evals):
        evec = evecs[:, idx]

        mat_prod = mat.matmul(evec)
        scaled = evec * ev

        for mp, scale in zip(mat_prod, scaled):
            if not torch.allclose(mp, scale, atol=atol, rtol=rtol):
                raise ValueError(f"(mat prod) {mp:5f} ≠ {scale:5f} (EVal)")


def autograd_gradient_covariance(
    X, y, model, loss_function, param_groups, center=False
):
    """Compute the gradient covariance matrix via autodiff.

    Args:
        center (bool): Whether to compute the centered gradient covariance matrix.
            If set to ``False``, the uncentered covariance (second gradient moments)
            is computed.
    """
    N_axis = 0
    N = X.shape[N_axis]

    individual_gradients = autograd_individual_gradients(
        X, y, model, loss_function, param_groups
    )

    if center:
        individual_gradients = [
            igrad - igrad.mean(N_axis) for igrad in individual_gradients
        ]

    cov_mats = [
        1 / N * torch.einsum("ni,nj->ij", igrad, igrad)
        for igrad in individual_gradients
    ]

    return cov_mats


def autograd_individual_gradients(X, y, model, loss_function, param_groups):
    """Compute individual gradients via autodiff.

    Assume mean reduction in loss_function.
    """
    N = X.shape[0]
    D = sum(p.numel() for p in model.parameters())

    individual_gradients = torch.zeros(N, D)

    for n in range(N):
        outputs = model(X[n].unsqueeze(0))
        loss = loss_function(outputs, y[n].unsqueeze(0))

        grad = torch.autograd.grad(loss, model.parameters())
        individual_gradients[n, :] = torch.cat([g.flatten() for g in grad])

    indices = parameter_groups_to_idx(param_groups, list(model.parameters()))

    return [individual_gradients[:, idx] for idx in indices]


def autograd_generalized_gauss_newton(X, y, model, loss_function, param_groups):
    """Compute the generalized Gauss-Newton matrix via autodiff."""
    D = sum(p.numel() for p in model.parameters())

    outputs = model(X)
    loss = loss_function(outputs, y)

    ggn_mat = torch.zeros(D, D)

    # compute GGN columns by GGNVPs with one-hot vectors
    for d in range(D):
        e_d = torch.zeros(D)
        e_d[d] = 1.0
        e_d_list = vector_to_parameter_list(e_d, model.parameters())

        ggn_d_list = ggn_vector_product(loss, outputs, model, e_d_list)

        ggn_mat[d, :] = parameters_to_vector(ggn_d_list)

    indices = parameter_groups_to_idx(param_groups, list(model.parameters()))

    return [ggn_mat[idx, :][:, idx] for idx in indices]


def parameter_groups_to_idx(param_groups, parameters):
    """Return indices for parameter groups in parameters."""
    params_in_group_ids = [id(p) for group in param_groups for p in group["params"]]
    params_ids = [id(p) for p in parameters]

    if len(params_in_group_ids) != len(set(params_in_group_ids)):
        raise ValueError("Same parameters occur in different groups.")
    if sorted(params_in_group_ids) != sorted(params_ids):
        raise ValueError("Parameters and group parameters don't match.")

    num_params = [param.numel() for param in parameters]
    param_ids = [id(p) for p in parameters]
    indices = []

    for group in param_groups:
        param_indices = []

        for param in group["params"]:
            param_idx = param_ids.index(id(param))

            start = sum(num_params[:param_idx])
            end = sum(num_params[: param_idx + 1])
            param_indices += list(range(start, end))

        indices.append(param_indices)

    return indices


def autograd_generalized_gauss_newton_eigs(
    X, y, model, loss_function, param_groups, criterion=None
):
    """Compute the filtered spectrum of the generalized Gauss-Newton."""
    group_ggn = autograd_generalized_gauss_newton(
        X, y, model, loss_function, param_groups
    )
    group_evals = []
    group_evecs = []

    for ggn in group_ggn:
        evals, evecs = ggn.symeig(eigenvectors=True)

        if criterion is not None:
            keep = criterion(evals)
            evals = evals[keep]
            evecs = evecs[:, keep]

        group_evals.append(evals)
        group_evecs.append(evecs)

    return group_ggn, group_evals, group_evecs


def autograd_gammas(
    X, y, model, loss_function, criterion, param_groups, ggn_evecs=None
):
    """Compute first-order directional derivatives ``γ[n,k]`` via autograd."""
    if ggn_evecs is None:
        _, _, ggn_evecs = autograd_generalized_gauss_newton_eigs(
            X, y, model, loss_function, param_groups, criterion=criterion
        )
    individual_gradients = autograd_individual_gradients(
        X, y, model, loss_function, param_groups
    )

    gammas = []

    for evecs, igrad in zip(ggn_evecs, individual_gradients):
        gammas.append(torch.einsum("ni,ik->nk", igrad, evecs))

    return gammas


def autograd_lambdas(
    X, y, model, loss_function, criterion, param_groups, ggn_evecs=None
):
    """Compute second-order directional derivatives ``λ[n,k]`` via autograd."""
    if ggn_evecs is None:
        _, _, ggn_evecs = autograd_generalized_gauss_newton_eigs(
            X, y, model, loss_function, param_groups, criterion=criterion
        )

    lambdas = [[] for _ in param_groups]

    batch_axis = 0
    split_size = 1

    for X_n, y_n in zip(
        X.split(split_size, dim=batch_axis), y.split(split_size, dim=batch_axis)
    ):
        ggn_n = autograd_generalized_gauss_newton(
            X_n, y_n, model, loss_function, param_groups
        )

        for group_idx, (group_ggn_n, group_ggn_evecs) in enumerate(
            zip(ggn_n, ggn_evecs)
        ):
            lambdas[group_idx].append(
                torch.einsum(
                    "ik,ij,jk->k", group_ggn_evecs, group_ggn_n, group_ggn_evecs
                )
            )

    return [torch.stack(group_lambdas) for group_lambdas in lambdas]


def autograd_directional_derivatives(
    X, y, model, loss_function, criterion, param_groups
):
    """Compute first- & second-order directional derivatives and GGN directions."""
    _, evals, ggn_evecs = autograd_generalized_gauss_newton_eigs(
        X, y, model, loss_function, param_groups, criterion=criterion
    )
    gammas = autograd_gammas(
        X, y, model, loss_function, criterion, param_groups, ggn_evecs=ggn_evecs
    )
    lambdas = autograd_lambdas(
        X, y, model, loss_function, criterion, param_groups, ggn_evecs=ggn_evecs
    )

    return evals, gammas, lambdas


def autograd_damped_newton_step(
    X, y, model, loss_function, criterion, param_groups, damping
):
    _, _, group_ggn_evecs = autograd_generalized_gauss_newton_eigs(
        X, y, model, loss_function, param_groups, criterion=criterion
    )
    group_gammas = autograd_gammas(
        X, y, model, loss_function, criterion, param_groups, ggn_evecs=group_ggn_evecs
    )
    group_lambdas = autograd_lambdas(
        X, y, model, loss_function, criterion, param_groups, ggn_evecs=group_ggn_evecs
    )

    group_newton_steps = []
    batch_axis = 0

    for group, gammas, lambdas, ggn_evecs in zip(
        param_groups, group_gammas, group_lambdas, group_ggn_evecs
    ):
        deltas = damping(gammas, lambdas)
        scale = -gammas.mean(batch_axis) / (lambdas.mean(batch_axis) + deltas)

        step = torch.einsum("ik,k->i", ggn_evecs, scale)

        group_newton_steps.append(vector_to_parameter_list(step, group["params"]))

    return group_newton_steps


def human_readable(captured, param_groups, model, loss_function):
    """Convert verbose output into human-readable format by replacing object ids."""
    replace = []

    for idx, group in enumerate(param_groups):
        replace.append([id(group), idx])
    for name, param in model.named_parameters():
        replace.append([id(param), name])
    for idx, module in enumerate(list(model.modules()) + [loss_function]):
        replace.append([id(module), idx])

    for old, new in replace:
        captured = captured.replace(str(old), str(new))

    return captured
