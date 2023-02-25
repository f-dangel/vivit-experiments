"""Illustration of directional derivative computation at reduced memory.

These can be computed in the Gram space, hence the associated BackPACK buffers can
be deleted immediately during backpropagation.
"""

import io
from contextlib import redirect_stdout

import torch
from backpack import backpack, disable, extend

from exp.examples.utils import autograd_directional_derivatives, human_readable
from vivit.optim import GramComputations

torch.manual_seed(0)

N = 4
D_out = 3
D_hidden = 5
D_in = 7

X = torch.rand(N, D_in)
y = torch.rand(N, D_out)

model = extend(
    torch.nn.Sequential(
        torch.nn.Linear(D_in, D_hidden),
        torch.nn.ReLU(),
        # nested sequential
        torch.nn.Sequential(
            torch.nn.Linear(D_hidden, D_hidden),
            torch.nn.ReLU(),
        ),
        torch.nn.Linear(D_hidden, D_out),
    )
)

loss_function = extend(torch.nn.MSELoss(reduction="mean"))


def criterion(evals, min=1e-5):
    """Filter out small eigenvalues. Keep at least the largest"""
    idx = [i for i, val in enumerate(evals) if val > min]

    if not idx:
        idx = [evals.numel() - 1]

    return idx


def run_example(param_groups, verbose=False):  # noqa: C901
    """Execute the example for a specific parameter blocking."""
    # first, with autograd
    with disable():
        (
            torch_group_evals,
            torch_group_gammas,
            torch_group_lambdas,
        ) = autograd_directional_derivatives(
            X, y, model, loss_function, criterion, param_groups
        )

    # now, with BackPACK
    output = model(X)
    loss = loss_function(output, y)

    computations = GramComputations(verbose=verbose)

    with backpack(
        *computations.get_extensions(param_groups),
        extension_hook=computations.get_extension_hook(
            param_groups,
            keep_backpack_buffers=False,
            keep_gram_mat=False,
            keep_gram_evecs=False,
            keep_batch_size=False,
            keep_gammas=True,
            keep_lambdas=True,
            keep_gram_evals=True,
        ),
    ):
        loss.backward()

        print("Check BackPACK buffers freed during backpropagation")
        for name, param in model.named_parameters():
            for savefield in ["grad_batch", "sqrt_ggn_exact"]:
                discarded = not hasattr(param, savefield)
                marker = "✔" if discarded else "❌"

                print(f"\t{marker} {name:11s} discarded '{savefield}'")

                if not discarded:
                    raise AssertionError(f"Buffer '{savefield}' was not discarded")

    rtol, atol = 5e-3, 5e-3

    print("Check directional derivatives")

    for group_idx, group in enumerate(param_groups):
        print(f"Group {group_idx}")

        group_id = id(group)
        lib_evals = computations._gram_evals[group_id]
        lib_gammas = computations._gammas[group_id]
        lib_lambdas = computations._lambdas[group_id]

        torch_evals = torch_group_evals[group_idx]
        torch_gammas = torch_group_gammas[group_idx]
        torch_lambdas = torch_group_lambdas[group_idx]

        try:
            if torch.allclose(torch_evals, lib_evals, rtol=rtol, atol=atol):
                print("\t✔ λ[k]        of library and torch match.")
            else:
                raise AssertionError("❌ λ[k] don't match.")

            if torch.allclose(torch_lambdas, lib_lambdas, rtol=rtol, atol=atol):
                print("\t✔ λ[n,k]      of library and torch match.")
            else:
                raise AssertionError("❌ λ[n,k] don't match.")

            if torch.allclose(
                torch_gammas.abs(), lib_gammas.abs(), rtol=rtol, atol=atol
            ):
                print("\t✔ abs(γ[n,k]) of library and torch match.")
            else:
                raise AssertionError("❌ abs(γ[n,k]) don't match:")

        except AssertionError as e:
            print(f"λ[k]:\n{torch_evals}\n{lib_evals}")
            print(f"λ[n,k]:\n{torch_lambdas}\n{lib_lambdas}")
            print(f"abs(γ[n,k]):\n{torch_gammas.abs()}\n{lib_gammas.abs()}")

            last_layer_bias_only = len(group["params"]) == 1 and id(
                group["params"][0]
            ) == id(list(model.parameters())[-1])

            if last_layer_bias_only:
                print(
                    "The last-layer bias GGN eigenspace is degenerate for MSELoss."
                    + " This can lead to permuted directions and failing comparisons."
                    + " Continuing without error as behavior is to be expected."
                )
            else:
                raise e


if __name__ == "__main__":
    cases = {}

    cases["full net"] = [
        {
            "params": list(model.parameters()),
            "criterion": criterion,
        }
    ]

    cases["layer-wise"] = [
        {"params": [p], "criterion": criterion} for p in model.parameters()
    ]

    selection = [2, 3]
    param_blocks = [
        [p for idx, p in enumerate(list(model.parameters())) if idx in selection],
        [p for idx, p in enumerate(list(model.parameters())) if idx not in selection],
    ]
    cases["selection"] = [
        {"params": p_list, "criterion": criterion} for p_list in param_blocks
    ]

    # print what is happening when
    VERBOSE = True

    for case, param_groups in cases.items():
        print(f"Running case '{case}'")

        if VERBOSE:
            print(f"Net: {model}")

        error = None

        try:
            # capture output of print statements
            f = io.StringIO()
            with redirect_stdout(f):
                run_example(param_groups, verbose=VERBOSE)

        except Exception as e:
            error = e

        finally:
            captured = f.getvalue()
            print(human_readable(captured, param_groups, model, loss_function))

            if error is not None:
                raise error
