"""Illustration of computing a damped Newton step, freeing up buffers earliest possible.

Parameter blocking breaks up the Newton step computation into multiple groups, whose
BackPACK buffers can be deleted at earlier stages during backpropagation
"""

import io
from contextlib import redirect_stdout
from typing import Any, Dict, List

import torch
from backpack import backpack, disable, extend
from torch import Tensor

from exp.examples.utils import autograd_damped_newton_step, human_readable
from vivit.optim import BaseComputations, ConstantDamping, DampedNewton

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


def run_example(param_groups, verbose=False):
    """Execute the example for a specific parameter blocking."""
    damping = ConstantDamping(0.1)

    # first, with autograd
    with disable():
        torch_group_newton_steps = autograd_damped_newton_step(
            X, y, model, loss_function, criterion, param_groups, damping
        )

    # now, with BackPACK
    output = model(X)
    loss = loss_function(output, y)

    computations = BaseComputations(verbose=verbose)
    opt = DampedNewton(param_groups, damping, computations, criterion)

    opt.zero_grad()
    opt.zero_newton()

    with backpack(
        *opt.get_extensions(),
        extension_hook=opt.get_extension_hook(
            keep_backpack_buffers=False,
            keep_gram_mat=False,
            keep_gram_evecs=False,
            keep_batch_size=False,
            keep_gammas=False,
            keep_lambdas=False,
            keep_gram_evals=False,
            keep_coefficients=False,
            keep_newton_step=True,
        ),
    ):
        loss.backward()
        opt.step()
        check_backpack_buffers_freed(model)

    compare_newton_steps(param_groups, computations, torch_group_newton_steps)


def run_example_closure(param_groups, verbose=False):
    """Execute the example for a specific parameter blocking. Use a closure."""
    damping = ConstantDamping(0.1)

    # first, with autograd
    with disable():
        torch_group_newton_steps = autograd_damped_newton_step(
            X, y, model, loss_function, criterion, param_groups, damping
        )

    # now, with BackPACK
    def closure() -> Tensor:
        """Evaluate the loss on a fixed mini-batch.

        Returns:
            Mini-batch loss.
        """
        return loss_function(model(X), y)

    computations = BaseComputations(verbose=verbose)
    opt = DampedNewton(param_groups, damping, computations, criterion)

    opt.step(
        closure=closure,
        keep_backpack_buffers=False,
        keep_gram_mat=False,
        keep_gram_evecs=False,
        keep_batch_size=False,
        keep_gammas=False,
        keep_lambdas=False,
        keep_gram_evals=False,
        keep_coefficients=False,
        keep_newton_step=True,
    )

    check_backpack_buffers_freed(model)
    compare_newton_steps(param_groups, computations, torch_group_newton_steps)


def compare_newton_steps(
    param_groups: Dict[int, Any],
    computations: BaseComputations,
    ground_truth: List[Tensor],
):
    """Compare Newton steps of ``BaseComputations`` and ``autograd``.

    Args:
        param_groups: Parameter groups of the optimizer.
        computations: Object used to compute the Newton steps with BackPACK.
        ground_truth: Newton steps computed via ``autograd``.

    Raises:
        AssertionError: If the Newton steps don't match.
    """
    rtol, atol = 5e-3, 5e-5

    print("Check Newton steps")

    for group_idx, group in enumerate(param_groups):
        print(f"Group {group_idx}")

        group_id = id(group)
        lib_newton_step = computations._newton_step[group_id]
        torch_newton_step = ground_truth[group_idx]

        for lib_step, torch_step in zip(lib_newton_step, torch_newton_step):
            if torch.allclose(lib_step, torch_step, rtol=rtol, atol=atol):
                print("\t✔ Damped Newton step of library and torch match.")
            else:
                raise AssertionError(
                    f"❌ Damped Newton steps don't match.\n{lib_step}\n{torch_step}"
                )


def check_backpack_buffers_freed(model: torch.nn.Module):
    """Check if buffers of BackPACK extensions are free.

    Args:
        model: Network that will be checked.

    Raises:
        AssertionError: If the buffers are not freed.
    """
    print("Check BackPACK buffers freed during backpropagation")
    for name, param in model.named_parameters():
        for savefield in ["grad_batch", "sqrt_ggn_exact"]:
            discarded = not hasattr(param, savefield)
            marker = "✔" if discarded else "❌"

            print(f"\t{marker} {name:11s} discarded '{savefield}'")

            if not discarded:
                raise AssertionError(f"Buffer '{savefield}' was not discarded")


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
        for closure in [False, True]:
            print(f"Running case '{case}', closure: {closure}")

            if VERBOSE:
                print(f"Net: {model}")

            error = None

            try:
                # capture output of print statements
                f = io.StringIO()
                with redirect_stdout(f):
                    if closure:
                        run_example_closure(param_groups, verbose=VERBOSE)
                    else:
                        run_example(param_groups, verbose=VERBOSE)

            except Exception as e:
                error = e

            finally:
                captured = f.getvalue()
                print(human_readable(captured, param_groups, model, loss_function))

                if error is not None:
                    raise error
