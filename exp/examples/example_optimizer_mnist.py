"""Basic example illustrating how to use the damped Newton optimizer on MNIST."""

import io
import os
from contextlib import redirect_stdout

import torch
from backpack import backpack, extend
from backpack.utils.examples import get_mnist_dataloader

from exp.examples.utils import human_readable
from exp.utils.hotfix import download_MNIST_hotfix
from exp.utils.vivit import make_top_k_criterion
from vivit.optim import BaseComputations, ConstantDamping, DampedNewton

torch.manual_seed(0)

BATCH_SIZE = 64
MAX_STEPS = 20

# Hotfix: Manually download MNIST if it is not already there
HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
download_MNIST_hotfix(HEREDIR)

# Build MNIST classifier
mnist_loader = get_mnist_dataloader(batch_size=BATCH_SIZE)
model = extend(
    torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 10),
    )
)
loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))

# Create DampedNewton Optimizer
damping = ConstantDamping(10000)

# print what is happening when
VERBOSE = True
computations = BaseComputations(verbose=VERBOSE)


opt = DampedNewton(
    model.parameters(), damping, computations, criterion=make_top_k_criterion(2)
)


def train_loop():
    """Run the training loop."""
    global_step = 0

    for inputs, labels in iter(mnist_loader):
        opt.zero_grad()
        opt.zero_newton()

        # forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # backward pass
        with backpack(
            *opt.get_extensions(),
            extension_hook=opt.get_extension_hook(
                keep_gram_mat=False,
                keep_gram_evals=False,
                keep_gram_evecs=False,
                keep_gammas=False,
                keep_lambdas=False,
                keep_batch_size=False,
                keep_coefficients=False,
                keep_newton_step=False,
                keep_backpack_buffers=False,
            ),
        ):
            loss.backward()

        # optimizer step
        opt.step()
        global_step += 1

        print(f"Step: {global_step:5d} | Loss: {loss.item():.4f}")

        if global_step >= MAX_STEPS:
            break


error = None

try:
    # capture output of print statements
    f = io.StringIO()
    with redirect_stdout(f):
        train_loop()

except Exception as e:
    error = e

finally:
    captured = f.getvalue()
    print(human_readable(captured, opt.param_groups, model, loss_fn))

    if error is not None:
        raise error
