"""Illustration of the ``EigvalshComputation`` class.

It computes GGN eigenvalues via the Gram matrix during backpropagation.
"""

from backpack import backpack, extend
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import allclose, cuda, device, manual_seed, rand, stack
from torch.linalg import eigvalsh
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.linalg.eigvalsh import EigvalshComputation

manual_seed(0)

N = 4
D_out = 3
D_hidden = 5
D_in = 7
DEVICE = device("cuda" if cuda.is_available() else "cpu")

X = rand(N, D_in).to(DEVICE)
y = rand(N, D_out).to(DEVICE)

model = extend(
    Sequential(
        Linear(D_in, D_hidden),
        ReLU(),
        # nested sequential
        Sequential(Linear(D_hidden, D_hidden), ReLU()),
        Linear(D_hidden, D_out),
    ).to(DEVICE)
)

loss_function = extend(MSELoss(reduction="mean").to(DEVICE))

loss = loss_function(model(X), y)

computation = EigvalshComputation(verbose=True)
group = {"params": list(model.parameters())}
param_groups = [group]

with backpack(
    computation.get_extension(),
    extension_hook=computation.get_extension_hook(param_groups),
):
    loss.backward()

gram_evals = computation._evals[id(group)]

###############################################################################
#                           Comparison with autograd                          #
###############################################################################
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])
ggn_evals = eigvalsh(ggn)
ggn_evals = ggn_evals[-gram_evals.numel() :]

assert allclose(gram_evals, ggn_evals, rtol=1e-4, atol=5e-7), "GGN evals don't match"
print("GGN evals match")
