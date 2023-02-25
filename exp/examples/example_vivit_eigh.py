"""Illustration of the ``EighComputation`` class.

It computed GGN eigenvalues & eigenvectors via the Gram matrix during backpropagation.
"""

from typing import List

from backpack import backpack, extend
from backpack.utils.convert_parameters import vector_to_parameter_list
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import Tensor, allclose, cuda, device, manual_seed, rand, stack
from torch.linalg import eigh
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.linalg.eigh import EighComputation

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

computation = EighComputation(verbose=True)


def keep_criterion(evals: Tensor) -> List[int]:
    """Filter criterion for eigenvalues. Only accept non-small values.

    Returns:
        Indices of eigenvalues to keep.
    """
    return [i for i in range(evals.numel()) if evals[i].abs() > 1e-4]


parameters = list(model.parameters())
group = {"params": parameters, "criterion": keep_criterion}
param_groups = [group]


with backpack(
    computation.get_extension(),
    extension_hook=computation.get_extension_hook(param_groups),
):
    loss.backward()

# eigenvalues and normalized eigenvectors can be accessed for each group via
evals = computation._evals[id(group)]  # shape [E] with E = (number of eigenvalues)
evecs = computation._evecs[id(group)]  # shapes [[E, *p1.shape], [E, *p2.shape], ...]

###############################################################################
#                           Comparison with autograd                          #
###############################################################################
ggn = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])
autograd_evals, autograd_evecs = eigh(ggn)

# apply filter criterion
keep = group["criterion"](autograd_evals)
autograd_evals, autograd_evecs = autograd_evals[keep], autograd_evecs[:, keep]

# convert eigenvectors from ``torch.eig`` into same shape format
# [D, E] with D = p1.numel() + p2.numel() + ...
# into [[E, *p1.shape], [E, *p2.shape], ...]
autograd_evecs = autograd_evecs.transpose(0, 1)
autograd_evecs = [vector_to_parameter_list(vec, parameters) for vec in autograd_evecs]
autograd_evecs = list(zip(*autograd_evecs))
autograd_evecs = [stack(evecs) for evecs in autograd_evecs]

# Compare eigenvalues
assert allclose(evals, autograd_evals, rtol=5e-5, atol=1e-6), "GGN evals don't match"
print("GGN evals match")

# Compare eigenvectors (absolute value because their direction is ambiguous)
abs_evecs = [evec.abs() for evec in evecs]
abs_autograd_evecs = [evec.abs() for evec in autograd_evecs]

for e1, e2 in zip(abs_evecs, abs_autograd_evecs):
    assert allclose(e1, e2, rtol=1e-4, atol=5e-5), "GGN evecs don't match"
print("GGN evecs match")
