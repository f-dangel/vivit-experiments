"""Explain how to compute the GGN (exact version)spectrum (EVals, EVecs).

To use an MC-sampled approximation over the exact GGN, you have to ...

- ... call ``loss.backward`` in a BackPACK context with the ``SqrtGGNMC`` extension
  rather than the ``SqrtGGNExact``.
- ... modify the ``savefield`` where the active BackPACK extension stores its output
  from ``sqrt_ggn_exact`` to ``sqrt_ggn_mc``.

Note that in this case the comparisons with spectral properties of the exact GGN,
computed via autodiff, that are performed in this script will fail.
"""

import torch
from backpack import backpack, extend, extensions
from problem import make_data, make_loss_function, make_model
from utils import autograd_generalized_gauss_newton, check_symeig

from vivit.utils.eig import symeig
from vivit.utils.gram import compute_gram_mat, sqrt_gram_mat_prod

ATOL = 1e-6

# Make deterministic
torch.manual_seed(0)

# Dummy model and data
X, y = make_data()
model = extend(make_model())
loss_function = extend(make_loss_function())

# Compute the GGN square root decomposition ``V``
outputs = model(X)
loss = loss_function(outputs, y)

with backpack(extensions.SqrtGGNExact()):
    loss.backward()

# An ``[N * C, N * C]`` matrix with entries ``gram_mat[(n, m), (i, j)] = ⟨vₙₘ, vᵢⱼ⟩``
gram_mat = compute_gram_mat(model.parameters(), "sqrt_ggn_exact", start_dim=2)

# Compute Gram matrix EVals and EVecs
gram_evals, gram_evecs = symeig(gram_mat, eigenvectors=True, atol=ATOL)

# Confirm that the Gram matrix (EVal, EVec) pairs are correct
check_symeig(gram_mat, gram_evals, gram_evecs, atol=ATOL)

# Obtain GGN EVecs by multiplying the Gram matrix EVecs with ``V``
# For simplicity, we flatten the parameter dimensions and concatenate along layers
# Keep in mind though that the resulting EVecs need not be normalized.
ggn_evecs = sqrt_gram_mat_prod(
    gram_evecs, model.parameters(), "sqrt_ggn_exact", start_dim=2, concat=True
)

# Double check that EVecs are correct by computing the GGN matrix via autodiff
param_groups = [{"params": list(model.parameters())}]
autograd_ggn_mat = autograd_generalized_gauss_newton(
    X, y, model, loss_function, param_groups
)[0]

# Should have same eigenvalues as Gram matrix
autograd_ggn_evals, _ = symeig(autograd_ggn_mat, atol=ATOL)
print(f"EVals from BackPACK's Gram matrix: {gram_evals}")
print(f"EVals from autograd GGN matrix:    {autograd_ggn_evals}")

if not torch.allclose(autograd_ggn_evals, gram_evals, atol=ATOL):
    raise ValueError("EVals of autograd GGN and BackPACK Gram matrix don't match.")

# Note that in, in contrast to the gradient covariance, the eigenvalues are already
# correctly scaled for the GGN. Finally, check if EVecs are actual eigenvectors.
check_symeig(autograd_ggn_mat, gram_evals, ggn_evecs, atol=ATOL)
