"""Explain how to compute the centered gradient covariance spectrum (EVals, EVecs)."""

import torch
from backpack import backpack, extend, extensions
from problem import make_data, make_loss_function, make_model
from utils import autograd_gradient_covariance, check_symeig

from vivit.extensions import hooks
from vivit.utils.eig import symeig
from vivit.utils.gram import compute_gram_mat, sqrt_gram_mat_prod

ATOL = 1e-6

# Make deterministic
torch.manual_seed(0)

# Dummy model and data
X, y = make_data()
model = extend(make_model())
loss_function = extend(make_loss_function())

# Compute the centered gradient covariance square root decomposition ``U``
outputs = model(X)
loss = loss_function(outputs, y)

# Because the loss uses reduction="mean", BackPACK computes (gₙ - ∑ᵢ₌₁ᴺ gᵢ)/N.
# Hence we will have to rescale the eigenvalues later on to compensate for that.
with backpack(extensions.BatchGrad(), extension_hook=hooks.CenteredBatchGrad()):
    loss.backward()

# An ``[N, N]`` matrix with entries
# ``gram_mat[i, j] = ⟨(gᵢ - ∑ₙ₌₁ᴺ gₙ)/N, (gⱼ - ∑ₙ₌₁ᴺ gₙ)/N⟩``
gram_mat = compute_gram_mat(model.parameters(), "centered_grad_batch", start_dim=1)

# Compute Gram matrix EVals and EVecs
gram_evals, gram_evecs = symeig(gram_mat, eigenvectors=True, atol=ATOL)

# Confirm that the Gram matrix (EVal, EVec) pairs are correct
check_symeig(gram_mat, gram_evals, gram_evecs, atol=ATOL)

# Obtain centered gradient covariance EVecs by multiplying Gram EVecs with ``U``
# For simplicity, we flatten the parameter dimensions and concatenate along layers
# Keep in mind though that the resulting EVecs need not be normalized.
cov_evecs = sqrt_gram_mat_prod(
    gram_evecs, model.parameters(), "centered_grad_batch", start_dim=1, concat=True
)

# Double check that EVecs are correct by computing the centered gradient covariance
# matrix via autodiff
param_groups = [{"params": list(model.parameters())}]
autograd_cov_mat = autograd_gradient_covariance(
    X, y, model, loss_function, param_groups, center=True
)[0]

# Should have (up to scaling) same eigenvalues as Gram matrix
autograd_cov_evals, _ = symeig(autograd_cov_mat, atol=ATOL)
print(f"EVals from BackPACK's Gram matrix:                {gram_evals}")
print(f"EVals from autograd centered gradient covariance: {autograd_cov_evals}")
print("⚠ Remember to rescale the EVals computed with BackPACK")

# We have to rescale the eigenvalues computed from BackPACK's Gram matrix.
# This matrix has entries ``⟨(gᵢ - ∑ₙ₌₁ᴺ gₙ)/N, (gⱼ - ∑ₙ₌₁ᴺ gₙ)/N⟩``, so we computed
# EVals of the matrix ``∑ₙ₌₁ᴺ ((gₙ - ∑ᵢ₌₁ᴺ gᵢ)/N) ((gₙ - ∑ᵢ₌₁ᴺ gᵢ)/N)ᵀ``, but the
# centered gradient covariance is ``1/N ∑ₙ₌₁ᴺ (gₙ - ∑ᵢ₌₁ᴺ gᵢ) (gₙ - ∑ᵢ₌₁ᴺ gᵢ)ᵀ``.
# Rescaling the EVals from the Gram matrix by ``N`` compensates this effect.
N = X.shape[0]
gram_evals_corrected = N * gram_evals

# Now the EVals should be the same. Also check that EVecs get scaled according to EVal.
if not torch.allclose(autograd_cov_evals, gram_evals_corrected, atol=ATOL):
    raise ValueError(
        "EVals of centered autograd covariance and BackPACK Gram matrix don't match."
    )

check_symeig(autograd_cov_mat, gram_evals_corrected, cov_evecs, atol=ATOL)
