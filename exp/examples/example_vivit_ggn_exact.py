"""Illustration of the ``ViViTGGNExact`` extension.

It provides access to the Gram matrix and multiplication by ``V`` & ``Vᵀ``.
"""

from backpack import backpack, extend
from backpack.hessianfree.ggnvp import ggn_vector_product
from backpack.utils.examples import _autograd_ggn_exact_columns
from torch import (
    allclose,
    cuda,
    device,
    manual_seed,
    rand,
    rand_like,
    stack,
    zeros_like,
)
from torch.linalg import eigh, eigvalsh
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.extensions.secondorder.vivit import ViViTGGNExact
from vivit.utils.gram import reshape_as_square

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

extension = ViViTGGNExact()
savefield = extension.savefield

with backpack(extension):
    loss.backward()

###############################################################################
#          Compare GGN-vector product of the extension with autograd          #
###############################################################################
num_vecs = 5
vec = [stack([rand_like(p) for _ in range(num_vecs)]) for p in model.parameters()]

# BackPACK result
V_t_vec = sum(
    getattr(p, savefield)["V_t_mat_prod"](v) for p, v in zip(model.parameters(), vec)
)
V_V_t_vec = [getattr(p, savefield)["V_mat_prod"](V_t_vec) for p in model.parameters()]

# autograd result
output = model(X)
loss = loss_function(output, y)
G_vec = [zeros_like(v) for v in V_V_t_vec]
for vector_idx in range(num_vecs):
    vector = [v[vector_idx] for v in vec]
    for param_idx, ggn_vp in enumerate(ggn_vector_product(loss, output, model, vector)):
        G_vec[param_idx][vector_idx] = ggn_vp

# compare
for v1, v2 in zip(V_V_t_vec, G_vec):
    assert allclose(v1, v2), "GGNVPs don't match"
print("GGNVPs match")

###############################################################################
#             Compare eigenvalues of the Gram matrix with autograd            #
###############################################################################

# BackPACK result
gram = sum(getattr(p, savefield)["gram_mat"]() for p in model.parameters())
gram_as_square = reshape_as_square(gram)
gram_evals = eigvalsh(gram_as_square)

# autograd result
G = stack([col for _, col in _autograd_ggn_exact_columns(X, y, model, loss_function)])
G_evals = eigvalsh(G)
G_evals = G_evals[-gram_evals.numel() :]

assert allclose(gram_evals, G_evals, rtol=1e-4, atol=1e-7), "GGN evals don't match"
print("GGN evals match")

###############################################################################
#       Check transformed Gram matrix eigenvectors are eigenvectors of G      #
###############################################################################
gram_evals, gram_evecs = eigh(gram_as_square)

for eval_idx, gram_eval in enumerate(gram_evals):
    evec = gram_evecs[:, eval_idx].reshape(1, *gram.shape[:2])
    evec = [
        getattr(p, savefield)["V_mat_prod"](evec).squeeze(0) for p in model.parameters()
    ]

    G_evec = ggn_vector_product(loss, output, model, evec)

    for v, G_v in zip(evec, G_evec):
        assert allclose(
            gram_eval * v, G_v, rtol=1e-4, atol=5e-6
        ), "Eigenvector property failed"

print("Eigenvector property passed")
