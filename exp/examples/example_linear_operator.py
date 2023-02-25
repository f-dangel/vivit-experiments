"""Use the Hessian as SciPy linear operator."""

from scipy.sparse.linalg import eigsh
from torch import cuda, device, manual_seed, rand
from torch.nn import Linear, MSELoss

from vivit.hessianfree import HessianLinearOperator

manual_seed(0)
dev = device("cuda" if cuda.is_available() else "cpu")

N, D_in, D_out = 3, 10, 2
model = Linear(D_in, D_out)
loss_func = MSELoss(reduction="mean")

# data set with some mini-batches
data = []
num_batches = 8
for _ in range(num_batches):
    data.append((rand(N, D_in), rand(N, D_out)))

hessian = HessianLinearOperator(model, loss_func, data, dev, progressbar=True)
# use the SciPy routine
evals, evecs = eigsh(hessian, k=1)
grad, loss = hessian.gradient_and_loss()

print(f"Eigenvalues:        {evals}")
print(f"Eigenvectors shape: {evecs.shape}")
print(f"Loss:               {loss}")
print(f"Gradient shapes:    {[g.shape for g in grad]}")
