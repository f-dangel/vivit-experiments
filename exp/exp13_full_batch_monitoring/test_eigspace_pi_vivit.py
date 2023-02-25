"""Compare the output of the functions ``eval_eigspace_vivit`` and ``eval_eigspace_pi`` 
implemented in ``utils_shared.py``. Both function compute eigenspaces on a mini-
batch (using ViViT or the power iteration method implemented by ``GGNLinearOperator``, 
respectively). Using ViViT without curvature subsampling and without MC-sampling, the 
results should be identical. This is tested here for a toy problem.  
"""

import torch
from utils_shared import eval_eigspace_pi, eval_eigspace_vivit, subspaces_overlap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
IN = 100
HIDDEN = 64
OUT = 5
model = torch.nn.Sequential(
    torch.nn.Linear(IN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, OUT)
)

lossfunc = torch.nn.CrossEntropyLoss()

# Data
BATCH_SIZE = 32
torch.manual_seed(0)
X, y = (
    torch.rand((BATCH_SIZE, IN)),
    torch.randint(0, OUT, (BATCH_SIZE,)),
)

# Power iteration
evals_pi, evecs_pi = eval_eigspace_pi(model, lossfunc, (X, y), OUT, DEVICE)

# ViViT
CASE = {"batch_size": BATCH_SIZE, "subsampling": None, "mc_samples": 0}
evals_vivit, evecs_vivit = eval_eigspace_vivit(
    CASE, model, lossfunc, (X, y), OUT, DEVICE
)

# Comparison for evecs
print("pairwise evec overlaps = \n", evecs_pi.T @ evecs_vivit)

overlap = subspaces_overlap(evecs_pi, evecs_vivit, OUT)
print("Overlap = ", overlap)
assert torch.allclose(torch.Tensor([overlap]), torch.ones(1)), "Overlap not close to 1."
print("Test 1 passed.")

# Comparison for evals
assert torch.allclose(evals_pi, evals_vivit), "Eigenvalues not close."
print("Test 2 passed.")
