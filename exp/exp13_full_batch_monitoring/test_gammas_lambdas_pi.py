"""This is just a very basic CI script used for debugging the 
``eval_gammas_lambdas_pi`` function.
"""

import torch
from utils_shared import eval_gammas_lambdas_pi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct data, model and loss function
N = 4
D_out = 3
D_hidden = 5
D_in = 7

torch.manual_seed(0)
X = torch.rand(N, D_in)
y = torch.rand(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_hidden),
    torch.nn.ReLU(),
    torch.nn.Sequential(
        torch.nn.Linear(D_hidden, D_hidden),
        torch.nn.ReLU(),
    ),
    torch.nn.Linear(D_hidden, D_out),
)

loss_function = torch.nn.MSELoss(reduction="mean")

gammas_nk, lambdas_nk = eval_gammas_lambdas_pi(
    model,
    loss_function,
    (X, y),
    D_out,
    DEVICE,
    check_deterministic=True,
    verbose=True,
)

print("gammas_nk = \n", gammas_nk)
print("lambdas_nk = \n", lambdas_nk)
