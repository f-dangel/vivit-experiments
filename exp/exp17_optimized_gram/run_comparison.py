# NOTE You need to install vivit-for-pytorch for this script to work (the
# library changed a bit in comparison to the research code for the paper)
import time

from backpack import backpack, extend, extensions
from torch import cuda, einsum, manual_seed, randn
from torch.nn import Linear, MSELoss, ReLU, Sequential

from vivit.extensions.secondorder.vivit import ViViTGGNExact

N, C = 128, 10
REPEATS = 20
# NOTE On CPU the overhead is much larger, I believe due to slow memory
# allocation for the larger quantities.
DEVICE = "cuda" if cuda.is_available() else "cpu"


def maybe_synchronize():
    if cuda.is_available():
        cuda.synchronize()


def setup():
    manual_seed(0)

    def mlp():
        return Sequential(
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, C),
        )

    X = randn(N, 1024, device=DEVICE)
    y = randn(N, C, device=DEVICE)

    net = mlp().to(DEVICE)
    loss_func = MSELoss().to(DEVICE)

    return net, loss_func, X, y


# Setting 1


def time_grad():
    net, loss_func, X, y = setup()

    maybe_synchronize()
    start = time.time()

    loss = loss_func(net(X), y)
    loss.backward()
    maybe_synchronize()

    return time.time() - start


t_grad = min(time_grad() for _ in range(REPEATS))

# Setting 2


def time_grad_V():
    net, loss_func, X, y = setup()
    net = extend(net)
    loss_func = extend(MSELoss())

    maybe_synchronize()
    start = time.time()

    loss = loss_func(net(X), y)

    with backpack(extensions.SqrtGGNExact()):
        loss.backward()

    maybe_synchronize()
    return time.time() - start


t_grad_V = min(time_grad_V() for _ in range(REPEATS))
t_V = t_grad_V - t_grad

# Setting 3


class ComputeAccumulateGramHook:
    """Compute and accumulate the Gram matrix during backpropagation with BackPACK."""

    def __init__(self, delete_buffers):
        self.gram = None
        self.delete_buffers = delete_buffers

    def __call__(self, module):
        for p in module.parameters():
            sqrt_ggn = p.sqrt_ggn_exact
            gram_p = einsum("cn...,dm...->cndm", sqrt_ggn, sqrt_ggn)
            self.gram = gram_p if self.gram is None else self.gram + gram_p

            if self.delete_buffers:
                del p.sqrt_ggn_exact


def time_grad_V_G():
    net, loss_func, X, y = setup()
    net = extend(net)
    loss_func = extend(MSELoss())

    hook = ComputeAccumulateGramHook(delete_buffers=True)

    maybe_synchronize()
    start = time.time()

    loss = loss_func(net(X), y)

    with backpack(extensions.SqrtGGNExact(), extension_hook=hook):
        loss.backward()

    _ = hook.gram
    maybe_synchronize()

    return time.time() - start


t_grad_V_G = min(time_grad_V_G() for _ in range(REPEATS))
t_V_G = t_grad_V_G - t_grad

# Setting 4


class AccumulateGramHook:
    """Accumulate the Gram matrix during backpropagation with BackPACK."""

    def __init__(self, delete_buffers):
        self.gram = None
        self.delete_buffers = delete_buffers

    def __call__(self, module):
        for p in module.parameters():
            gram_p = p.vivit_ggn_exact["gram_mat"]()
            self.gram = gram_p if self.gram is None else self.gram + gram_p

            if self.delete_buffers:
                del p.vivit_ggn_exact


def time_grad_V_G_opt():
    net, loss_func, X, y = setup()
    net = extend(net)
    loss_func = extend(MSELoss())

    hook = AccumulateGramHook(delete_buffers=True)

    maybe_synchronize()
    start = time.time()

    loss = loss_func(net(X), y)

    with backpack(ViViTGGNExact(), extension_hook=hook):
        loss.backward()

    _ = hook.gram
    maybe_synchronize()

    return time.time() - start


t_grad_V_G_opt = min(time_grad_V_G_opt() for _ in range(REPEATS))
t_V_G_opt = t_grad_V_G_opt - t_grad

# Print results
print(f"C = {C}")
print("")

print(f"t_grad: {t_grad}")
print(f"t_V: {t_V}")
print(f"t_V_G: {t_V_G}")
print(f"t_V_G_opt: {t_V_G_opt}")
print("")

print(f"t_V / t_grad: {t_V / t_grad}, (expected: {C})")
print(f"t_V_G / t_grad: {t_V_G / t_grad}")
print(f"t_G_opt / t_grad: {t_V_G_opt / t_grad}")
