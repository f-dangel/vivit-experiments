"""Problem definition (data and model) for the examples."""

import torch
from utils import classification_targets

# Dummy model and data
N, in_features = 3, 10
hidden_features = 7
C = 5


def make_data():
    return torch.rand(N, in_features), classification_targets((N,), C)


def make_model():
    """Simple FCNN with  one hidden layer and sigmoid activation."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_features),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_features, C),
    )


def make_loss_function():
    return torch.nn.CrossEntropyLoss(reduction="mean")
