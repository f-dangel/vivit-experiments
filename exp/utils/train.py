"""Utility functions for evaluating models during training."""


def get_accuracy(output, targets):
    """Compute accuracy from model predictions and ground truth."""
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()
