"""Utility functions to control random seeds."""

import torch


class temporary_seed:
    """Temporarily set PyTorch seed to a different value, then restore current value.

    This has the effect that code inside this context does not influence the outer
    loop's random generator state.
    """

    def __init__(self, temp_seed):
        self._temp_seed = temp_seed

    def __enter__(self):
        """Store the current seed."""
        self._old_state = torch.get_rng_state()
        torch.manual_seed(self._temp_seed)

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the old random generator state."""
        torch.set_rng_state(self._old_state)


def test_temporary_seed():
    """Test if temporary_seed works as expected."""
    torch.manual_seed(3)

    num1 = torch.rand(1)

    with temporary_seed(2):
        num2 = torch.rand(1)

    num3 = torch.rand(1)

    torch.manual_seed(3)

    num4 = torch.rand(1)
    num5 = torch.rand(1)

    torch.manual_seed(2)

    num6 = torch.rand(1)

    assert torch.allclose(num1, num4)
    assert torch.allclose(num3, num5)
    assert torch.allclose(num2, num6)


if __name__ == "__main__":
    test_temporary_seed()
