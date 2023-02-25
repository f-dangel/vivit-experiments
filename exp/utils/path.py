"""Experiment utilities for handling paths."""

import json_tricks


def read_from_json(path, verbose=True):
    """Return content of a ``.json`` file.

    Args:
        path (str, Path): Path ending in ``.json`` pointing to the file.
    """
    if verbose:
        print(f"[exp] Reading {path}")

    with open(path) as json_file:
        data = json_tricks.load(json_file)

    return data


def write_to_json(path, data, verbose=True):
    """Write content to a ``.json`` file. Overwrite if file already exists.

    Args:
        path (str, Path): Path ending in ``.json`` pointing to the file.
    """
    if verbose:
        print(f"[exp] Writing to {path}")

    with open(path, "w") as json_file:
        json_tricks.dump(data, json_file)
