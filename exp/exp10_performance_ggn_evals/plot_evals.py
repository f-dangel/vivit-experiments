"""Plot GGN eigenvalue spectra."""

import os

import matplotlib.pyplot as plt
import numpy
from run_evals import (
    architecture_cases,
    batch_size_key,
    device,
    get_output_file,
    num_params_key,
    param_groups_cases,
)
from shared import eigenvalue_cutoff
from shared_plot import COLORS

from exp.utils.path import read_from_json
from exp.utils.plot import TikzExport

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)
FIG_DIR = os.path.join(HEREDIR, "fig", "evals")


def fill_missing_zeros(spectrum, total):
    """Add zero eigenvalues to spectrum."""
    spectrum_with_zeros = numpy.zeros(total)
    spectrum_with_zeros[: len(spectrum)] = spectrum

    return spectrum_with_zeros


def load(architecture, param_groups, device):
    """Load and pre-process experiment data (fill zero eigenvalues)."""
    data_file = get_output_file(architecture, device, param_groups)
    spectra = read_from_json(data_file)

    N = spectra.pop(batch_size_key)
    D = spectra.pop(num_params_key)

    for computation in spectra.keys():
        spectra[computation] = fill_missing_zeros(spectra[computation], D)

    return N, D, spectra


def get_fig_savepath(architecture, param_groups, device, computation, extension=""):
    """Return path where figure for a configuration is stored."""
    return os.path.join(
        FIG_DIR,
        f"{architecture.__name__}_{device}_{param_groups.__name__}"
        + f"_{computation}{extension}",
    )


def get_xlimits(data):
    """Determine x limits shared among computations."""
    xmin = eigenvalue_cutoff

    max_eigvals = [numpy.max(evals) for evals in data.values()]
    xmax = numpy.max(max_eigvals)

    return xmin, xmax


def plot(architecture, param_groups, device, num_bins=50, logscale=True):
    """Plot the different spectra at each step of training."""
    N, D, data = load(architecture, param_groups, device)

    xmin, xmax = get_xlimits(data)
    ymin, ymax = 1 / (D + 1), 1

    for computation, spectrum in data.items():
        title = f"{computation}, {param_groups.__name__}, N={N}, D={D}"
        print(f"Generating plot for {title}")

        plt.figure()
        plt.title(title)
        plt.xlabel("eigenvalues")
        plt.ylabel("density")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        if logscale:
            bins = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), num=num_bins)
            plt.xscale("log")
        else:
            bins = numpy.linspace(xmin, xmax, num=num_bins)

        # clip outliers as they would otherwise be ignored
        assert numpy.max(spectrum) <= xmax, "No outliers on the right"
        spectrum = numpy.clip(spectrum, xmin, xmax)

        plt.hist(
            spectrum,
            log=logscale,
            bins=bins,
            weights=numpy.ones_like(spectrum) / D,
            color=COLORS[computation],
            bottom=ymin,
        )

        savepath = get_fig_savepath(architecture, param_groups, device, computation)
        TikzExport().save_fig(savepath, tex_preview=False)

        plt.close("all")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    # plot
    for architecture in architecture_cases:
        for param_groups in param_groups_cases:
            plot(architecture, param_groups, device, num_bins=30)
