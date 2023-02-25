"""Shared ploting functionality among sub-experiments."""

from palettable.colorbrewer.sequential import Reds_7

###############################################################################
#                        Plotting styles (colors, etc.)                       #
###############################################################################
COLORS = {
    "power iteration": Reds_7.mpl_colors[4],
    "full_batch_exact": "#000000",
    "frac_batch_exact": "#469990",
    "full_batch_mc": "#de9f16",
    "frac_batch_mc": "#802f99",
}
