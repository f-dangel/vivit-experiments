"""This file contains shared plotting utilities."""

from tueplots import axes, bundles

AXES_DEFAULTS = {
    **axes.lines(),
    **axes.grid(grid_linestyle="dashed"),
    **axes.legend(),
    **axes.color(),
    **axes.spines(),
    **axes.tick_direction(),
}

TUE_DEFAULTS = bundles.neurips2022()

# Add `AXES_DEFAULTS` to `TUE_DEFAULTS`
for key in AXES_DEFAULTS.keys():
    TUE_DEFAULTS[key] = AXES_DEFAULTS[key]

# Add custom settings
TUE_DEFAULTS["legend.facecolor"] = "white"
TUE_DEFAULTS["legend.framealpha"] = 1


def add_tue_defaults(config):
    """Add the default key-value pairs to `config` if they are not specified
    by `config`.
    """

    for defaults_key, defaults_value in TUE_DEFAULTS.items():
        if defaults_key not in config.keys():
            config[defaults_key] = defaults_value
    return config
