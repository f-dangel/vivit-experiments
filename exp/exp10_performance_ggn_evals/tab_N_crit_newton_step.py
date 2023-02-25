"""Create LaTeX tables of critical batch sizes for Newton step computation."""

from os import makedirs, path
from typing import Dict

from call_run_N_crit_newton_step import N_max, configurations, get_output_file
from run_N_crit_newton_step import (
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
)

from exp.utils.path import read_from_json

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
FIG_DIR = path.join(HEREDIR, "fig", "N_crit", "newton")


def load(architecture: str, device: str, param_groups: str) -> Dict[str, int]:
    """Load critical batch sizes of a configuration."""
    output_file = get_output_file(architecture, device, param_groups)

    return read_from_json(output_file)


def tabularize(data: Dict[str, int]) -> str:
    """Create LaTeX table of the critical batch sizes."""

    def format(n: int) -> str:
        """Format the critical batch size."""
        if n > N_max:
            return f"> {N_max}"
        else:
            return str(n)

    prefix = r"""\begin{tabular}{lll}
    \toprule
    $_{\text{\tiny{\ggn}}}$$^{\text{\tiny{Data}}}$ & mb & sub \\
    \midrule"""
    body = rf"""
    exact & {format(data[full_batch_exact.__name__])}
              & {format(data[frac_batch_exact.__name__])} \\
    mc   & {format(data[full_batch_mc.__name__])}
              & {format(data[frac_batch_mc.__name__])} \\
"""
    postfix = r"""    \bottomrule
\end{tabular}"""

    return prefix + body + postfix


def get_tab_savepath(
    architecture: str, device: str, param_groups: str, extension: str = ".tex"
) -> str:
    """Return save path of a figure."""
    return path.join(FIG_DIR, f"tab_{architecture}_{device}_{param_groups}{extension}")


def create_tables():
    for architecture, device, param_groups in configurations():
        data = load(architecture, device, param_groups)
        table = tabularize(data)
        savepath = get_tab_savepath(architecture, device, param_groups)

        print(f"Writing table {savepath}")
        makedirs(path.dirname(savepath), exist_ok=True)

        with open(savepath, "w") as f:
            f.write(table)


if __name__ == "__main__":
    create_tables()
