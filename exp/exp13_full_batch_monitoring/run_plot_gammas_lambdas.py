"""This script coordinates the execution of the plotting script 
`plot_gammas_lambdas.py` for all configurations. It creates one job for each 
configuration. The switch `SLRUM` allows `plot_gammas_lambdas.py` to be executed 
in parallel on Slurm or locally in a sequential way.
"""

import os
import subprocess

from config import CONFIGURATIONS, PLOT_OUTPUT, config_to_config_str

from exp.utils.slurm_utils import SlurmJob

SLURM = SlurmJob.sbatch_exists()
VERBOSE = False
PLOT_SUBDIR = os.path.join(PLOT_OUTPUT, "gammas_lambdas")


def get_plot_savedir():
    """Return subdirectory `PLOT_SUBDIR` for this plotting script"""
    savedir = PLOT_SUBDIR
    os.makedirs(savedir, exist_ok=True)
    return savedir


if __name__ == "__main__":
    for config in CONFIGURATIONS:
        # Construct command line
        config_str = config_to_config_str(config)
        cmd_str = rf"python ./plot_gammas_lambdas.py --config_str {config_str}"

        # Construct job-path
        job_path = get_plot_savedir()

        print(f"config_str = {config_str}")
        if VERBOSE:
            print(f"cmd_str = {cmd_str}")
            print(f"job_path = {job_path}")

        # Execute on Slurm or locally
        if SLURM:
            SJ = SlurmJob(
                job_name="run_plot_gammas_lambdas",
                job_path=job_path,
                gpu="gpu-2080ti",
                time="3-00:00:00",
                memory="32G",
                cmd_str=cmd_str,
            )

            if VERBOSE:
                print("Slurmjob batch script content: \n", SJ.create_bash_str())

            SJ.submit()
        else:
            subprocess.check_call(cmd_str.split())
