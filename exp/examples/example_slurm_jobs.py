"""This is an example how to use the ``SlurmJob``-class (in ``exp.utils.slurm_utils``)
to run the python function ``run_optimizer`` in ``run_optimizer.py`` (in this
directory). The results are stored in a subfolder ``results`` (see ``job_path`` variable
below).
"""

import itertools
import os
import subprocess

from exp.utils.slurm_utils import SlurmJob

if __name__ == "__main__":
    # Compute on SLURM cluster
    SLURM = SlurmJob.sbatch_exists()  # to make the CI pass

    # Specify hyperparameters
    lr_list = [0.1, 0.01]
    damping_list = [0.1, 1.0, 10.0]

    # Specify folder for results: This can be a separate (sub-)folder for every job!
    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)
    job_path = os.path.join(HEREDIR, "results")

    # Run optimizer with all hyperparameter combinations
    for lr, damping in itertools.product(lr_list, damping_list):
        cmd_str = rf"python ./run_optimizer.py --lr={lr} --damping={damping}"

        if SLURM:
            SJ = SlurmJob(
                job_name="run_optimizer",
                job_path=job_path,
                time="0-00:00:10",
                memory="4G",
                cmd_str=cmd_str,
            )
            SJ.submit()
        else:
            subprocess.run(cmd_str, shell=True)
