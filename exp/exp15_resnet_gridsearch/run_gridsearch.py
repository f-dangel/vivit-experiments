"""This script runs the grid search (implemented by ``gridsearch_SGD.py`` and 
``gridsearch_Adam.py``) either locally or on the Slurm cluster. It creates one job 
for each script. 
"""

import os
import subprocess

from exp.utils.slurm_utils import SlurmJob

SLURM = True
VERBOSE = True
CMD_STRS = [
    r"python ./gridsearch_SGD.py ",
    r"python ./gridsearch_Adam.py ",
]
HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)

if __name__ == "__main__":
    for cmd_str in CMD_STRS:
        if VERBOSE:
            print("cmd_str = ", cmd_str)

        if SLURM:
            SJ = SlurmJob(
                job_name="run_gridsearch",
                job_path=HEREDIR,
                gpu="gpu-2080ti-long",
                time="10-00:00:00",
                memory="32G",
                cmd_str=cmd_str,
            )

            if VERBOSE:
                print("Slurmjob batch script content: \n", SJ.create_bash_str())

            SJ.submit()
        else:
            subprocess.check_call(cmd_str)
