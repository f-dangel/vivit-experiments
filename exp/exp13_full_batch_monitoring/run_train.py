"""This script runs `train.py` either locally or on the Slurm cluster. It creates only
one job that executes the training for all configurations. 
"""

import subprocess

from config import CHECKPOINTS_OUTPUT

from exp.utils.slurm_utils import SlurmJob

SLURM = SlurmJob.sbatch_exists()
VERBOSE = True


if __name__ == "__main__":
    # Construct command line
    cmd_str = r"python ./train.py "
    if VERBOSE:
        print("cmd_str = ", cmd_str)

    # Execute on Slurm or locally
    if SLURM:
        SJ = SlurmJob(
            job_name="run_train",
            job_path=CHECKPOINTS_OUTPUT,
            gpu="gpu-2080ti",
            time="3-00:00:00",
            memory="32G",
            cmd_str=cmd_str,
        )

        if VERBOSE:
            print("Slurmjob batch script content: \n", SJ.create_bash_str())

        SJ.submit()
    else:
        subprocess.check_call(cmd_str)
