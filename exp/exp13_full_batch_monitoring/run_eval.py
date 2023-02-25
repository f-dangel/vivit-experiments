"""This script coordinates the evaluation for all checkpoints. It loops over all 
configurations and checkpoints and turns them into command line arguments for `eval.py`.
Using the switch `SLURM`, `eval.py` is either executed locally or on the Slurm cluster.
"""

import os
import subprocess

from config import CONFIGURATIONS, EVAL_OUTPUT, config_to_config_str

from exp.utils.slurm_utils import SlurmJob

SLURM = True
VERBOSE = True


def get_eval_savedir(problem_cls, optimizer_cls, checkpoint):
    """Return sub-directory where results of the evaluation are saved."""
    epoch_count, batch_count = checkpoint
    savedir = os.path.join(
        EVAL_OUTPUT,
        f"{problem_cls.__name__}",
        f"{optimizer_cls.__name__}",
        f"epoch_{epoch_count:05d}_batch_{batch_count:05d}",
    )
    os.makedirs(savedir, exist_ok=True)
    return savedir


if __name__ == "__main__":
    for config in CONFIGURATIONS:
        problem_cls = config["problem_cls"]
        optimizer_cls = config["optimizer_cls"]
        config_str = config_to_config_str(config)

        for checkpoint in config["checkpoints"]:
            # Construct command line
            checkpoint_epoch, checkpoint_batch = checkpoint
            cmd_str = r"python ./eval.py "
            cmd_str += rf"--config_str {config_str} "
            cmd_str += rf"--checkpoint {checkpoint_epoch} {checkpoint_batch}"

            if VERBOSE:
                print(f"config_str = {config_str}, checkpoint = {checkpoint}")
                print(f"cmd_str = {cmd_str}")

            # Execute on Slurm or locally
            if SLURM:
                SJ = SlurmJob(
                    job_name="run_eval",
                    job_path=get_eval_savedir(problem_cls, optimizer_cls, checkpoint),
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
