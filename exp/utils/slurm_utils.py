import os
import re
import stat
import subprocess
from shutil import which


class SlurmJob:
    """The Slurm job class can be used for conveniently submitting jobs to the Slurm
    cluster. It creates a bash script with the relevant commands, executes it and
    deletes it afterwards.

    Example:
    ``
    SJ = SlurmJob(
            job_name="run",
            time="0-00:01:00",
            memory="8G",
            cmd_str="python ./run.py --param=1.0",
        )
    SJ.submit()
    ``
    """

    def __init__(
        self,
        job_name,
        time,
        memory,
        cmd_str,
        job_path=None,
        gpu="gpu-2080ti-dev",
        num_gpus=1,
        email=None,
        email_type="ALL",
    ):
        """
        SlurmJob constructor: Store (and check some of the) parameters

        Args:
            job_name (string): Name of the Slurm job
            time (string): Maximum runtime specified in the format D-HH:MM:SS, e.g.
                ``"0-08:00:00"`` for 8 hours. This needs to be compatible with the
                partition specified by ``gpu``.
            memory (string): Amount of memory used by the job specified in megabytes
                (suffix ``M``) or gigabytes (suffix ``G``), e.g. ``"16G"`` for 16
                gigabytes
            cmd_str (string): The command line string for executing the actual program
                code, e.g. ``"python ./run.py --param=1.0"``
            job_path (os.path object): The output and error files will be stored in this
                folder. Default is ``None``, i.e. the output and error file are stored
                in the working directory.
            gpu (string): This setting specifies the partition. For a complete list of
                available partitions, execute the Slurm command ``sinfo -s``. Examples:
                gpu-2080ti-dev: 12h time limit (default)
                gpu-2080ti:     3d time limit
                gpu-v100:       3d time limit
            num_gpus (int): The number of GPUs. Default is ``1``
            email (string): Notifications are sent to this email address. Default is
                ``None`` in which case no notifications are sent.
            email_type (string): Type of email notification: ``"BEGIN"``, ``"END"``,
                ``"FAIL"`` or ``"ALL"`` (default)
        """
        self.job_name = job_name
        self.cmd_str = cmd_str
        self.job_path = job_path
        self.gpu = gpu
        self.num_gpus = num_gpus
        self.email = email
        self.email_type = email_type

        self.check_time_format(time)
        self.time = time

        self.check_memory_format(memory)
        self.memory = memory

        self.output_file_path, self.error_file_path = self.create_file_paths()

    @staticmethod
    def check_time_format(time):
        """Make sure that ``time`` has the right format D-HH:MM:SS."""
        time_format = r"^(\d{1})-(\d{2}):(\d{2}):(\d{2})$"
        assert re.match(time_format, time), "Specify time in format D-HH:MM:SS"

    @staticmethod
    def check_memory_format(memory):
        """Make sure that ``memory`` has the right format, e.g. ``"13.4M"`` (for 13.4
        megabytes) or ``"8G"`` (for 8 gigabytes).
        """
        memory_format = r"^(\d+)\.(\d+)[G,M]$|^(\d+)[G,M]$"
        assert re.match(memory_format, memory), "Memory has incorrect format"

    def create_file_paths(self):
        """Create output and error file names based on ``self.job_name``, return
        absolute paths.
        """
        # ``%j`` is a placeholder for the job-id and will be filled in by Slurm
        output_file = self.job_name + r"_%j.out"
        error_file = self.job_name + r"_%j.err"

        if self.job_path is not None:
            output_file = os.path.join(self.job_path, output_file)
            error_file = os.path.join(self.job_path, error_file)

        return os.path.abspath(output_file), os.path.abspath(error_file)

    def create_sbatch_str(self):
        """Create the configuration string that contains the ``SBATCH`` commands."""
        sbatch_str = (
            f"#SBATCH --job-name={self.job_name} \n"
            + "#SBATCH --ntasks=1 \n"
            + "#SBATCH --cpus-per-task=1 \n"
            + "#SBATCH --nodes=1 \n"
            + f"#SBATCH --mem={self.memory} \n"
            + f"#SBATCH --partition={self.gpu} \n"
            + f"#SBATCH --gres=gpu:{self.num_gpus} \n"
            + f"#SBATCH --time={self.time} \n"
            + f"#SBATCH --output={self.output_file_path} \n"
            + f"#SBATCH --error={self.error_file_path} \n"
        )

        # Append email configuration, if available
        if self.email is not None:
            sbatch_str += (
                f"#SBATCH --mail-type={self.email_type} \n"
                + f"#SBATCH --mail-user={self.email} \n"
            )

        return sbatch_str

    @staticmethod
    def create_scontrol_str():
        """Create the ``scontrol`` string. The command below prints important
        information to the output file.
        """
        return r"scontrol show job $SLURM_JOB_ID"

    def create_cmd_str(self):
        """Create the command line string for executing the actual program."""
        return self.cmd_str

    def create_bash_str(self):
        """Create one string that represents the content of the bash file. It basically
        just joins the components defined above.
        """
        bash_str = (
            "#!/bin/bash\n\n"
            + self.create_sbatch_str()
            + "\n"
            + self.create_scontrol_str()
            + "\n"
            + self.create_cmd_str()
        )
        return bash_str

    def submit(self):
        """Submit the job to Slurm: Create the bash script, execute and delete it after
        execution. Note that the ``sbatch`` command returns immediately.

        Raises:
            RuntimeError: If ``sbatch`` command is not available.
        """
        if not self.sbatch_exists():
            raise RuntimeError("No 'sbatch' command found on the system.")

        if self.job_path is not None:
            os.makedirs(self.job_path, exist_ok=True)

        bash_file_name = f"./{self.job_name}.sh"
        with open(bash_file_name, "w") as f:
            f.write(self.create_bash_str())

        os.chmod(bash_file_name, stat.S_IRWXU)
        try:
            subprocess.run("sbatch " + bash_file_name, shell=True, check=True)
        except Exception as e:
            raise e
        finally:
            os.remove(bash_file_name)

    @staticmethod
    def sbatch_exists() -> bool:
        """Whether the ``sbatch`` command is available.

        Returns:
            If ``sbatch`` is available on the system.
        """
        return which("sbatch") is not None
