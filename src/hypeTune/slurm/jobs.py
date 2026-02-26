import subprocess
import logging
import pathlib
import time

logger = logging.getLogger(__name__)


def submit_job(trial_number: int, path_run: pathlib.Path) -> int:

    command = f"sbatch {path_run.as_posix()} {trial_number}"
    out = subprocess.check_output(command, shell=True, text=True).strip()
    job_id = int(out.split()[-1])
    logger.info(f"Submitted job with ID {job_id} for trial {trial_number}")
    return job_id


def monitor_job(job_id: int, time_limit: int | None = None) -> None:
    start_time = time.time()
    while True:
        command = f"squeue -j {job_id} -h -o '%T'"
        status = subprocess.check_output(command, shell=True, text=True).strip()
        if status == "COMPLETED":
            logger.info(f"Job {job_id} completed successfully.")
            return 0
        elif status in ["FAILED", "CANCELLED"]:
            logger.error(f"Job {job_id} failed with status: {status}")
            return 1
        if time_limit is not None and time.time() - start_time > time_limit:
            logger.warning(f"Job {job_id} exceeded time limit of {time_limit} seconds.")
            return 2
        time.sleep(10)
