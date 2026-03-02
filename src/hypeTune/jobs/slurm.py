import subprocess
import logging
import pathlib
import time
from typing import Callable

logger = logging.getLogger(__name__)


def submit_job(trial_number: int, path_run: pathlib.Path) -> int:

    command = ["sbatch", path_run.as_posix(), str(trial_number)]
    out = subprocess.check_output(command, text=True).strip()
    job_id = int(out.split()[-1])
    logger.info(
        f"Submitted job with ID {job_id}",
        extra={"time": time.time(), "iteration": trial_number},
    )
    return job_id


def monitor_job(
    job_id: int,
    time_limit: int | None = None,
    poll_interval: float = 10.0,
    callback: Callable[[int, float], None] | None = None,
) -> int:
    start_time = time.time()

    # -j: job ID
    # -X: Show only the main job allocation (ignores sub-steps like .batch or .extern)
    # -n: No header
    # -o State: Output only the job state
    command = ["sacct", "-j", str(job_id), "-X", "-n", "-o", "State"]

    while True:
        try:
            output = subprocess.check_output(command, text=True).strip()

            # If the output is totally empty, Slurm's accounting database
            # might not have registered the job yet. We default to UNKNOWN.
            status = output.split()[0] if output else "UNKNOWN"

        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to query job {job_id}: {e}", extra={"time": time.time()}
            )
            status = "UNKNOWN"

        if status == "COMPLETED":
            logger.info(
                f"Job {job_id} completed successfully.", extra={"time": time.time()}
            )
            return 0
        elif status in ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"]:
            logger.info(
                f"Job {job_id} terminated with status: {status}",
                extra={"time": time.time()},
            )
            return 1

        if time_limit is not None and time.time() - start_time > time_limit:
            logger.warning(
                f"Job {job_id} exceeded time limit of {time_limit} seconds (including queue time).",
                extra={"time": time.time()},
            )
            return 2

        if callback is not None:
            try:
                callback(job_id, time.time() - start_time)
            except Exception:
                logger.warning(
                    f"Failed to execute callback for job ID {job_id}",
                    extra={"time": time.time()},
                )
        time.sleep(poll_interval)
