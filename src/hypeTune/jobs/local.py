import subprocess
import logging
import pathlib
import time
import signal
import os

logger = logging.getLogger(__name__)
_LOCAL_PROCESSES: dict[int, subprocess.Popen[str]] = {}


def submit_job(
    trial_number: int, path_venv: pathlib.Path, path_run: pathlib.Path
) -> int:
    cmd = [
        str(path_venv / "bin" / "python"),
        str(path_run),
        f"--trial-number={trial_number}",
    ]
    logger.info("Submitting job with command: %s", cmd)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    _LOCAL_PROCESSES[process.pid] = process
    logger.info("Job submitted successfully with PID %d.", process.pid)
    return process.pid


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def monitor_job(
    job_pid: int, time_limit: int | None = None, poll_interval: float = 10.0
) -> int:
    start_time = time.time()
    process = _LOCAL_PROCESSES.get(job_pid)

    while True:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                stdout, stderr = process.communicate()
                _LOCAL_PROCESSES.pop(job_pid, None)
                if return_code == 0:
                    logger.info("Job PID %d completed successfully.", job_pid)
                    if stdout:
                        logger.debug("Job PID %d stdout: %s", job_pid, stdout.strip())
                    return 0
                logger.error(
                    "Job PID %d failed (exit %d): %s",
                    job_pid,
                    return_code,
                    stderr.strip(),
                )
                return 1
        else:
            if not _is_pid_alive(job_pid):
                logger.info("Job PID %d is no longer running.", job_pid)
                return 0

        if time_limit is not None and time.time() - start_time > time_limit:
            logger.warning(
                "Job PID %d exceeded time limit of %d seconds.", job_pid, time_limit
            )
            kill_job(job_pid)
            return 2

        time.sleep(poll_interval)


def kill_job(job_pid: int, kill_grace_seconds: float = 5.0) -> None:
    process = _LOCAL_PROCESSES.get(job_pid)

    if process is not None:
        if process.poll() is not None:
            _LOCAL_PROCESSES.pop(job_pid, None)
            return
        process.terminate()
        try:
            process.wait(timeout=kill_grace_seconds)
            logger.info("Terminated job PID %d.", job_pid)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=kill_grace_seconds)
            logger.warning("Killed job PID %d after timeout.", job_pid)
        finally:
            _LOCAL_PROCESSES.pop(job_pid, None)
        return

    if not _is_pid_alive(job_pid):
        return

    os.kill(job_pid, signal.SIGTERM)
    deadline = time.time() + kill_grace_seconds
    while time.time() < deadline:
        if not _is_pid_alive(job_pid):
            logger.info("Terminated job PID %d.", job_pid)
            return
        time.sleep(0.2)
    os.kill(job_pid, signal.SIGKILL)
    logger.warning("Killed job PID %d after timeout.", job_pid)
