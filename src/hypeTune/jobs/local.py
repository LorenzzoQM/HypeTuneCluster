import subprocess
import logging
import pathlib
import time
import signal
import os
from typing import Callable

logger = logging.getLogger(__name__)
_LOCAL_PROCESSES: dict[int, subprocess.Popen[str]] = {}


def submit_job(
    trial_number: int, path_venv: pathlib.Path, path_run: pathlib.Path
) -> int:
    """Start a local trial process and return its PID."""
    cmd = [
        str(path_venv / "bin" / "python"),
        str(path_run),
        f"--trial-number={trial_number}",
    ]
    logger.info(f"Submitting job with command: {cmd}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    _LOCAL_PROCESSES[process.pid] = process
    logger.info(
        f"Job submitted successfully with PID {process.pid}.",
        extra={"time": time.time()},
    )
    return process.pid


def _is_pid_alive(pid: int) -> bool:
    """Check whether a PID is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def monitor_job(
    job_pid: int,
    time_limit: int | None = None,
    poll_interval: float = 10.0,
    callback: Callable[[int, float], None] | None = None,
) -> int:
    """Poll a local job until completion, failure, or timeout."""
    start_time = time.time()
    process = _LOCAL_PROCESSES.get(job_pid)

    while True:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                stdout, stderr = process.communicate()
                _LOCAL_PROCESSES.pop(job_pid, None)
                if return_code == 0:
                    logger.info(
                        f"Job PID {job_pid} completed successfully.",
                        extra={"time": time.time()},
                    )
                    if stdout:
                        logger.debug(
                            f"Job PID {job_pid} stdout: {stdout.strip()}",
                            extra={"time": time.time()},
                        )
                    return 0
                logger.error(
                    f"Job PID {job_pid} failed (exit {return_code}): {stderr.strip()}",
                    extra={"time": time.time()},
                )
                return 1
        else:
            if not _is_pid_alive(job_pid):
                logger.info(
                    f"Job PID {job_pid} is no longer running.",
                    extra={"time": time.time()},
                )
                return 0

        if time_limit is not None and time.time() - start_time > time_limit:
            logger.warning(
                f"Job PID {job_pid} exceeded time limit of {time_limit} seconds.",
                extra={"time": time.time()},
            )
            kill_job(job_pid)
            return 2

        if callback is not None:
            try:
                callback(job_pid, time.time() - start_time)
            except Exception:
                logger.warning(
                    f"Failed to execute callback for job PID {job_pid}",
                    extra={"time": time.time()},
                )
        time.sleep(poll_interval)


def kill_job(job_pid: int, kill_grace_seconds: float = 5.0) -> None:
    """Terminate a local job, escalating to SIGKILL if needed."""
    process = _LOCAL_PROCESSES.get(job_pid)

    if process is not None:
        if process.poll() is not None:
            _LOCAL_PROCESSES.pop(job_pid, None)
            return
        process.terminate()
        try:
            process.wait(timeout=kill_grace_seconds)
            logger.info(f"Terminated job PID {job_pid}.", extra={"time": time.time()})
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=kill_grace_seconds)
            logger.warning(
                f"Killed job PID {job_pid} after timeout.", extra={"time": time.time()}
            )
        finally:
            _LOCAL_PROCESSES.pop(job_pid, None)
        return

    if not _is_pid_alive(job_pid):
        return

    os.kill(job_pid, signal.SIGTERM)
    deadline = time.time() + kill_grace_seconds
    while time.time() < deadline:
        if not _is_pid_alive(job_pid):
            logger.info(f"Terminated job PID {job_pid}.", extra={"time": time.time()})
            return
        time.sleep(0.2)
    os.kill(job_pid, signal.SIGKILL)
    logger.warning(
        f"Killed job PID {job_pid} after timeout.", extra={"time": time.time()}
    )
