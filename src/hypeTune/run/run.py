from hypeTune.fileHandling.write_config import write_config
from hypeTune.fileHandling.read_output import read_tensorboard
from hypeTune.jobs import local, slurm
import numpy as np
import logging
from typing import Union, Callable
import pathlib

logger = logging.getLogger(__name__)


def run_case(
    trial_number: int,
    params: dict,
    eval_function: Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], float],
    path_write: pathlib.Path,
    path_read: pathlib.Path,
    path_script: pathlib.Path,
    read_metric: str,
    path_venv: pathlib.Path | None = None,
    time_limit: int | None = None,
):

    write_config(trial_number, params, path_write)
    if path_venv is None:
        job_id = slurm.submit_job(trial_number, path_script)
        slurm.monitor_job(job_id, time_limit=time_limit)
    else:
        job_id = local.submit_job(trial_number, path_venv, path_script)
        local.monitor_job(job_id, time_limit=time_limit)
    steps, values = read_tensorboard(trial_number, path_read, read_metric)
    val = eval_function(steps, values)
    logger.debug(f"Trial {trial_number} ended with reward: {val}")

    return val
