from hypeTune.fileHandling.write_config import write_config
from hypeTune.fileHandling.read_output import read_tensorboard
from hypeTune.jobSubmission.submit_job import submit_job
from hypeTune.jobSubmission.monitor_job import monitor_job
import numpy as np
import logging
from typing import Union
import pathlib

logger = logging.getLogger(__name__)


def run_case(
    trial_number: int,
    params: dict,
    eval_function: callable[[Union[list, np.ndarray], Union[list, np.ndarray]], float],
    path_write: pathlib.Path,
    path_read: pathlib.Path,
    path_script: pathlib.Path,
    read_metric: str,
):

    write_config(trial_number, params, path_write)
    job_id = submit_job(trial_number, path_script)
    monitor_job(job_id, time_limit=300)
    steps, values = read_tensorboard(trial_number, path_read, read_metric)
    val = eval_function(steps, values)
    logger.debug(f"Trial {trial_number} ended with reward: {val}")

    return val
