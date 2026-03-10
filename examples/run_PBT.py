import numpy as np
from hypeTune.run.run import run_case
from hypeTune.algorithms.PBT import PBT, Hyperparameter
import pathlib
import argparse
import concurrent.futures
import logging
from hypeTune.fileHandling.read_output import read_tensorboard
import os

logger = logging.getLogger(__name__)


def objective(steps, vals):
    """Score a trial from logged values."""
    if len(vals) < 10:
        return np.mean(vals)
    else:
        return np.mean(vals[:-10])


def run_one_case(trial):
    """Sample parameters and run a single trial."""
    width = trial.hyperparameters["width"]
    trial_number = trial.trial_number
    load_from = trial.parent_trial
    params = {
        "width": width,
        "load_weights_from": load_from,
    }

    score = run_case(
        trial_number,
        params,
        eval_function=objective,
        path_write=pathlib.Path("./configs"),
        path_read=pathlib.Path("./logs"),
        path_script=pathlib.Path("./run_script.py"),
        path_venv=pathlib.Path("./venv"),
        read_metric="reward",
    )

    return (trial_number, score)


def run_trials(total_cases: int, max_concurrent: int) -> None:
    """Run multiple trials with bounded concurrency."""

    lr = Hyperparameter(
        name="learning_rate",
        values=(1e-6, 1e-4),
        log_space=True,
    )

    clip_param = Hyperparameter(
        name="clip_param",
        values=(0.1, 0.4),
        log_space=False,
        perturbation_range=(0.95, 1.05),
    )

    population_size = max_concurrent
    total_trials = total_cases

    pbt = PBT(
        hyperparameters=[lr, clip_param],
        population_size=population_size,
        upper_percentile=30,
        lower_percentile=50,
        total_trials=total_trials,
        minimum_trials=population_size,
        always_perturb=False,
    )

    iterations = total_trials // population_size

    if os.path.exists("./PBT_out.json"):
        pbt.from_json(pathlib.Path("./PBT_out.json"))

    for _ in range(iterations):
        trials = pbt.get_new_trials()
        pbt.to_json(pathlib.Path("./PBT_out.json"))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            futures = [executor.submit(run_one_case, trial) for trial in trials]
            for idx, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                try:
                    out = future.result()
                    logger.info("Completed %d/%d trial submissions", idx, total_cases)
                    pbt.update_trial(out[0], out[1])
                except Exception:
                    logger.exception("A trial worker failed (%d/%d).", idx, total_cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()

    if args.n_cases < 1:
        raise ValueError("--n-cases must be >= 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be >= 1")

    logging.basicConfig(level=logging.INFO)
    run_trials(total_cases=args.n_cases, max_concurrent=args.max_concurrent)
