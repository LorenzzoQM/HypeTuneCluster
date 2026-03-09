import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import numpy as np
from hypeTune.run.run import run_case
import pathlib
import argparse
import concurrent.futures
import logging
from hypeTune.fileHandling.read_output import read_tensorboard

logger = logging.getLogger(__name__)


def objective(steps, vals):
    """Score a trial from logged values."""
    if len(vals) < 10:
        return np.mean(vals)
    else:
        return np.mean(vals[:-10])


def run_one_case(trial):
    """Sample parameters and run a single trial."""
    width = trial.suggest_int("width", 26, 38)
    trial_number = trial.number
    params = {"width": width}

    def prune_callback(*args, **kwargs):
        """Report intermediate metrics for pruning."""
        try:
            steps, vals = read_tensorboard(
                trial_number, pathlib.Path("./logs"), "reward"
            )
        except Exception:
            return None
        vals = objective(steps, vals)
        trial.report(vals, step=steps[-1])

        if trial.should_prune():
            raise optuna.TrialPruned()

    return run_case(
        trial_number,
        params,
        eval_function=objective,
        path_write=pathlib.Path("./configs"),
        path_read=pathlib.Path("./logs"),
        path_script=pathlib.Path("./run_script.sh"),
        read_metric="reward",
        callback=prune_callback,
    )


def individual_thread():
    """Run one optimization step in its own study worker."""
    study = optuna.create_study(
        study_name="Test",
        storage=JournalStorage(JournalFileBackend(file_path="./Test.log")),
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(run_one_case, n_trials=1)


def run_trials(total_cases: int, max_concurrent: int) -> None:
    """Run multiple trials with bounded concurrency."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(individual_thread) for _ in range(total_cases)]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                future.result()
                logger.info("Completed %d/%d trial submissions", idx, total_cases)
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
