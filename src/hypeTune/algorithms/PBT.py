from dataclasses import dataclass
import numpy as np
from typing import Callable
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class Hyperparameter:
    name: str
    values: tuple[float, float]
    log_space: bool = False
    distribution: None | Callable = None
    perturbation_range: tuple[float, float] = (0.8, 1.2)


@dataclass
class Trial:
    trial_number: int
    hyperparameters: dict[str, float]
    performance: float | None = None
    parent_trial: int | None = None
    finished: bool = False
    exploited: bool = False
    start_step: int | None = None
    start_time: float | None = None
    end_step: int | None = None
    end_time: float | None = None


class PBT:

    def __init__(
        self,
        hyperparameters: list[Hyperparameter],
        population_size: int,
        upper_percentile: float,
        lower_percentile: float,
        total_trials: int | None = None,
        always_perturb: bool = False,
    ):

        self.hyperparameters = hyperparameters
        self.population_size = population_size
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.total_trials = total_trials
        self.always_perturb = always_perturb

        self._initialize()

    def _initialize(self):
        self.set_of_trials = dict()
        self.set_of_ongoing_trials = set()

    def _sample_hyperparameters(self) -> dict[str, float]:
        hype_params = dict()
        for hyper in self.hyperparameters:
            if hyper.distribution is not None:
                hype_params[hyper.name] = hyper.distribution()
            else:
                if hyper.log_space:
                    # TODO: implement log space sampling (and inverse log space sampling for mutation)
                    pass
                else:
                    hype_params[hyper.name] = np.random.uniform(
                        hyper.values[0], hyper.values[1]
                    )

        return hype_params

    def _perturb(self, trial_hyperparameter: dict[str, float]) -> dict[str, float]:
        hyper_prior = trial_hyperparameter.copy()
        hyper_perturbed = dict()
        for hyper in self.hyperparameters:
            hyper_value = hyper_prior[hyper.name]
            perturb_range = hyper.perturbation_range
            if hyper.log_space:
                pass
            else:
                decrease = np.random.normal() > 0.5
                if decrease:
                    hyper_perturbed[hyper.name] = hyper_value * perturb_range[0]
                else:
                    hyper_perturbed[hyper.name] = hyper_value * perturb_range[1]

        return hyper_perturbed

    def _exploit(self, trial: Trial) -> int | None:
        """
        Exploit the best trial by copying its hyperparameters to the worst trial.

        trial: trial to check if it is in the lower percentile and needs to be exploited
        """

        if trial.finished:
            performance = trial.performance
            all_performances = [
                [t.performance, t.trial_number]
                for t in self.set_of_trials.values()
                if t.finished
            ]
            performances = [p[0] for p in all_performances]
            trials = [p[1] for p in all_performances]

            upper_threshold = np.percentile(performances, self.upper_percentile)
            lower_threshold = np.percentile(performances, self.lower_percentile)

            if performance < lower_threshold:
                indexes = np.where(np.array(performances) > upper_threshold)[0]
                trial_to_copy = np.random.choice(indexes)
                trial_to_copy_number = trials[trial_to_copy]
                return trial_to_copy_number

            else:
                return trial.trial_number

        else:
            logger.warning(
                f"Trial {trial.trial_number} is not finished yet. Cannot exploit."
            )

        return None

    def get_initial_trials(self) -> list[Trial]:
        for i in range(self.population_size):
            hype_params = self._sample_hyperparameters()
            trial_i = Trial(
                trial_number=i,
                hyperparameters=hype_params,
                start_step=0,
                start_time=time.time(),
            )
            self.set_of_trials[i] = trial_i
            self.set_of_ongoing_trials.add(i)

        return self.get_ongoing_trials()

    def update_trial(
        self,
        trial_number: int,
        performance: float,
        end_step: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        trial = self.set_of_trials[trial_number]
        trial.performance = performance
        trial.finished = True
        trial.end_step = end_step
        if start_time is not None:
            trial.start_time = start_time
        trial.end_time = end_time

        if (
            self.total_trials is None
            or len(self.set_of_trials.keys()) < self.total_trials
        ):
            trial_to_exploit = self._exploit(trial)
            exploited = trial_to_exploit != trial_number
            if trial_to_exploit == trial_number and not self.always_perturb:
                new_hyperparameters = self.set_of_trials[
                    trial_number
                ].hyperparameters.copy()
            else:
                new_hyperparameters = self._perturb(
                    self.set_of_trials[trial_to_exploit].hyperparameters
                )
            new_trial_number = len(self.set_of_trials.keys())
            new_trial = Trial(
                trial_number=new_trial_number,
                hyperparameters=new_hyperparameters,
                performance=0.0,
                parent_trial=trial_to_exploit,
                exploited=exploited,
                start_step=end_step,
                start_time=end_time,
            )
            self.set_of_trials[new_trial_number] = new_trial
            self.set_of_ongoing_trials.add(new_trial_number)

        self.set_of_ongoing_trials.remove(trial_number)

        if len(self.set_of_ongoing_trials) != self.population_size:
            logger.warning(
                f"Number of ongoing trials {len(self.set_of_ongoing_trials)} is not equal to population size {self.population_size}."
            )

    def get_ongoing_trials(self) -> list[Trial]:
        return [self.set_of_trials[i] for i in self.set_of_ongoing_trials]

    def get_finished_trials(self) -> list[Trial]:
        return [t for t in self.set_of_trials.values() if t.finished]

    def get_trial_results(self) -> list[tuple[float, int]]:
        return [
            (t.performance, t.trial_number)
            for t in self.set_of_trials.values()
            if t.finished
        ]

    def to_json(self, path: Path):
        import json

        with open(path, "w") as f:
            json.dump(
                {
                    "hyperparameters": [
                        {
                            "name": hyper.name,
                            "values": hyper.values,
                            "log_space": hyper.log_space,
                            "perturbation_range": hyper.perturbation_range,
                        }
                        for hyper in self.hyperparameters
                    ],
                    "population_size": self.population_size,
                    "upper_percentile": self.upper_percentile,
                    "lower_percentile": self.lower_percentile,
                    "total_trials": self.total_trials,
                    "trials": [
                        {
                            "trial_number": trial.trial_number,
                            "hyperparameters": trial.hyperparameters,
                            "performance": trial.performance,
                            "parent_trial": trial.parent_trial,
                            "finished": trial.finished,
                        }
                        for trial in self.set_of_trials.values()
                    ],
                },
                f,
                indent=4,
            )

    def from_json(self, path: Path):
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self.hyperparameters = [
            Hyperparameter(
                name=hyper["name"],
                values=tuple(hyper["values"]),
                log_space=hyper["log_space"],
                perturbation_range=tuple(hyper["perturbation_range"]),
            )
            for hyper in data["hyperparameters"]
        ]
        self.population_size = data["population_size"]
        self.upper_percentile = data["upper_percentile"]
        self.lower_percentile = data["lower_percentile"]
        self.total_trials = data["total_trials"]

        self.set_of_trials = dict()
        for trial_data in data["trials"]:
            trial_i = Trial(
                trial_number=trial_data["trial_number"],
                hyperparameters=trial_data["hyperparameters"],
                performance=trial_data["performance"],
                parent_trial=trial_data["parent_trial"],
                finished=trial_data["finished"],
            )
            self.set_of_trials[trial_i.trial_number] = trial_i

        self.set_of_ongoing_trials = set(
            [t.trial_number for t in self.set_of_trials.values() if not t.finished]
        )
