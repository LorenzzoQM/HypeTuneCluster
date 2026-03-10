# HypeTuneCluster

## Overview

HypeTuneCluster is a lightweight toolkit for running hyperparameter tuning workflows
on local machines and Slurm-based HPC clusters. It handles the repetitive parts of a
trial loop:

- write one JSON config per trial
- launch the training job locally or through `sbatch`
- monitor job completion
- read TensorBoard scalars back into the tuning loop

The core entry point is `hypeTune.run.run.run_case()`, which is used by the Optuna
and Population Based Training examples in this repository.

## Installation

The package uses a standard `src/` layout and can be installed in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The library code itself declares a minimal dependency set. The bundled examples also
expect a few runtime packages:

```bash
pip install numpy optuna tensorboard
```


## Quick start

For a minimal local Optuna tuning loop, use:

```bash
python examples/run_optuna.py --n-cases 4 --max-concurrent 2
```

That script samples parameters, writes configs into `./configs`, launches one worker
script per trial, waits for completion, then reads the `reward` TensorBoard scalar
from `./logs`.

For Population Based Training, use:

```bash
python examples/run_PBT.py --n-cases 20 --max-concurrent 4
```

That driver persists PBT state in `./PBT_out.json`, reuses it on the next run, and
passes parent-trial information through the generated trial config.

At a high level, the local workflow looks like this:

```python
from pathlib import Path
from hypeTune.run.run import run_case


def objective(steps, values):
    return sum(values) / len(values)


score = run_case(
    trial_number=0,
    params={"learning_rate": 1e-3, "width": 32},
    eval_function=objective,
    path_write=Path("./configs"),
    path_read=Path("./logs"),
    path_script=Path("./run_script.py"),
    path_venv=Path("./.venv"),
    read_metric="reward",
)
```

Use `path_venv` for local execution. Omit `path_venv` to submit through Slurm instead.

## Example config

Each trial is written as `trial_<n>.json`. A typical generated config looks like:

```json
{
  "batch_size": 512,
  "learning_rate": 0.001,
  "gamma": 0.998
}
```

Your training script is responsible for reading that file, running the trial, and
writing TensorBoard event files under a directory like `logs/trial_<n>/...`.

## Running on the cluster

Cluster execution is selected by leaving `path_venv=None` in `run_case()`. In that
mode, HypeTuneCluster calls:

```bash
sbatch <path_to_submission_script> <trial_number>
```

and then polls job state with:

```bash
sacct -j <job_id> -X -n -o State
```

The shipped example drivers are:

```bash
python examples/run_optuna.py --n-cases 3 --max-concurrent 3
python examples/run_PBT.py --n-cases 20 --max-concurrent 4
```

Before running on a cluster, update the hard-coded paths in the example scripts to
match your environment:

- config output directory
- TensorBoard log directory
- training script path
- Optuna journal path

Your batch script must accept the trial number as a positional argument and pass it
through to the Python training script that consumes `trial_<n>.json`.

## Resuming / monitoring jobs

Monitoring is built into `run_case()`:

- local jobs are started with `subprocess.Popen(...)` and polled by PID
- Slurm jobs are polled with `sacct`
- an optional `callback(job_id, elapsed_seconds)` can run on each polling cycle

For Optuna-based searches, resuming usually comes from reusing the same study storage.
The examples already do this with `JournalStorage` and `load_if_exists=True`, so
rerunning the driver continues the existing study instead of starting from scratch.

There is not yet a generic repository-level checkpoint/resume layer for arbitrary
trial processes. Resume behavior inside an individual training job is handled by your
training script. The shipped PBT example resumes the search state itself by reading
and writing `./PBT_out.json`, and it passes the parent trial identifier in the
generated parameters.

To inspect results while jobs are running:

- read the Optuna journal file such as `Test.log`
- inspect generated configs under `configs/`
- point TensorBoard at the corresponding log directory
- check Slurm state with `sacct` or `squeue`
