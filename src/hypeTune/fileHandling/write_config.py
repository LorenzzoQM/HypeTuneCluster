import pathlib
import json
import os


def write_config(trial_number: int, params: dict, path: pathlib.Path) -> None:
    """Write trial parameters to a JSON config file."""
    os.makedirs(path, exist_ok=True)
    path = pathlib.Path(path) / f"trial_{trial_number}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
