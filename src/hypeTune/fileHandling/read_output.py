import pathlib
import glob


def read_tensorboard(
    trial_number: int, path_read: pathlib.Path, value_parameter: str
) -> (list, list):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        raise ImportError(
            "TensorBoard is not installed. Please install it to read TensorBoard logs."
        )

    files = glob.glob(
        str(path_read / f"trial_{trial_number}" / "events.out.tfevents.*")
    )

    vec_steps = []
    vec_values = []

    for file in files:
        event_acc = EventAccumulator(file)
        event_acc.Reload()

        scalar_list = event_acc.Scalars(value_parameter)
        for scalar in scalar_list:
            vec_steps.append(scalar.step)
            vec_values.append(scalar.value)

    ordered_steps = sorted(vec_steps)
    ordered_values = [x for _, x in sorted(zip(vec_steps, vec_values))]
    return ordered_steps, ordered_values
