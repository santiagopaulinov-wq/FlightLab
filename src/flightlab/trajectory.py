from typing import NamedTuple

import numpy as np


class TrajectoryExtrema(NamedTuple):
    minimum: np.ndarray
    minimum_time: np.ndarray
    maximum: np.ndarray
    maximum_time: np.ndarray


def trajectory_extrema(time, trajectory):
    """Return component-wise extrema and their first occurrence times."""
    time = np.asarray(time, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)

    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a non-empty 1D array")
    if trajectory.ndim != 2 or trajectory.shape[1] == 0:
        raise ValueError("trajectory must have shape (n_time_samples, n_components)")
    if trajectory.shape[0] != time.size:
        raise ValueError("time and trajectory must have the same number of samples")
    if not np.all(np.isfinite(time)):
        raise ValueError("time values must be finite")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory values must be finite")

    minimum_indices = np.argmin(trajectory, axis=0)
    maximum_indices = np.argmax(trajectory, axis=0)
    component_indices = np.arange(trajectory.shape[1])

    return TrajectoryExtrema(
        minimum=trajectory[minimum_indices, component_indices],
        minimum_time=time[minimum_indices],
        maximum=trajectory[maximum_indices, component_indices],
        maximum_time=time[maximum_indices],
    )
