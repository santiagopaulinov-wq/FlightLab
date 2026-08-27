import numpy as np
import pytest

from flightlab.state_space import StateSpace
from flightlab.trajectory import trajectory_extrema


def test_trajectory_extrema_for_multiple_components():
    result = trajectory_extrema(
        [0.0, 0.5, 1.0, 1.5],
        [[2.0, -1.0], [-3.0, 4.0], [1.0, 2.0], [0.0, -2.0]],
    )

    np.testing.assert_array_equal(result.minimum, [-3.0, -2.0])
    np.testing.assert_array_equal(result.minimum_time, [0.5, 1.5])
    np.testing.assert_array_equal(result.maximum, [2.0, 4.0])
    np.testing.assert_array_equal(result.maximum_time, [0.0, 0.5])


def test_trajectory_extrema_uses_first_occurrence():
    result = trajectory_extrema(
        [0.0, 1.0, 2.0, 3.0],
        [[-2.0, 5.0], [4.0, 1.0], [4.0, 1.0], [-2.0, 5.0]],
    )

    np.testing.assert_array_equal(result.minimum_time, [0.0, 1.0])
    np.testing.assert_array_equal(result.maximum_time, [1.0, 0.0])


def test_trajectory_extrema_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number of samples"):
        trajectory_extrema([0.0, 1.0], [[1.0], [2.0], [3.0]])


@pytest.mark.parametrize(
    ("time", "trajectory", "message"),
    [
        (0.0, [[1.0]], "time must be a non-empty 1D array"),
        ([], np.empty((0, 1)), "time must be a non-empty 1D array"),
        ([0.0], [1.0], "trajectory must have shape"),
        ([0.0], [[[1.0]]], "trajectory must have shape"),
        ([0.0], np.empty((1, 0)), "trajectory must have shape"),
    ],
)
def test_trajectory_extrema_rejects_invalid_dimensions(time, trajectory, message):
    with pytest.raises(ValueError, match=message):
        trajectory_extrema(time, trajectory)


@pytest.mark.parametrize(
    ("time", "trajectory", "message"),
    [
        ([0.0, np.inf], [[1.0], [2.0]], "time values must be finite"),
        ([0.0, 1.0], [[1.0], [np.nan]], "trajectory values must be finite"),
    ],
)
def test_trajectory_extrema_requires_finite_data(time, trajectory, message):
    with pytest.raises(ValueError, match=message):
        trajectory_extrema(time, trajectory)


def test_trajectory_extrema_accepts_state_space_simulation_result():
    system = StateSpace(
        [[0.0, 1.0], [-2.0, -3.0]],
        [[0.0], [1.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )
    time = np.array([0.0, 0.1, 0.2])
    states, outputs = system.simulate([1.0, 2.0], [3.0], time)

    state_extrema = trajectory_extrema(time, states)
    output_extrema = trajectory_extrema(time, outputs)

    assert state_extrema.minimum.shape == (system.n_states,)
    assert output_extrema.maximum.shape == (system.n_outputs,)
    np.testing.assert_array_equal(state_extrema.maximum_time, [0.2, 0.0])
    np.testing.assert_array_equal(output_extrema.maximum_time, [0.2])
