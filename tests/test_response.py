import numpy as np
import pytest

from flightlab.response import SISOResponseMetrics, response_metrics


def test_response_metrics_zero_tracking_error():
    metrics = response_metrics(
        [0.0, 0.5, 2.0],
        [0.0, 0.25, 1.0],
        [0.0, 0.25, 1.0],
    )

    assert isinstance(metrics, SISOResponseMetrics)
    np.testing.assert_array_equal(metrics.tracking_error, [0.0, 0.0, 0.0])
    assert metrics.final_output == pytest.approx(1.0)
    assert metrics.final_reference == pytest.approx(1.0)
    assert metrics.steady_state_error == pytest.approx(0.0)
    assert metrics.peak_output == pytest.approx(1.0)
    assert metrics.maximum_absolute_tracking_error == pytest.approx(0.0)
    assert metrics.rms_tracking_error == pytest.approx(0.0)
    assert metrics.iae == pytest.approx(0.0)
    assert metrics.ise == pytest.approx(0.0)
    assert metrics.overshoot_percent == pytest.approx(0.0)
    assert metrics.settling_time == pytest.approx(2.0)
    assert metrics.settling_tolerance == pytest.approx(0.02)


def test_response_metrics_constant_nonzero_error():
    metrics = response_metrics(
        [0.0, 1.0, 3.0],
        [1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0],
    )

    np.testing.assert_array_equal(metrics.tracking_error, [0.5, 0.5, 0.5])
    assert metrics.final_output == pytest.approx(1.5)
    assert metrics.final_reference == pytest.approx(2.0)
    assert metrics.steady_state_error == pytest.approx(0.5)
    assert metrics.peak_output == pytest.approx(1.5)
    assert metrics.maximum_absolute_tracking_error == pytest.approx(0.5)
    assert metrics.rms_tracking_error == pytest.approx(0.5)
    assert metrics.iae == pytest.approx(1.5)
    assert metrics.ise == pytest.approx(0.75)
    assert metrics.overshoot_percent == pytest.approx(0.0)
    assert metrics.settling_time is None


def test_response_metrics_uses_time_weighted_trapezoids_on_irregular_grid():
    metrics = response_metrics(
        [0.0, 0.5, 2.0, 3.0],
        [1.0, -1.0, 2.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )

    np.testing.assert_array_equal(metrics.tracking_error, [0.0, 2.0, -1.0, 1.0])
    assert metrics.iae == pytest.approx(3.75)
    assert metrics.ise == pytest.approx(5.75)
    assert metrics.rms_tracking_error == pytest.approx(np.sqrt(23.0 / 12.0))
    assert metrics.peak_output == pytest.approx(2.0)
    assert metrics.maximum_absolute_tracking_error == pytest.approx(2.0)


def test_response_metrics_copies_read_only_arrays_and_result_is_immutable():
    time = np.array([2.0, 4.0, 7.0])
    output = np.array([-3.0, -1.0, -2.0])
    reference = np.array([-2.0, -2.0, -2.0])

    metrics = response_metrics(time, output, reference)
    time[:] = 100.0
    output[:] = 100.0
    reference[:] = 100.0

    np.testing.assert_array_equal(metrics.time, [2.0, 4.0, 7.0])
    np.testing.assert_array_equal(metrics.output, [-3.0, -1.0, -2.0])
    np.testing.assert_array_equal(metrics.reference, [-2.0, -2.0, -2.0])
    np.testing.assert_array_equal(metrics.tracking_error, [1.0, -1.0, 0.0])
    assert metrics.peak_output == pytest.approx(-1.0)
    for values in (
        metrics.time,
        metrics.output,
        metrics.reference,
        metrics.tracking_error,
    ):
        assert values.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0
    with pytest.raises(AttributeError):
        metrics.final_output = 0.0


@pytest.mark.parametrize(
    ("reference", "output", "expected_overshoot"),
    [
        ([200.0, 200.0, 200.0, 100.0], [0.0, 80.0, 120.0, 100.0], 20.0),
        ([100.0, 100.0, 100.0], [0.0, 80.0, 100.0], 0.0),
        ([-100.0, -100.0, -100.0, -100.0], [0.0, -80.0, -125.0, -100.0], 25.0),
    ],
)
def test_response_metrics_overshoot_uses_final_reference_and_target_direction(
    reference,
    output,
    expected_overshoot,
):
    metrics = response_metrics(np.arange(len(output), dtype=float), output, reference)

    assert metrics.overshoot_percent == pytest.approx(expected_overshoot)


@pytest.mark.parametrize("final_reference", [0.0, 10.0 * np.finfo(float).eps])
def test_response_metrics_relative_metrics_are_undefined_for_near_zero_target(
    final_reference,
):
    metrics = response_metrics(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, final_reference],
        [final_reference, final_reference, final_reference],
    )

    assert metrics.overshoot_percent is None
    assert metrics.settling_time is None


@pytest.mark.parametrize(
    ("output", "expected_settling_time"),
    [
        ([0.0, 97.0, 98.5, 101.0, 100.0], 2.0),
        ([0.0, 99.0, 97.0, 101.0, 100.0], 3.0),
        ([0.0, 97.0, 99.0, 97.0], None),
    ],
)
def test_response_metrics_settling_time_uses_remaining_trajectory(
    output,
    expected_settling_time,
):
    time = np.arange(len(output), dtype=float)
    metrics = response_metrics(time, output, np.full(len(output), 100.0))

    if expected_settling_time is None:
        assert metrics.settling_time is None
    else:
        assert metrics.settling_time == pytest.approx(expected_settling_time)


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([[0.0, 1.0]], "time must be a 1D array"),
        ([], "time must contain at least two samples"),
        ([0.0], "time must contain at least two samples"),
    ],
)
def test_response_metrics_rejects_invalid_time_dimensions_and_length(time, message):
    with pytest.raises(ValueError, match=message):
        response_metrics(time, [0.0, 1.0], [0.0, 1.0])


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([0.0, np.inf], "time values must be finite"),
        ([0.0, 0.0], "time values must be strictly increasing"),
        ([1.0, 0.0], "time values must be strictly increasing"),
    ],
)
def test_response_metrics_rejects_invalid_time_values(time, message):
    with pytest.raises(ValueError, match=message):
        response_metrics(time, [0.0, 1.0], [0.0, 1.0])


@pytest.mark.parametrize(
    ("output", "reference", "message"),
    [
        ([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], "y must be a 1D array"),
        ([0.0, 1.0, 2.0], [[0.0], [1.0], [2.0]], "reference must be a 1D array"),
        (
            [0.0, 1.0],
            [0.0, 1.0, 2.0],
            "y must have the same number of samples as time",
        ),
        (
            [0.0, 1.0, 2.0],
            [0.0, 1.0],
            "reference must have the same number of samples as time",
        ),
    ],
)
def test_response_metrics_rejects_invalid_signal_shapes(output, reference, message):
    with pytest.raises(ValueError, match=message):
        response_metrics([0.0, 1.0, 2.0], output, reference)


@pytest.mark.parametrize(
    ("output", "reference", "message"),
    [
        ([0.0, np.nan], [0.0, 1.0], "y values must be finite"),
        ([0.0, 1.0], [0.0, np.inf], "reference values must be finite"),
    ],
)
def test_response_metrics_rejects_nonfinite_signal_values(output, reference, message):
    with pytest.raises(ValueError, match=message):
        response_metrics([0.0, 1.0], output, reference)


def test_response_metrics_rejects_invalid_settling_tolerances():
    invalid_tolerances = [0.0, -0.01, np.nan, np.inf, [0.02], 0.02 + 0.0j]

    for tolerance in invalid_tolerances:
        with pytest.raises((TypeError, ValueError), match="settling_tolerance"):
            response_metrics(
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                settling_tolerance=tolerance,
            )


def test_response_metrics_rejects_complex_inputs():
    cases = [
        ("time", [0.0 + 0.0j, 1.0 + 0.0j], [0.0, 1.0], [0.0, 1.0]),
        ("y", [0.0, 1.0], [0.0 + 0.0j, 1.0 + 0.0j], [0.0, 1.0]),
        ("reference", [0.0, 1.0], [0.0, 1.0], [0.0 + 0.0j, 1.0 + 0.0j]),
    ]

    for name, time, output, reference in cases:
        with pytest.raises(ValueError, match=name):
            response_metrics(time, output, reference)
