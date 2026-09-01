from typing import NamedTuple

import numpy as np

_DEFAULT_SETTLING_TOLERANCE = 0.02
_FINAL_REFERENCE_ATOL = 100.0 * np.finfo(float).eps


class SISOResponseMetrics(NamedTuple):
    time: np.ndarray
    output: np.ndarray
    reference: np.ndarray
    tracking_error: np.ndarray
    final_output: float
    final_reference: float
    steady_state_error: float
    peak_output: float
    maximum_absolute_tracking_error: float
    rms_tracking_error: float
    iae: float
    ise: float
    overshoot_percent: float | None
    settling_time: float | None
    settling_tolerance: float


def _as_real_vector(name, values):
    try:
        raw_values = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric 1D array") from error

    if np.iscomplexobj(raw_values):
        raise ValueError(f"{name} values must be real")

    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric 1D array") from error

    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} values must be finite")

    return np.array(vector, dtype=float, copy=True)


def _validated_settling_tolerance(settling_tolerance):
    try:
        raw_tolerance = np.asarray(settling_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("settling_tolerance must be a finite positive scalar") from error

    if raw_tolerance.ndim != 0 or np.iscomplexobj(raw_tolerance):
        raise ValueError("settling_tolerance must be a finite positive scalar")

    try:
        tolerance = float(raw_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("settling_tolerance must be a finite positive scalar") from error

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("settling_tolerance must be a finite positive scalar")

    return tolerance


def _readonly(array):
    array.setflags(write=False)
    return array


def response_metrics(
    time,
    y,
    reference,
    *,
    settling_tolerance=_DEFAULT_SETTLING_TOLERANCE,
):
    """Evaluate a finite sampled SISO output and reference trajectory.

    Tracking error is ``reference - y``. IAE and ISE use trapezoidal time
    integration, and RMS error is ``sqrt(ISE / (time[-1] - time[0]))``.

    Overshoot is the percentage by which the furthest output sample in the
    final-reference direction exceeds the final reference magnitude. Settling
    time is the earliest sampled time whose entire remaining output trajectory
    lies in an inclusive relative band around the final reference; it does not
    interpolate between samples. Both relative quantities are undefined and
    returned as ``None`` when the final reference magnitude is no larger than
    ``100 * machine epsilon``.
    """
    time = _as_real_vector("time", time)
    output = _as_real_vector("y", y)
    reference = _as_real_vector("reference", reference)
    settling_tolerance = _validated_settling_tolerance(settling_tolerance)

    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    if not np.all(time[1:] > time[:-1]):
        raise ValueError("time values must be strictly increasing")
    if output.size != time.size:
        raise ValueError("y must have the same number of samples as time")
    if reference.size != time.size:
        raise ValueError("reference must have the same number of samples as time")

    with np.errstate(invalid="ignore", over="ignore"):
        tracking_error = reference - output
        absolute_error = np.abs(tracking_error)
        squared_error = tracking_error**2
        duration = time[-1] - time[0]
        iae = float(np.trapezoid(absolute_error, x=time))
        ise = float(np.trapezoid(squared_error, x=time))
        rms_tracking_error = float(np.sqrt(ise / duration))

    if not np.all(np.isfinite(tracking_error)):
        raise ValueError("tracking error values must be finite")
    if not np.all(np.isfinite([duration, iae, ise, rms_tracking_error])):
        raise ValueError("computed response metrics must be finite")

    final_output = float(output[-1])
    final_reference = float(reference[-1])
    steady_state_error = float(tracking_error[-1])
    peak_output = float(np.max(output))
    maximum_absolute_tracking_error = float(np.max(absolute_error))

    if abs(final_reference) <= _FINAL_REFERENCE_ATOL:
        overshoot_percent = None
        settling_time = None
    else:
        target_magnitude = abs(final_reference)
        target_direction = np.copysign(1.0, final_reference)
        directional_peak = float(np.max(target_direction * output))
        overshoot_percent = 100.0 * max(
            0.0,
            (directional_peak - target_magnitude) / target_magnitude,
        )

        settling_band = settling_tolerance * target_magnitude
        inside_band = np.abs(output - final_reference) <= settling_band
        remains_inside = np.logical_and.accumulate(inside_band[::-1])[::-1]
        settling_indices = np.flatnonzero(remains_inside)
        settling_time = (
            float(time[settling_indices[0]]) if settling_indices.size else None
        )

    return SISOResponseMetrics(
        time=_readonly(time),
        output=_readonly(output),
        reference=_readonly(reference),
        tracking_error=_readonly(tracking_error),
        final_output=final_output,
        final_reference=final_reference,
        steady_state_error=steady_state_error,
        peak_output=peak_output,
        maximum_absolute_tracking_error=maximum_absolute_tracking_error,
        rms_tracking_error=rms_tracking_error,
        iae=iae,
        ise=ise,
        overshoot_percent=overshoot_percent,
        settling_time=settling_time,
        settling_tolerance=settling_tolerance,
    )
