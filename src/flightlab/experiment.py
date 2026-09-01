from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from flightlab.response import SISOResponseMetrics


@dataclass(frozen=True, slots=True, eq=False)
class ExperimentRun:
    run_id: str
    created_at: datetime
    method: str
    start_time: float
    end_time: float
    duration: float
    sample_count: int
    initial_state: np.ndarray
    system: Mapping[str, object]
    controller: Mapping[str, object]
    reference: Mapping[str, object]
    user_metadata: Mapping[str, object]
    metrics: SISOResponseMetrics

    def reproducibility_record(self):
        """Return a detached JSON-compatible record with scalar metrics."""
        metrics = {
            "final_output": float(self.metrics.final_output),
            "final_reference": float(self.metrics.final_reference),
            "steady_state_error": float(self.metrics.steady_state_error),
            "peak_output": float(self.metrics.peak_output),
            "maximum_absolute_tracking_error": float(
                self.metrics.maximum_absolute_tracking_error
            ),
            "rms_tracking_error": float(self.metrics.rms_tracking_error),
            "iae": float(self.metrics.iae),
            "ise": float(self.metrics.ise),
            "overshoot_percent": _optional_float(self.metrics.overshoot_percent),
            "settling_time": _optional_float(self.metrics.settling_time),
            "settling_tolerance": float(self.metrics.settling_tolerance),
        }

        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "method": self.method,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sample_count": self.sample_count,
            "initial_state": self.initial_state.tolist(),
            "system": _metadata_record(self.system),
            "controller": _metadata_record(self.controller),
            "reference": _metadata_record(self.reference),
            "user_metadata": _metadata_record(self.user_metadata),
            "metrics": metrics,
        }


def _optional_float(value):
    return None if value is None else float(value)


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


def _metadata_scalar(name, key, value):
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} metadata float for {key!r} must be finite")
        return value
    if isinstance(value, np.str_):
        return str(value)
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{name} metadata float for {key!r} must be finite")
        return value
    if type(value) is str:
        return value
    raise ValueError(
        f"{name} metadata value for {key!r} must be a supported simple value"
    )


def _metadata_value(name, key, value):
    if type(value) is tuple:
        return tuple(
            _metadata_scalar(name, f"{key}[{index}]", item)
            for index, item in enumerate(value)
        )
    return _metadata_scalar(name, key, value)


def _frozen_metadata(name, metadata):
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} metadata must be a mapping")

    items = list(metadata.items())
    if any(not isinstance(key, str) for key, _ in items):
        raise ValueError(f"{name} metadata keys must be strings")

    frozen = {
        str(key): _metadata_value(name, str(key), value)
        for key, value in sorted(items, key=lambda item: str(item[0]))
    }
    return MappingProxyType(frozen)


def _metadata_record(metadata):
    return {
        key: list(value) if type(value) is tuple else value
        for key, value in metadata.items()
    }


def _validated_run_id(run_id):
    if run_id is None:
        return str(uuid4())
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    return str(run_id)


def _validated_created_at(created_at):
    if created_at is None:
        return datetime.now(UTC)
    if not isinstance(created_at, datetime) or created_at.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    return created_at.astimezone(UTC)


def experiment_run(
    time,
    initial_state,
    metrics,
    method,
    system,
    controller,
    reference,
    *,
    user_metadata=None,
    run_id=None,
    created_at=None,
):
    """Build an immutable record for one already-completed SISO experiment."""
    time = _as_real_vector("time", time)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    if not np.all(time[1:] > time[:-1]):
        raise ValueError("time values must be strictly increasing")

    if not isinstance(metrics, SISOResponseMetrics):
        raise TypeError("metrics must be a SISOResponseMetrics")
    if not np.array_equal(time, metrics.time):
        raise ValueError("time must exactly match metrics.time")

    initial_state = _as_real_vector("initial_state", initial_state)
    initial_state.setflags(write=False)

    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    method = str(method)

    start_time = float(time[0])
    end_time = float(time[-1])
    with np.errstate(over="ignore"):
        duration = float(time[-1] - time[0])
    if not np.isfinite(duration):
        raise ValueError("time duration must be finite")

    if user_metadata is None:
        user_metadata = {}

    return ExperimentRun(
        run_id=_validated_run_id(run_id),
        created_at=_validated_created_at(created_at),
        method=method,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        sample_count=int(time.size),
        initial_state=initial_state,
        system=_frozen_metadata("system", system),
        controller=_frozen_metadata("controller", controller),
        reference=_frozen_metadata("reference", reference),
        user_metadata=_frozen_metadata("user", user_metadata),
        metrics=metrics,
    )
