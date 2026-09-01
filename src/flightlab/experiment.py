from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from types import MappingProxyType
from typing import NamedTuple
from uuid import uuid4

import numpy as np

from flightlab.response import SISOResponseMetrics, response_metrics


class SISOSimulationResult(NamedTuple):
    """Sampled result returned by one caller-supplied SISO simulation."""

    time: object
    output: object
    reference: object


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class ExperimentCase:
    """Describe one explicit call to ``execute_experiment``."""

    simulation: Callable[[], SISOSimulationResult]
    initial_state: object
    method: str
    system: Mapping[str, object]
    controller: Mapping[str, object]
    reference: Mapping[str, object]
    user_metadata: Mapping[str, object] | None = None
    settling_tolerance: float = 0.02
    run_id: str | None = None
    created_at: datetime | None = None


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


def execute_experiment(
    simulation,
    initial_state,
    method,
    system,
    controller,
    reference,
    *,
    user_metadata=None,
    settling_tolerance=0.02,
    run_id=None,
    created_at=None,
):
    """Execute one zero-argument SISO simulation and return one experiment run.

    The callable is invoked exactly once and must return a
    ``SISOSimulationResult``. Its sampled reference trajectory is distinct from
    the ``reference`` provenance mapping recorded on the returned run.
    """
    if not callable(simulation):
        raise TypeError("simulation must be callable")

    result = simulation()
    if not isinstance(result, SISOSimulationResult):
        raise TypeError("simulation must return a SISOSimulationResult")

    metrics = response_metrics(
        result.time,
        result.output,
        result.reference,
        settling_tolerance=settling_tolerance,
    )
    return experiment_run(
        time=result.time,
        initial_state=initial_state,
        metrics=metrics,
        method=method,
        system=system,
        controller=controller,
        reference=reference,
        user_metadata=user_metadata,
        run_id=run_id,
        created_at=created_at,
    )


def execute_experiments(
    cases: Iterable[ExperimentCase],
) -> tuple[ExperimentRun, ...]:
    """Execute an ordered finite collection of explicit cases sequentially.

    The collection is snapshotted before execution. Each case delegates to
    ``execute_experiment`` in caller order. Execution stops at the first raised
    exception; completed earlier calls are not rolled back and later cases are
    not invoked.
    """
    try:
        case_iterator = iter(cases)
    except TypeError as error:
        raise TypeError(
            "cases must be an iterable of ExperimentCase objects"
        ) from error
    cases = tuple(case_iterator)

    for index, case in enumerate(cases):
        if not isinstance(case, ExperimentCase):
            raise TypeError(f"cases[{index}] must be an ExperimentCase")

    return tuple(
        execute_experiment(
            case.simulation,
            initial_state=case.initial_state,
            method=case.method,
            system=case.system,
            controller=case.controller,
            reference=case.reference,
            user_metadata=case.user_metadata,
            settling_tolerance=case.settling_tolerance,
            run_id=case.run_id,
            created_at=case.created_at,
        )
        for case in cases
    )


def expand_experiment_cases(
    parameter_values: Iterable[object],
    case_factory: Callable[[object], ExperimentCase],
) -> tuple[ExperimentCase, ...]:
    """Map explicit ordered parameter values to immutable experiment cases.

    The values are snapshotted before the factory is invoked exactly once for
    each value in caller order. Factory exceptions propagate unchanged, and no
    simulation is executed.
    """
    if not callable(case_factory):
        raise TypeError("case_factory must be callable")

    try:
        value_iterator = iter(parameter_values)
    except TypeError as error:
        raise TypeError("parameter_values must be an iterable") from error
    parameter_values = tuple(value_iterator)

    cases = []
    for index, value in enumerate(parameter_values):
        case = case_factory(value)
        if not isinstance(case, ExperimentCase):
            raise TypeError(
                "case_factory result for "
                f"parameter_values[{index}] must be an ExperimentCase"
            )
        cases.append(case)

    return tuple(cases)


def expand_cartesian_experiment_cases(
    parameter_axes: Iterable[Iterable[object]],
    case_factory: Callable[[tuple[object, ...]], ExperimentCase],
) -> tuple[ExperimentCase, ...]:
    """Expand explicit ordered parameter axes into experiment cases.

    Combinations follow ``itertools.product`` order, with the rightmost axis
    varying fastest. Zero axes produce one empty combination; any empty axis
    produces no combinations. No generated simulation is executed.
    """
    if not callable(case_factory):
        raise TypeError("case_factory must be callable")

    try:
        axes_iterator = iter(parameter_axes)
    except TypeError as error:
        raise TypeError("parameter_axes must be an iterable") from error

    axes = []
    for index, axis in enumerate(axes_iterator):
        try:
            axis_iterator = iter(axis)
        except TypeError as error:
            raise TypeError(f"parameter_axes[{index}] must be an iterable") from error
        axes.append(tuple(axis_iterator))

    return expand_experiment_cases(product(*axes), case_factory)


def execute_cartesian_experiments(
    parameter_axes: Iterable[Iterable[object]],
    case_factory: Callable[[tuple[object, ...]], ExperimentCase],
) -> tuple[ExperimentRun, ...]:
    """Expand and sequentially execute one explicit Cartesian case set."""
    return execute_experiments(
        expand_cartesian_experiment_cases(parameter_axes, case_factory)
    )
