from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from flightlab.experiment import (
    ExperimentRun,
    SISOSimulationResult,
    execute_experiment,
)
from flightlab.response import SISOResponseMetrics, response_metrics

_CREATED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)


def _result():
    return SISOSimulationResult(
        time=np.array([2.0, 2.5, 4.0]),
        output=np.array([0.0, 0.75, 1.0]),
        reference=np.array([1.0, 1.0, 1.0]),
    )


def _execute(simulation=_result, **overrides):
    arguments = {
        "initial_state": [0.25, -0.5],
        "method": "exact",
        "system": {"name": "demo", "order": 2},
        "controller": {"type": "state_feedback", "gains": (2.0, 3.0)},
        "reference": {"type": "step", "value": 1.0},
        "user_metadata": {"seed": 7},
        "run_id": "run-001",
        "created_at": _CREATED_AT,
    }
    arguments.update(overrides)
    return execute_experiment(simulation, **arguments)


def test_siso_simulation_result_is_an_immutable_named_result():
    result = _result()

    assert result._fields == ("time", "output", "reference")
    with pytest.raises(AttributeError):
        result.output = np.array([1.0])


def test_execute_experiment_runs_one_simulation_and_constructs_one_run():
    calls = 0
    simulation_result = _result()

    def simulation():
        nonlocal calls
        calls += 1
        return simulation_result

    run = _execute(simulation)

    assert calls == 1
    assert isinstance(run, ExperimentRun)
    assert isinstance(run.metrics, SISOResponseMetrics)
    np.testing.assert_array_equal(run.metrics.time, simulation_result.time)
    np.testing.assert_array_equal(run.metrics.output, simulation_result.output)
    np.testing.assert_array_equal(run.metrics.reference, simulation_result.reference)
    np.testing.assert_array_equal(run.metrics.tracking_error, [1.0, 0.25, 0.0])
    np.testing.assert_array_equal(run.initial_state, [0.25, -0.5])
    assert run.method == "exact"
    assert run.metrics.settling_tolerance == pytest.approx(0.02)


def test_execute_experiment_reuses_response_metrics_with_custom_tolerance():
    simulation_result = _result()
    run = _execute(lambda: simulation_result, settling_tolerance=0.1)
    expected = response_metrics(
        simulation_result.time,
        simulation_result.output,
        simulation_result.reference,
        settling_tolerance=0.1,
    )

    for field in SISOResponseMetrics._fields[4:]:
        assert getattr(run.metrics, field) == getattr(expected, field)
    assert run.metrics.settling_tolerance == pytest.approx(0.1)


def test_execute_experiment_propagates_reproducibility_metadata():
    source_created_at = datetime(
        2026,
        9,
        1,
        6,
        30,
        tzinfo=timezone(timedelta(hours=-6)),
    )
    run = _execute(
        system={"z": 3, "a": "system"},
        controller={"z": 2, "a": "controller"},
        reference={"value": 1.0, "type": "step"},
        user_metadata={"seed": np.int64(7), "notes": None},
        run_id="explicit-id",
        created_at=source_created_at,
    )

    record = run.reproducibility_record()
    assert record["run_id"] == "explicit-id"
    assert record["created_at"] == "2026-09-01T12:30:00+00:00"
    assert record["initial_state"] == [0.25, -0.5]
    assert record["system"] == {"a": "system", "z": 3}
    assert record["controller"] == {"a": "controller", "z": 2}
    assert record["reference"] == {"type": "step", "value": 1.0}
    assert record["user_metadata"] == {"notes": None, "seed": 7}


def test_execute_experiment_snapshots_inputs_without_mutating_them():
    simulation_result = _result()
    initial_state = np.array([0.25, -0.5])
    system = {"name": "demo"}
    controller = {"type": "none"}
    reference = {"type": "step", "value": 1.0}
    user_metadata = {"seed": 7}
    expected_result = tuple(value.copy() for value in simulation_result)

    run = _execute(
        lambda: simulation_result,
        initial_state=initial_state,
        system=system,
        controller=controller,
        reference=reference,
        user_metadata=user_metadata,
    )

    for actual, expected in zip(simulation_result, expected_result, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(initial_state, [0.25, -0.5])
    assert system == {"name": "demo"}
    assert controller == {"type": "none"}
    assert reference == {"type": "step", "value": 1.0}
    assert user_metadata == {"seed": 7}

    initial_state[:] = 100.0
    simulation_result.time[:] = 100.0
    simulation_result.output[:] = 100.0
    simulation_result.reference[:] = 100.0
    system["name"] = "changed"
    controller["type"] = "changed"
    reference["value"] = 2.0
    user_metadata["seed"] = 99

    np.testing.assert_array_equal(run.initial_state, [0.25, -0.5])
    np.testing.assert_array_equal(run.metrics.time, [2.0, 2.5, 4.0])
    np.testing.assert_array_equal(run.metrics.output, [0.0, 0.75, 1.0])
    np.testing.assert_array_equal(run.metrics.reference, [1.0, 1.0, 1.0])
    assert run.system["name"] == "demo"
    assert run.controller["type"] == "none"
    assert run.reference["value"] == pytest.approx(1.0)
    assert run.user_metadata["seed"] == 7


def test_execute_experiment_is_reproducible_with_explicit_identity_and_timestamp():
    first = _execute(
        system={"z": 3, "a": 1},
        user_metadata={"z": 3, "a": 1},
    )
    second = _execute(
        system={"a": 1, "z": 3},
        user_metadata={"a": 1, "z": 3},
    )

    assert first.reproducibility_record() == second.reproducibility_record()


@pytest.mark.parametrize("simulation", [None, object(), 7])
def test_execute_experiment_rejects_noncallable_simulations(simulation):
    with pytest.raises(TypeError, match="simulation must be callable"):
        _execute(simulation)


@pytest.mark.parametrize(
    "result",
    [
        None,
        ([0.0, 1.0], [0.0, 1.0], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        object(),
    ],
)
def test_execute_experiment_rejects_an_invalid_simulation_result_type(result):
    with pytest.raises(TypeError, match="simulation must return a SISOSimulationResult"):
        _execute(lambda: result)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            SISOSimulationResult([0.0], [0.0], [1.0]),
            "time must contain at least two samples",
        ),
        (
            SISOSimulationResult([0.0, 1.0], [[0.0], [1.0]], [1.0, 1.0]),
            "y must be a 1D array",
        ),
        (
            SISOSimulationResult([0.0, 1.0], [0.0, np.nan], [1.0, 1.0]),
            "y values must be finite",
        ),
        (
            SISOSimulationResult([0.0, 1.0], [0.0], [1.0, 1.0]),
            "y must have the same number of samples as time",
        ),
        (
            SISOSimulationResult([0.0, 1.0], [0.0, 1.0], [1.0]),
            "reference must have the same number of samples as time",
        ),
    ],
)
def test_execute_experiment_delegates_invalid_sampled_results_to_metrics(
    result,
    message,
):
    with pytest.raises(ValueError, match=message):
        _execute(lambda: result)


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"initial_state": [0.0, np.inf]}, ValueError, "initial_state.*finite"),
        ({"method": ""}, ValueError, "method must be a non-empty string"),
        ({"system": None}, TypeError, "system metadata must be a mapping"),
        ({"user_metadata": {"nested": []}}, ValueError, "user metadata value"),
        (
            {"settling_tolerance": 0.0},
            ValueError,
            "settling_tolerance must be a finite positive scalar",
        ),
        ({"run_id": ""}, ValueError, "run_id must be a non-empty string"),
        (
            {"created_at": _CREATED_AT.replace(tzinfo=None)},
            ValueError,
            "created_at must be a timezone-aware datetime",
        ),
    ],
)
def test_execute_experiment_delegates_invalid_run_inputs(
    overrides,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        _execute(**overrides)


def test_execute_experiment_propagates_simulation_failures_unchanged():
    failure = RuntimeError("simulation failed")

    def simulation():
        raise failure

    with pytest.raises(RuntimeError, match="simulation failed") as caught:
        _execute(simulation)

    assert caught.value is failure
