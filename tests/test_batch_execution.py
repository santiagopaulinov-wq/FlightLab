from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from flightlab.experiment import (
    ExperimentCase,
    ExperimentRun,
    SISOSimulationResult,
    execute_experiments,
)

_CREATED_AT = datetime(2026, 9, 2, 12, 30, tzinfo=UTC)


def _simulation_result(final_output=1.0):
    return SISOSimulationResult(
        time=np.array([0.0, 1.0, 2.0]),
        output=np.array([0.0, 0.75, final_output]),
        reference=np.array([1.0, 1.0, 1.0]),
    )


def _case(simulation=None, **overrides):
    if simulation is None:
        simulation = _simulation_result
    arguments = {
        "simulation": simulation,
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
    return ExperimentCase(**arguments)


def test_experiment_case_is_a_frozen_explicit_single_run_description():
    case = _case()

    assert not hasattr(case, "__dict__")
    assert case.simulation is _simulation_result
    assert case.initial_state == [0.25, -0.5]
    assert case.method == "exact"
    assert case.system == {"name": "demo", "order": 2}
    assert case.controller == {
        "type": "state_feedback",
        "gains": (2.0, 3.0),
    }
    assert case.reference == {"type": "step", "value": 1.0}
    assert case.user_metadata == {"seed": 7}
    assert case.settling_tolerance == pytest.approx(0.02)
    assert case.run_id == "run-001"
    assert case.created_at == _CREATED_AT
    with pytest.raises(FrozenInstanceError):
        case.method = "rk4"


def test_execute_experiments_executes_one_explicit_case():
    runs = execute_experiments([_case()])

    assert type(runs) is tuple
    assert len(runs) == 1
    assert isinstance(runs[0], ExperimentRun)
    assert runs[0].run_id == "run-001"
    assert runs[0].metrics.final_output == pytest.approx(1.0)


def test_execute_experiments_preserves_order_and_executes_each_case_once():
    call_order = []
    call_counts = {"first": 0, "second": 0, "third": 0}

    def simulation(name, final_output):
        def run():
            call_counts[name] += 1
            call_order.append(name)
            return _simulation_result(final_output)

        return run

    cases = (
        _case(
            simulation("first", 0.9),
            run_id="first",
            method="euler",
        ),
        _case(
            simulation("second", 1.0),
            run_id="second",
            method="rk4",
        ),
        _case(
            simulation("third", 1.1),
            run_id="third",
            method="exact",
        ),
    )

    runs = execute_experiments(cases)

    assert call_order == ["first", "second", "third"]
    assert call_counts == {"first": 1, "second": 1, "third": 1}
    assert tuple(run.run_id for run in runs) == ("first", "second", "third")
    assert tuple(run.method for run in runs) == ("euler", "rk4", "exact")
    assert tuple(run.metrics.final_output for run in runs) == pytest.approx(
        (0.9, 1.0, 1.1)
    )


def test_execute_experiments_returns_an_immutable_tuple():
    runs = execute_experiments((_case(),))

    with pytest.raises(TypeError):
        runs[0] = runs[0]
    assert not hasattr(runs, "append")


def test_execute_experiments_keeps_case_provenance_independent():
    first_state = np.array([1.0, 2.0])
    second_state = np.array([3.0])
    first_system = {"name": "first"}
    second_system = {"name": "second"}
    first_metadata = {"seed": 1}
    second_metadata = {"seed": 2}
    cases = [
        _case(
            initial_state=first_state,
            system=first_system,
            user_metadata=first_metadata,
            run_id="first",
        ),
        _case(
            initial_state=second_state,
            system=second_system,
            user_metadata=second_metadata,
            run_id="second",
        ),
    ]

    runs = execute_experiments(cases)

    np.testing.assert_array_equal(runs[0].initial_state, [1.0, 2.0])
    np.testing.assert_array_equal(runs[1].initial_state, [3.0])
    assert runs[0].system == {"name": "first"}
    assert runs[1].system == {"name": "second"}
    assert runs[0].user_metadata == {"seed": 1}
    assert runs[1].user_metadata == {"seed": 2}

    first_state[:] = 100.0
    second_state[:] = 100.0
    first_system["name"] = "changed"
    second_system["name"] = "changed"
    first_metadata["seed"] = 100
    second_metadata["seed"] = 100

    np.testing.assert_array_equal(runs[0].initial_state, [1.0, 2.0])
    np.testing.assert_array_equal(runs[1].initial_state, [3.0])
    assert runs[0].system == {"name": "first"}
    assert runs[1].system == {"name": "second"}
    assert runs[0].user_metadata == {"seed": 1}
    assert runs[1].user_metadata == {"seed": 2}


def test_execute_experiments_snapshots_the_case_collection_before_execution():
    cases = []
    calls = []

    def first_simulation():
        calls.append("first")
        cases.clear()
        return _simulation_result()

    def second_simulation():
        calls.append("second")
        return _simulation_result()

    cases.extend(
        [
            _case(first_simulation, run_id="first"),
            _case(second_simulation, run_id="second"),
        ]
    )

    runs = execute_experiments(cases)

    assert calls == ["first", "second"]
    assert tuple(run.run_id for run in runs) == ("first", "second")
    assert cases == []


def test_execute_experiments_propagates_single_run_validation_failures():
    calls = []

    def simulation(name):
        def run():
            calls.append(name)
            return _simulation_result()

        return run

    cases = (
        _case(simulation("first"), run_id="first"),
        _case(simulation("invalid"), method="", run_id="invalid"),
        _case(simulation("later"), run_id="later"),
    )

    with pytest.raises(ValueError, match="method must be a non-empty string"):
        execute_experiments(cases)

    assert calls == ["first", "invalid"]


def test_execute_experiments_propagates_sampled_result_validation_failures():
    calls = 0

    def simulation():
        nonlocal calls
        calls += 1
        return SISOSimulationResult(
            time=[0.0, 1.0],
            output=[0.0, np.nan],
            reference=[1.0, 1.0],
        )

    with pytest.raises(ValueError, match="y values must be finite"):
        execute_experiments([_case(simulation)])

    assert calls == 1


def test_execute_experiments_propagates_simulation_exceptions_unchanged():
    failure = RuntimeError("second simulation failed")
    calls = []

    def first_simulation():
        calls.append("first")
        return _simulation_result()

    def failed_simulation():
        calls.append("failed")
        raise failure

    def later_simulation():
        calls.append("later")
        return _simulation_result()

    cases = (
        _case(first_simulation, run_id="first"),
        _case(failed_simulation, run_id="failed"),
        _case(later_simulation, run_id="later"),
    )

    with pytest.raises(RuntimeError, match="second simulation failed") as caught:
        execute_experiments(cases)

    assert caught.value is failure
    assert calls == ["first", "failed"]


@pytest.mark.parametrize("cases", [None, 7, object()])
def test_execute_experiments_rejects_noniterable_batch_inputs(cases):
    with pytest.raises(TypeError, match="cases must be an iterable"):
        execute_experiments(cases)


@pytest.mark.parametrize("invalid_case", [None, {}, (), object()])
def test_execute_experiments_rejects_malformed_case_definitions_before_running(
    invalid_case,
):
    calls = 0

    def simulation():
        nonlocal calls
        calls += 1
        return _simulation_result()

    with pytest.raises(TypeError, match=r"cases\[1\] must be an ExperimentCase"):
        execute_experiments([_case(simulation), invalid_case])

    assert calls == 0


@pytest.mark.parametrize("cases", [(), [], iter(())])
def test_execute_experiments_accepts_an_empty_finite_collection(cases):
    result = execute_experiments(cases)

    assert result == ()
    assert type(result) is tuple


def test_execute_experiments_accepts_a_one_shot_ordered_iterable():
    cases = (
        _case(run_id=run_id, user_metadata={"order": order})
        for order, run_id in enumerate(("first", "second"))
    )

    runs = execute_experiments(cases)

    assert tuple(run.run_id for run in runs) == ("first", "second")
    assert tuple(run.user_metadata["order"] for run in runs) == (0, 1)
    assert tuple(cases) == ()


def test_execute_experiments_has_deterministic_explicit_records():
    cases = (
        _case(
            run_id="first",
            created_at=_CREATED_AT,
            system={"z": 3, "a": 1},
            user_metadata={"z": 3, "a": 1},
        ),
        _case(
            run_id="second",
            created_at=_CREATED_AT + timedelta(seconds=1),
            system={"m": 2, "a": 1},
            user_metadata={"m": 2, "a": 1},
        ),
    )
    reordered_metadata_cases = (
        replace(
            cases[0],
            system={"a": 1, "z": 3},
            user_metadata={"a": 1, "z": 3},
        ),
        replace(
            cases[1],
            system={"a": 1, "m": 2},
            user_metadata={"a": 1, "m": 2},
        ),
    )

    first_records = tuple(
        run.reproducibility_record() for run in execute_experiments(cases)
    )
    second_records = tuple(
        run.reproducibility_record()
        for run in execute_experiments(reordered_metadata_cases)
    )

    assert first_records == second_records
