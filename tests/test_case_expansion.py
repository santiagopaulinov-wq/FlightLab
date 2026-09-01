from datetime import UTC, datetime

import pytest

from flightlab.experiment import (
    ExperimentCase,
    SISOSimulationResult,
    expand_experiment_cases,
)

_CREATED_AT = datetime(2026, 9, 3, 12, 30, tzinfo=UTC)


def _simulation():
    return SISOSimulationResult(
        time=[0.0, 1.0],
        output=[0.0, 1.0],
        reference=[1.0, 1.0],
    )


def _case(value):
    return ExperimentCase(
        simulation=_simulation,
        initial_state=[float(value)],
        method="exact",
        system={"parameter": value},
        controller={"type": "none"},
        reference={"type": "step", "value": 1.0},
        user_metadata={"parameter": value},
        run_id=f"case-{value}",
        created_at=_CREATED_AT,
    )


@pytest.mark.parametrize("parameter_values", [(), [], iter(())])
def test_expand_experiment_cases_returns_empty_tuple_for_empty_input(
    parameter_values,
):
    calls = 0

    def factory(value):
        nonlocal calls
        calls += 1
        return _case(value)

    cases = expand_experiment_cases(parameter_values, factory)

    assert cases == ()
    assert type(cases) is tuple
    assert calls == 0


def test_expand_experiment_cases_maps_one_explicit_value():
    cases = expand_experiment_cases([3], _case)

    assert type(cases) is tuple
    assert len(cases) == 1
    assert isinstance(cases[0], ExperimentCase)
    assert cases[0].run_id == "case-3"
    assert cases[0].system == {"parameter": 3}


def test_expand_experiment_cases_preserves_order_and_calls_factory_once_per_value():
    calls = []

    def factory(value):
        calls.append(value)
        return _case(value)

    cases = expand_experiment_cases((3, 1, 2), factory)

    assert calls == [3, 1, 2]
    assert tuple(case.run_id for case in cases) == (
        "case-3",
        "case-1",
        "case-2",
    )
    assert tuple(case.system["parameter"] for case in cases) == (3, 1, 2)


def test_expand_experiment_cases_returns_an_immutable_tuple():
    cases = expand_experiment_cases([1], _case)

    with pytest.raises(TypeError):
        cases[0] = cases[0]
    assert not hasattr(cases, "append")


def test_expand_experiment_cases_does_not_execute_generated_cases():
    simulation_calls = 0

    def simulation():
        nonlocal simulation_calls
        simulation_calls += 1
        return _simulation()

    def factory(value):
        case = _case(value)
        return ExperimentCase(
            simulation=simulation,
            initial_state=case.initial_state,
            method=case.method,
            system=case.system,
            controller=case.controller,
            reference=case.reference,
            user_metadata=case.user_metadata,
            run_id=case.run_id,
            created_at=case.created_at,
        )

    cases = expand_experiment_cases([1, 2], factory)

    assert len(cases) == 2
    assert simulation_calls == 0


@pytest.mark.parametrize("invalid_result", [None, {}, (), object()])
def test_expand_experiment_cases_rejects_invalid_factory_results(
    invalid_result,
):
    calls = []

    def factory(value):
        calls.append(value)
        return _case(value) if value == 1 else invalid_result

    with pytest.raises(
        TypeError,
        match=r"parameter_values\[1\] must be an ExperimentCase",
    ):
        expand_experiment_cases([1, 2, 3], factory)

    assert calls == [1, 2]


def test_expand_experiment_cases_propagates_factory_exceptions_unchanged():
    failure = RuntimeError("factory failed")
    calls = []

    def factory(value):
        calls.append(value)
        if value == 2:
            raise failure
        return _case(value)

    with pytest.raises(RuntimeError, match="factory failed") as caught:
        expand_experiment_cases([1, 2, 3], factory)

    assert caught.value is failure
    assert calls == [1, 2]


@pytest.mark.parametrize("case_factory", [None, 7, object()])
def test_expand_experiment_cases_rejects_noncallable_factories(case_factory):
    with pytest.raises(TypeError, match="case_factory must be callable"):
        expand_experiment_cases([1], case_factory)


@pytest.mark.parametrize("parameter_values", [None, 7, object()])
def test_expand_experiment_cases_rejects_noniterable_parameter_values(
    parameter_values,
):
    with pytest.raises(TypeError, match="parameter_values must be an iterable"):
        expand_experiment_cases(parameter_values, _case)


def test_expand_experiment_cases_snapshots_parameter_order_before_factory_calls():
    parameter_values = [1, 2]

    def factory(value):
        parameter_values.clear()
        return _case(value)

    cases = expand_experiment_cases(parameter_values, factory)

    assert tuple(case.run_id for case in cases) == ("case-1", "case-2")
    assert parameter_values == []
