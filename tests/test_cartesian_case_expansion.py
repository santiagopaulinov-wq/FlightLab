from datetime import UTC, datetime

import pytest

from flightlab.experiment import (
    ExperimentCase,
    SISOSimulationResult,
    expand_cartesian_experiment_cases,
)

_CREATED_AT = datetime(2026, 9, 4, 12, 30, tzinfo=UTC)


def _simulation():
    return SISOSimulationResult(
        time=[0.0, 1.0],
        output=[0.0, 1.0],
        reference=[1.0, 1.0],
    )


def _case(combination):
    label = "-".join(str(value) for value in combination) or "empty"
    return ExperimentCase(
        simulation=_simulation,
        initial_state=[0.0],
        method="exact",
        system={"combination": label},
        controller={"type": "none"},
        reference={"type": "step", "value": 1.0},
        user_metadata={"combination": label},
        run_id=f"case-{label}",
        created_at=_CREATED_AT,
    )


def test_expand_cartesian_experiment_cases_zero_axes_has_one_empty_combination():
    calls = []

    def factory(combination):
        calls.append(combination)
        return _case(combination)

    cases = expand_cartesian_experiment_cases([], factory)

    assert calls == [()]
    assert type(cases) is tuple
    assert tuple(case.run_id for case in cases) == ("case-empty",)


@pytest.mark.parametrize(
    "parameter_axes",
    [
        [[], [1, 2]],
        [[1, 2], []],
        [[1], iter(())],
    ],
)
def test_expand_cartesian_experiment_cases_empty_axis_has_no_combinations(
    parameter_axes,
):
    calls = 0

    def factory(combination):
        nonlocal calls
        calls += 1
        return _case(combination)

    cases = expand_cartesian_experiment_cases(parameter_axes, factory)

    assert cases == ()
    assert type(cases) is tuple
    assert calls == 0


def test_expand_cartesian_experiment_cases_preserves_one_axis_value_order():
    calls = []

    def factory(combination):
        calls.append(combination)
        return _case(combination)

    cases = expand_cartesian_experiment_cases([[3, 1, 2]], factory)

    assert calls == [(3,), (1,), (2,)]
    assert tuple(case.run_id for case in cases) == (
        "case-3",
        "case-1",
        "case-2",
    )


def test_expand_cartesian_experiment_cases_uses_deterministic_product_order():
    calls = []

    def factory(combination):
        calls.append(combination)
        return _case(combination)

    axes = (
        iter(("low", "high")),
        (2, 1),
        ["a", "b"],
    )
    cases = expand_cartesian_experiment_cases(axes, factory)
    expected = [
        ("low", 2, "a"),
        ("low", 2, "b"),
        ("low", 1, "a"),
        ("low", 1, "b"),
        ("high", 2, "a"),
        ("high", 2, "b"),
        ("high", 1, "a"),
        ("high", 1, "b"),
    ]

    assert calls == expected
    assert tuple(case.run_id for case in cases) == tuple(
        f"case-{'-'.join(str(value) for value in combination)}"
        for combination in expected
    )


def test_expand_cartesian_experiment_cases_returns_an_immutable_tuple():
    cases = expand_cartesian_experiment_cases([[1], [2]], _case)

    with pytest.raises(TypeError):
        cases[0] = cases[0]
    assert not hasattr(cases, "append")


def test_expand_cartesian_experiment_cases_does_not_execute_simulations():
    simulation_calls = 0

    def simulation():
        nonlocal simulation_calls
        simulation_calls += 1
        return _simulation()

    def factory(combination):
        case = _case(combination)
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

    cases = expand_cartesian_experiment_cases([[1, 2], [3, 4]], factory)

    assert len(cases) == 4
    assert simulation_calls == 0


@pytest.mark.parametrize("invalid_result", [None, {}, (), object()])
def test_expand_cartesian_experiment_cases_rejects_invalid_factory_results(
    invalid_result,
):
    calls = []

    def factory(combination):
        calls.append(combination)
        return _case(combination) if combination == (1, "a") else invalid_result

    with pytest.raises(
        TypeError,
        match=r"parameter_values\[1\] must be an ExperimentCase",
    ):
        expand_cartesian_experiment_cases([[1, 2], ["a", "b"]], factory)

    assert calls == [(1, "a"), (1, "b")]


def test_expand_cartesian_experiment_cases_propagates_factory_errors_unchanged():
    failure = RuntimeError("Cartesian factory failed")
    calls = []

    def factory(combination):
        calls.append(combination)
        if combination == (1, "b"):
            raise failure
        return _case(combination)

    with pytest.raises(RuntimeError, match="Cartesian factory failed") as caught:
        expand_cartesian_experiment_cases([[1, 2], ["a", "b"]], factory)

    assert caught.value is failure
    assert calls == [(1, "a"), (1, "b")]


@pytest.mark.parametrize("case_factory", [None, 7, object()])
def test_expand_cartesian_experiment_cases_rejects_noncallable_factories(
    case_factory,
):
    with pytest.raises(TypeError, match="case_factory must be callable"):
        expand_cartesian_experiment_cases([[1]], case_factory)


@pytest.mark.parametrize("parameter_axes", [None, 7, object()])
def test_expand_cartesian_experiment_cases_rejects_noniterable_axes(
    parameter_axes,
):
    with pytest.raises(TypeError, match="parameter_axes must be an iterable"):
        expand_cartesian_experiment_cases(parameter_axes, _case)


@pytest.mark.parametrize("invalid_axis", [None, 7, object()])
def test_expand_cartesian_experiment_cases_rejects_noniterable_axis_values(
    invalid_axis,
):
    with pytest.raises(
        TypeError,
        match=r"parameter_axes\[1\] must be an iterable",
    ):
        expand_cartesian_experiment_cases([[1], invalid_axis], _case)


def test_expand_cartesian_experiment_cases_snapshots_axes_before_factory_calls():
    first_axis = [1, 2]
    second_axis = ["a", "b"]

    def factory(combination):
        first_axis.clear()
        second_axis.clear()
        return _case(combination)

    cases = expand_cartesian_experiment_cases([first_axis, second_axis], factory)

    assert tuple(case.run_id for case in cases) == (
        "case-1-a",
        "case-1-b",
        "case-2-a",
        "case-2-b",
    )
    assert first_axis == []
    assert second_axis == []
