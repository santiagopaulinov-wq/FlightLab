from datetime import UTC, datetime

import pytest

from flightlab.experiment import (
    ExperimentCase,
    ExperimentRun,
    SISOSimulationResult,
    execute_cartesian_experiments,
)

_CREATED_AT = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)


def _label(combination):
    return "-".join(str(value) for value in combination) or "empty"


def _case(combination, *, simulation=None, **overrides):
    label = _label(combination)
    if simulation is None:

        def simulation():
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

    arguments = {
        "simulation": simulation,
        "initial_state": [0.0],
        "method": "exact",
        "system": {"combination": label},
        "controller": {"type": "none"},
        "reference": {"type": "step", "value": 1.0},
        "user_metadata": {"combination": combination},
        "run_id": f"case-{label}",
        "created_at": _CREATED_AT,
    }
    arguments.update(overrides)
    return ExperimentCase(**arguments)


def test_execute_cartesian_experiments_zero_axes_executes_empty_combination_once():
    factory_calls = []
    simulation_calls = []

    def factory(combination):
        factory_calls.append(combination)

        def simulation():
            simulation_calls.append(combination)
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    runs = execute_cartesian_experiments([], factory)

    assert factory_calls == [()]
    assert simulation_calls == [()]
    assert type(runs) is tuple
    assert tuple(run.run_id for run in runs) == ("case-empty",)


@pytest.mark.parametrize(
    "parameter_axes",
    [
        [[], [1, 2]],
        [[1, 2], []],
        [[1], iter(())],
    ],
)
def test_execute_cartesian_experiments_empty_axis_returns_no_runs(parameter_axes):
    factory_calls = 0

    def factory(combination):
        nonlocal factory_calls
        factory_calls += 1
        return _case(combination)

    runs = execute_cartesian_experiments(parameter_axes, factory)

    assert runs == ()
    assert type(runs) is tuple
    assert factory_calls == 0


def test_execute_cartesian_experiments_preserves_one_axis_order_exactly_once():
    factory_calls = []
    simulation_calls = []

    def factory(combination):
        factory_calls.append(combination)

        def simulation():
            simulation_calls.append(combination)
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, float(combination[0])],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    runs = execute_cartesian_experiments([[3, 1, 2]], factory)

    expected = [(3,), (1,), (2,)]
    assert factory_calls == expected
    assert simulation_calls == expected
    assert tuple(run.run_id for run in runs) == (
        "case-3",
        "case-1",
        "case-2",
    )
    assert tuple(run.metrics.final_output for run in runs) == (3.0, 1.0, 2.0)


def test_execute_cartesian_experiments_preserves_product_order_exactly_once():
    factory_calls = []
    simulation_calls = []

    def factory(combination):
        factory_calls.append(combination)

        def simulation():
            simulation_calls.append(combination)
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    runs = execute_cartesian_experiments(
        (["low", "high"], [2, 1], ["a", "b"]),
        factory,
    )
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

    assert factory_calls == expected
    assert simulation_calls == expected
    assert tuple(run.user_metadata["combination"] for run in runs) == tuple(
        expected
    )
    assert all(isinstance(run, ExperimentRun) for run in runs)


def test_execute_cartesian_experiments_returns_an_immutable_tuple():
    runs = execute_cartesian_experiments([[1], [2]], _case)

    with pytest.raises(TypeError):
        runs[0] = runs[0]
    assert not hasattr(runs, "append")


@pytest.mark.parametrize("invalid_result", [None, {}, (), object()])
def test_execute_cartesian_experiments_rejects_invalid_factory_output_before_execution(
    invalid_result,
):
    factory_calls = []
    simulation_calls = 0

    def factory(combination):
        factory_calls.append(combination)
        if combination == (1, "b"):
            return invalid_result

        def simulation():
            nonlocal simulation_calls
            simulation_calls += 1
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    with pytest.raises(
        TypeError,
        match=r"parameter_values\[1\] must be an ExperimentCase",
    ):
        execute_cartesian_experiments([[1, 2], ["a", "b"]], factory)

    assert factory_calls == [(1, "a"), (1, "b")]
    assert simulation_calls == 0


def test_execute_cartesian_experiments_propagates_factory_exception_before_execution():
    failure = RuntimeError("factory failed")
    factory_calls = []
    simulation_calls = 0

    def factory(combination):
        factory_calls.append(combination)
        if combination == (1, "b"):
            raise failure

        def simulation():
            nonlocal simulation_calls
            simulation_calls += 1
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    with pytest.raises(RuntimeError, match="factory failed") as caught:
        execute_cartesian_experiments([[1, 2], ["a", "b"]], factory)

    assert caught.value is failure
    assert factory_calls == [(1, "a"), (1, "b")]
    assert simulation_calls == 0


def test_execute_cartesian_experiments_stops_after_simulation_exception():
    failure = RuntimeError("simulation failed")
    factory_calls = []
    simulation_calls = []

    def factory(combination):
        factory_calls.append(combination)

        def simulation():
            simulation_calls.append(combination)
            if combination == (1, "b"):
                raise failure
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        return _case(combination, simulation=simulation)

    with pytest.raises(RuntimeError, match="simulation failed") as caught:
        execute_cartesian_experiments([[1, 2], ["a", "b"]], factory)

    assert caught.value is failure
    assert factory_calls == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
    assert simulation_calls == [(1, "a"), (1, "b")]


def test_execute_cartesian_experiments_propagates_run_validation_failure():
    factory_calls = []
    simulation_calls = []

    def factory(combination):
        factory_calls.append(combination)

        def simulation():
            simulation_calls.append(combination)
            return SISOSimulationResult(
                time=[0.0, 1.0],
                output=[0.0, 1.0],
                reference=[1.0, 1.0],
            )

        method = "" if combination == (1, "b") else "exact"
        return _case(combination, simulation=simulation, method=method)

    with pytest.raises(ValueError, match="method must be a non-empty string"):
        execute_cartesian_experiments([[1, 2], ["a", "b"]], factory)

    assert factory_calls == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
    assert simulation_calls == [(1, "a"), (1, "b")]


def test_execute_cartesian_experiments_has_deterministic_records():
    axes = ([2, 1], ["a", "b"])

    first_records = tuple(
        run.reproducibility_record()
        for run in execute_cartesian_experiments(axes, _case)
    )
    second_records = tuple(
        run.reproducibility_record()
        for run in execute_cartesian_experiments(axes, _case)
    )

    assert first_records == second_records
