import json
from math import exp

import numpy as np
import pytest

from flightlab.experiment import ExperimentRun
from flightlab.state_space import StateSpace
from flightlab.verification import run_linear_state_space_verification_benchmark

_TIME = np.array([0.0, 0.125, 0.5, 1.25, 2.0])
_INITIAL_STATE = np.array([1.5, -0.5])
_TOLERANCE = 1.0e-12


def _analytical_state():
    return np.array(
        [
            [
                1.0 - exp(-tau) + 1.5 * exp(-2.0 * tau),
                1.0 - 1.5 * exp(-2.0 * tau),
            ]
            for tau in _TIME
        ]
    )


def test_benchmark_returns_existing_experiment_run_for_fixed_system():
    run = run_linear_state_space_verification_benchmark()

    assert isinstance(run, ExperimentRun)
    assert run.method == "exact"
    np.testing.assert_array_equal(run.initial_state, _INITIAL_STATE)
    assert run.system == {
        "A": (-1.0, 1.0, 0.0, -2.0),
        "A_shape": (2, 2),
        "B": (0.0, 1.0),
        "B_shape": (2, 1),
        "C": (1.0, 0.0),
        "C_shape": (1, 2),
        "D": (0.0,),
        "D_shape": (1, 1),
        "name": "independent_analytical_two_state_linear_v1",
    }
    assert run.controller == {"type": "none"}
    assert run.reference["input"] == 2.0
    assert run.reference["time_grid"] == tuple(_TIME)


def test_benchmark_uses_analytical_eigenvalues_and_meets_tolerance():
    run = run_linear_state_space_verification_benchmark()

    assert run.reference["analytical_eigenvalues"] == (-2.0, -1.0)
    assert (
        run.user_metadata["maximum_absolute_eigenvalue_residual"] <= _TOLERANCE
    )


def test_benchmark_uses_analytical_state_and_output_trajectories():
    run = run_linear_state_space_verification_benchmark()
    expected_state = _analytical_state()

    np.testing.assert_array_equal(
        run.reference["analytical_state_trajectory"],
        expected_state.ravel(),
    )
    np.testing.assert_array_equal(run.metrics.reference, expected_state[:, 0])
    np.testing.assert_allclose(
        run.metrics.output,
        expected_state[:, 0],
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert run.user_metadata["maximum_absolute_state_residual"] <= _TOLERANCE
    assert run.user_metadata["maximum_absolute_output_residual"] <= _TOLERANCE


def test_benchmark_evidence_has_expected_shapes_finiteness_and_initial_state():
    run = run_linear_state_space_verification_benchmark()

    assert run.reference["analytical_state_shape"] == (5, 2)
    assert run.metrics.time.shape == (5,)
    assert run.metrics.output.shape == (5,)
    assert run.metrics.reference.shape == (5,)
    assert np.all(np.isfinite(run.metrics.time))
    assert np.all(np.isfinite(run.metrics.output))
    assert np.all(np.isfinite(run.metrics.reference))
    assert run.user_metadata["initial_state_exact"] is True
    assert run.user_metadata["passed"] is True


def test_benchmark_records_all_three_residuals_and_acceptance_threshold():
    run = run_linear_state_space_verification_benchmark()

    assert run.user_metadata["acceptance_tolerance"] == _TOLERANCE
    assert run.user_metadata["maximum_absolute_eigenvalue_residual"] == 0.0
    assert run.user_metadata["maximum_absolute_state_residual"] <= _TOLERANCE
    assert run.user_metadata["maximum_absolute_output_residual"] <= _TOLERANCE
    assert (
        run.metrics.maximum_absolute_tracking_error
        == run.user_metadata["maximum_absolute_output_residual"]
    )


def test_benchmark_reproducibility_records_are_repeatable_and_json_compatible():
    first = run_linear_state_space_verification_benchmark()
    second = run_linear_state_space_verification_benchmark()

    first_record = first.reproducibility_record()
    second_record = second.reproducibility_record()
    assert first_record == second_record
    assert first_record["run_id"] == (
        "verification-linear-state-space-closed-form-v1"
    )
    assert first_record["created_at"] == "2026-09-02T00:00:00+00:00"
    json.dumps(first_record, allow_nan=False)

    first_record["user_metadata"]["passed"] = False
    assert first.reproducibility_record() == second_record


@pytest.mark.parametrize(
    ("failed_quantity", "metadata_key"),
    [
        ("eigenvalue", "maximum_absolute_eigenvalue_residual"),
        ("state", "maximum_absolute_state_residual"),
        ("output", "maximum_absolute_output_residual"),
    ],
)
def test_benchmark_records_failed_acceptance_when_any_residual_exceeds_tolerance(
    monkeypatch, failed_quantity, metadata_key
):
    eigenvalues = np.array([-2.0, -1.0])
    state = _analytical_state()
    output = state[:, :1].copy()
    if failed_quantity == "eigenvalue":
        eigenvalues[1] += 2.0e-12
    elif failed_quantity == "state":
        state[2, 1] += 2.0e-12
    else:
        output[2, 0] += 2.0e-12

    monkeypatch.setattr(StateSpace, "eigenvalues", lambda self: eigenvalues)
    monkeypatch.setattr(
        StateSpace,
        "simulate",
        lambda self, x0, u, time, method="euler": (state, output),
    )

    run = run_linear_state_space_verification_benchmark()

    assert run.user_metadata[metadata_key] > _TOLERANCE
    assert run.user_metadata["passed"] is False


def test_benchmark_requires_exact_initial_state_even_within_residual_tolerance(
    monkeypatch,
):
    original_simulate = StateSpace.simulate

    def simulate_with_changed_initial_state(self, x0, u, time, method="euler"):
        state, output = original_simulate(self, x0, u, time, method=method)
        state[0, 0] = np.nextafter(state[0, 0], np.inf)
        return state, output

    monkeypatch.setattr(StateSpace, "simulate", simulate_with_changed_initial_state)

    run = run_linear_state_space_verification_benchmark()

    assert run.user_metadata["maximum_absolute_state_residual"] <= _TOLERANCE
    assert run.user_metadata["initial_state_exact"] is False
    assert run.user_metadata["passed"] is False


def test_benchmark_computes_independent_maximum_absolute_residuals(monkeypatch):
    analytical_state = _analytical_state()
    state = analytical_state.copy()
    output = analytical_state[:, :1].copy()
    state[2, 1] += 2.5e-13
    output[3, 0] -= 4.0e-13

    monkeypatch.setattr(
        StateSpace,
        "eigenvalues",
        lambda self: np.array([-2.0 + 3.0e-13, -1.0]),
    )
    monkeypatch.setattr(
        StateSpace,
        "simulate",
        lambda self, x0, u, time, method="euler": (state, output),
    )

    run = run_linear_state_space_verification_benchmark()

    assert run.user_metadata["maximum_absolute_eigenvalue_residual"] == pytest.approx(
        3.0e-13
    )
    assert run.user_metadata["maximum_absolute_state_residual"] == pytest.approx(
        2.5e-13
    )
    assert run.user_metadata["maximum_absolute_output_residual"] == pytest.approx(
        4.0e-13
    )
    assert run.user_metadata["passed"] is True


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.array([-2.0]), "benchmark eigenvalues must have shape"),
        (np.array([-2.0, np.nan]), "benchmark eigenvalues must contain only finite"),
    ],
)
def test_benchmark_rejects_inconsistent_eigenvalue_evidence(
    monkeypatch, eigenvalues, message
):
    monkeypatch.setattr(StateSpace, "eigenvalues", lambda self: eigenvalues)

    with pytest.raises(ValueError, match=message):
        run_linear_state_space_verification_benchmark()


@pytest.mark.parametrize(
    ("state", "output", "message"),
    [
        (
            np.zeros((4, 2)),
            np.zeros((5, 1)),
            "benchmark state trajectory must have shape",
        ),
        (
            np.zeros((5, 2)),
            np.zeros((5, 2)),
            "benchmark output trajectory must have shape",
        ),
        (
            np.full((5, 2), np.nan),
            np.zeros((5, 1)),
            "benchmark state trajectory must contain only finite real",
        ),
        (
            np.zeros((5, 2)),
            np.full((5, 1), np.inf),
            "benchmark output trajectory must contain only finite real",
        ),
    ],
)
def test_benchmark_rejects_inconsistent_trajectory_evidence(
    monkeypatch, state, output, message
):
    monkeypatch.setattr(
        StateSpace,
        "simulate",
        lambda self, x0, u, time, method="euler": (state, output),
    )

    with pytest.raises(ValueError, match=message):
        run_linear_state_space_verification_benchmark()
