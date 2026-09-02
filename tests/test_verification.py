import json
import subprocess
import sys
from math import exp

import numpy as np
import pytest

from flightlab import verification
from flightlab.experiment import ExperimentRun
from flightlab.state_space import StateSpace
from flightlab.verification import (
    run_linear_state_space_verification_benchmark,
    run_scipy_linear_state_space_verification_benchmark,
)

_TIME = np.array([0.0, 0.125, 0.5, 1.25, 2.0])
_INITIAL_STATE = np.array([1.5, -0.5])
_TOLERANCE = 1.0e-12
_SCIPY_TIME = np.array(
    [
        0.0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.375,
        1.5,
        1.625,
        1.75,
        1.875,
        2.0,
    ]
)
_SCIPY_INPUT = np.array(
    [0.5] * 4 + [-1.0] * 6 + [0.25] * 7
)
_SCIPY_INITIAL_STATE = np.array([0.75, -0.25])
_SCIPY_A = np.array([[0.0, 1.0], [-4.0, -0.8]])
_SCIPY_B = np.array([[0.0], [1.0]])
_SCIPY_C = np.array([[1.0, 0.25]])
_SCIPY_D = np.array([[0.1]])


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


def _scipy_reference():
    from scipy.linalg import eigvals
    from scipy.signal import lsim

    eigenvalues = eigvals(_SCIPY_A, check_finite=True)
    time, output, state = lsim(
        (_SCIPY_A, _SCIPY_B, _SCIPY_C, _SCIPY_D),
        U=_SCIPY_INPUT,
        T=_SCIPY_TIME,
        X0=_SCIPY_INITIAL_STATE,
        interp=False,
    )
    return eigenvalues, time, output, state


def test_verification_module_does_not_import_scipy_eagerly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import flightlab.verification; "
                "assert 'scipy' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_scipy_benchmark_calls_fixed_public_apis(monkeypatch):
    from scipy import linalg, signal

    original_eigvals = linalg.eigvals
    original_lsim = signal.lsim
    calls = []

    def checked_eigvals(matrix, *, check_finite):
        np.testing.assert_array_equal(matrix, _SCIPY_A)
        assert check_finite is True
        calls.append("eigvals")
        return original_eigvals(matrix, check_finite=check_finite)

    def checked_lsim(system, *, U, T, X0, interp):
        for actual, expected in zip(
            system,
            (_SCIPY_A, _SCIPY_B, _SCIPY_C, _SCIPY_D),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(U, _SCIPY_INPUT)
        np.testing.assert_array_equal(T, _SCIPY_TIME)
        np.testing.assert_array_equal(X0, _SCIPY_INITIAL_STATE)
        assert interp is False
        calls.append("lsim")
        return original_lsim(system, U=U, T=T, X0=X0, interp=interp)

    monkeypatch.setattr(linalg, "eigvals", checked_eigvals)
    monkeypatch.setattr(signal, "lsim", checked_lsim)

    run = run_scipy_linear_state_space_verification_benchmark()

    assert calls == ["eigvals", "lsim"]
    assert isinstance(run, ExperimentRun)
    assert run.system["A"] == tuple(_SCIPY_A.flat)
    assert run.system["B"] == tuple(_SCIPY_B.flat)
    assert run.system["C"] == tuple(_SCIPY_C.flat)
    assert run.system["D"] == tuple(_SCIPY_D.flat)
    assert run.reference["input_trajectory"] == tuple(_SCIPY_INPUT)
    assert run.reference["time_grid"] == tuple(_SCIPY_TIME)


def test_scipy_benchmark_real_eigenvalues_agree():
    scipy_eigenvalues, _, _, _ = _scipy_reference()
    flightlab_eigenvalues = StateSpace(
        _SCIPY_A, _SCIPY_B, _SCIPY_C, _SCIPY_D
    ).eigenvalues()
    expected_residual = np.max(
        np.abs(
            flightlab_eigenvalues[np.argsort(flightlab_eigenvalues.imag)]
            - scipy_eigenvalues[np.argsort(scipy_eigenvalues.imag)]
        )
    )

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.user_metadata["maximum_absolute_eigenvalue_residual"] == float(
        expected_residual
    )
    assert expected_residual <= 1.0e-12


def test_scipy_benchmark_real_states_agree():
    _, _, scipy_output, scipy_state = _scipy_reference()
    flightlab_state, flightlab_output = StateSpace(
        _SCIPY_A, _SCIPY_B, _SCIPY_C, _SCIPY_D
    ).simulate(
        _SCIPY_INITIAL_STATE,
        _SCIPY_INPUT[:, np.newaxis],
        _SCIPY_TIME,
        method="exact",
    )
    expected_residual = float(np.max(np.abs(flightlab_state - scipy_state)))

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.user_metadata["maximum_absolute_state_residual"] == expected_residual
    assert expected_residual <= 1.0e-10
    np.testing.assert_array_equal(
        run.reference["state_trajectory"], scipy_state.ravel()
    )
    np.testing.assert_array_equal(run.metrics.output, flightlab_output[:, 0])
    np.testing.assert_array_equal(run.metrics.reference, scipy_output)


def test_scipy_benchmark_real_outputs_agree():
    _, _, scipy_output, _ = _scipy_reference()
    _, flightlab_output = StateSpace(
        _SCIPY_A, _SCIPY_B, _SCIPY_C, _SCIPY_D
    ).simulate(
        _SCIPY_INITIAL_STATE,
        _SCIPY_INPUT[:, np.newaxis],
        _SCIPY_TIME,
        method="exact",
    )
    expected_residual = float(
        np.max(np.abs(flightlab_output[:, 0] - scipy_output))
    )

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.user_metadata["maximum_absolute_output_residual"] == expected_residual
    assert run.metrics.maximum_absolute_tracking_error == expected_residual
    assert expected_residual <= 1.0e-10


def test_scipy_benchmark_requires_exact_time_and_initial_states():
    run = run_scipy_linear_state_space_verification_benchmark()

    np.testing.assert_array_equal(run.metrics.time, _SCIPY_TIME)
    np.testing.assert_array_equal(run.initial_state, _SCIPY_INITIAL_STATE)
    assert run.user_metadata["flightlab_initial_state_exact"] is True
    assert run.user_metadata["scipy_initial_state_exact"] is True
    assert run.user_metadata["passed"] is True


def test_scipy_benchmark_records_version_tolerances_and_identity():
    import scipy

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.reference["library"] == "scipy"
    assert run.reference["library_version"] == scipy.__version__
    assert run.reference["eigenvalue_api"] == "scipy.linalg.eigvals"
    assert run.reference["simulation_api"] == "scipy.signal.lsim"
    assert run.user_metadata["eigenvalue_tolerance"] == 1.0e-12
    assert run.user_metadata["state_tolerance"] == 1.0e-10
    assert run.user_metadata["output_tolerance"] == 1.0e-10
    assert run.run_id == "verification-linear-state-space-scipy-lsim-v1"
    assert run.created_at.isoformat() == "2026-09-02T12:00:00+00:00"


def test_scipy_benchmark_reproducibility_records_are_deterministic_and_detached():
    first = run_scipy_linear_state_space_verification_benchmark()
    second = run_scipy_linear_state_space_verification_benchmark()

    first_record = first.reproducibility_record()
    second_record = second.reproducibility_record()
    assert first_record == second_record
    json.dumps(first_record, allow_nan=False)

    first_record["reference"]["library_version"] = "changed"
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
def test_scipy_benchmark_returns_failed_run_for_well_formed_over_limit_evidence(
    monkeypatch, failed_quantity, metadata_key
):
    from scipy import linalg, signal

    eigenvalues, scipy_time, scipy_output, scipy_state = _scipy_reference()
    if failed_quantity == "eigenvalue":
        eigenvalues = eigenvalues.copy()
        eigenvalues[0] += 2.0e-12
    elif failed_quantity == "state":
        scipy_state = scipy_state.copy()
        scipy_state[2, 1] += 2.0e-10
    else:
        scipy_output = scipy_output.copy()
        scipy_output[2] += 2.0e-10

    monkeypatch.setattr(linalg, "eigvals", lambda *args, **kwargs: eigenvalues)
    monkeypatch.setattr(
        signal,
        "lsim",
        lambda *args, **kwargs: (scipy_time, scipy_output, scipy_state),
    )

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.user_metadata[metadata_key] > run.user_metadata[
        metadata_key.replace("maximum_absolute_", "").replace("residual", "tolerance")
    ]
    assert run.user_metadata["passed"] is False


@pytest.mark.parametrize("implementation", ["flightlab", "scipy"])
def test_scipy_benchmark_returns_failed_run_when_initial_state_is_not_exact(
    monkeypatch, implementation
):
    from scipy import signal

    if implementation == "flightlab":
        original_simulate = StateSpace.simulate

        def changed_simulate(self, x0, u, time, method="euler"):
            state, output = original_simulate(self, x0, u, time, method=method)
            state[0, 0] = np.nextafter(state[0, 0], np.inf)
            return state, output

        monkeypatch.setattr(StateSpace, "simulate", changed_simulate)
    else:
        _, scipy_time, scipy_output, scipy_state = _scipy_reference()
        scipy_state = scipy_state.copy()
        scipy_state[0, 0] = np.nextafter(scipy_state[0, 0], np.inf)
        monkeypatch.setattr(
            signal,
            "lsim",
            lambda *args, **kwargs: (scipy_time, scipy_output, scipy_state),
        )

    run = run_scipy_linear_state_space_verification_benchmark()

    assert run.user_metadata[f"{implementation}_initial_state_exact"] is False
    assert run.user_metadata["maximum_absolute_state_residual"] <= 1.0e-10
    assert run.user_metadata["passed"] is False


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("time_shape", "SciPy benchmark time must have shape"),
        ("state_shape", "SciPy benchmark state trajectory must have shape"),
        ("output_shape", "SciPy benchmark output trajectory must have shape"),
        ("nonfinite_time", "SciPy benchmark time must contain only finite real"),
        (
            "nonfinite_state",
            "SciPy benchmark state trajectory must contain only finite real",
        ),
        (
            "complex_output",
            "SciPy benchmark output trajectory must contain only finite real",
        ),
        ("time_mismatch", "SciPy benchmark time must exactly match"),
    ],
)
def test_scipy_benchmark_rejects_malformed_reference_trajectory_evidence(
    monkeypatch, failure, message
):
    from scipy import signal

    _, scipy_time, scipy_output, scipy_state = _scipy_reference()
    if failure == "time_shape":
        scipy_time = scipy_time[:-1]
    elif failure == "state_shape":
        scipy_state = scipy_state[:, :1]
    elif failure == "output_shape":
        scipy_output = scipy_output[:, np.newaxis]
    elif failure == "nonfinite_time":
        scipy_time = scipy_time.copy()
        scipy_time[2] = np.nan
    elif failure == "nonfinite_state":
        scipy_state = scipy_state.copy()
        scipy_state[2, 0] = np.inf
    elif failure == "complex_output":
        scipy_output = scipy_output.astype(complex)
    else:
        scipy_time = scipy_time.copy()
        scipy_time[2] = np.nextafter(scipy_time[2], np.inf)

    monkeypatch.setattr(
        signal,
        "lsim",
        lambda *args, **kwargs: (scipy_time, scipy_output, scipy_state),
    )

    with pytest.raises(ValueError, match=message):
        run_scipy_linear_state_space_verification_benchmark()


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.array([-1.0]), "SciPy benchmark eigenvalues must have shape"),
        (
            np.array([complex(np.nan, 1.0), complex(np.nan, -1.0)]),
            "SciPy benchmark eigenvalues must contain only finite",
        ),
    ],
)
def test_scipy_benchmark_rejects_malformed_reference_eigenvalue_evidence(
    monkeypatch, eigenvalues, message
):
    from scipy import linalg

    monkeypatch.setattr(linalg, "eigvals", lambda *args, **kwargs: eigenvalues)

    with pytest.raises(ValueError, match=message):
        run_scipy_linear_state_space_verification_benchmark()


def test_scipy_benchmark_rejects_inconsistent_output_residual_evidence(monkeypatch):
    original_response_metrics = verification.response_metrics

    def inconsistent_metrics(*args, **kwargs):
        metrics = original_response_metrics(*args, **kwargs)
        return metrics._replace(
            maximum_absolute_tracking_error=(
                metrics.maximum_absolute_tracking_error + 1.0
            )
        )

    monkeypatch.setattr(verification, "response_metrics", inconsistent_metrics)

    with pytest.raises(ValueError, match="output residual evidence is inconsistent"):
        run_scipy_linear_state_space_verification_benchmark()
