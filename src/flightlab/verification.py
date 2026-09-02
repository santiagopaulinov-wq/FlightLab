"""Independent verification benchmarks for FlightLab's numerical foundation."""

from datetime import UTC, datetime
from math import exp

import numpy as np

from flightlab.experiment import ExperimentRun, experiment_run
from flightlab.response import response_metrics
from flightlab.state_space import StateSpace

_ACCEPTANCE_TOLERANCE = 1.0e-12
_RUN_ID = "verification-linear-state-space-closed-form-v1"
_CREATED_AT = datetime(2026, 9, 2, tzinfo=UTC)


def run_linear_state_space_verification_benchmark() -> ExperimentRun:
    """Verify one fixed linear system against its analytical closed form."""
    system = StateSpace(
        [[-1.0, 1.0], [0.0, -2.0]],
        [[0.0], [1.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )
    initial_state = np.array([1.5, -0.5])
    control = np.array([2.0])
    time = np.array([0.0, 0.125, 0.5, 1.25, 2.0])

    analytical_eigenvalues = np.array([-2.0, -1.0])
    analytical_state = np.array(
        [
            (
                1.0 - exp(-tau) + 1.5 * exp(-2.0 * tau),
                1.0 - 1.5 * exp(-2.0 * tau),
            )
            for tau in time - time[0]
        ]
    )
    analytical_output = analytical_state[:, 0]

    eigenvalues = np.asarray(system.eigenvalues())
    if eigenvalues.shape != (2,):
        raise ValueError("benchmark eigenvalues must have shape (2,)")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("benchmark eigenvalues must contain only finite values")
    eigenvalues = np.sort_complex(eigenvalues)

    state, output = system.simulate(initial_state, control, time, method="exact")
    state = np.asarray(state)
    output = np.asarray(output)
    if state.shape != (5, 2):
        raise ValueError("benchmark state trajectory must have shape (5, 2)")
    if output.shape != (5, 1):
        raise ValueError("benchmark output trajectory must have shape (5, 1)")
    if np.iscomplexobj(state) or not np.all(np.isfinite(state)):
        raise ValueError(
            "benchmark state trajectory must contain only finite real values"
        )
    if np.iscomplexobj(output) or not np.all(np.isfinite(output)):
        raise ValueError(
            "benchmark output trajectory must contain only finite real values"
        )

    eigenvalue_residual = float(
        np.max(np.abs(eigenvalues - analytical_eigenvalues))
    )
    state_residual = float(np.max(np.abs(state - analytical_state)))
    output_residual = float(
        np.max(np.abs(output[:, 0] - analytical_output))
    )
    initial_state_exact = bool(np.array_equal(state[0], initial_state))
    passed = bool(
        initial_state_exact
        and eigenvalue_residual <= _ACCEPTANCE_TOLERANCE
        and state_residual <= _ACCEPTANCE_TOLERANCE
        and output_residual <= _ACCEPTANCE_TOLERANCE
    )

    metrics = response_metrics(time, output[:, 0], analytical_output)
    if metrics.maximum_absolute_tracking_error != output_residual:
        raise ValueError("benchmark output residual evidence is inconsistent")

    return experiment_run(
        time=time,
        initial_state=initial_state,
        metrics=metrics,
        method="exact",
        system={
            "A": (-1.0, 1.0, 0.0, -2.0),
            "A_shape": (2, 2),
            "B": (0.0, 1.0),
            "B_shape": (2, 1),
            "C": (1.0, 0.0),
            "C_shape": (1, 2),
            "D": (0.0,),
            "D_shape": (1, 1),
            "name": "independent_analytical_two_state_linear_v1",
        },
        controller={"type": "none"},
        reference={
            "analytical_eigenvalues": (-2.0, -1.0),
            "analytical_state_shape": (5, 2),
            "analytical_state_trajectory": tuple(analytical_state.flat),
            "input": 2.0,
            "kind": "closed_form_constant_input",
            "time_grid": tuple(time),
            "x1_formula": "1 - exp(-tau) + 1.5 exp(-2 tau)",
            "x2_formula": "1 - 1.5 exp(-2 tau)",
        },
        user_metadata={
            "acceptance_tolerance": _ACCEPTANCE_TOLERANCE,
            "initial_state_exact": initial_state_exact,
            "maximum_absolute_eigenvalue_residual": eigenvalue_residual,
            "maximum_absolute_output_residual": output_residual,
            "maximum_absolute_state_residual": state_residual,
            "passed": passed,
        },
        run_id=_RUN_ID,
        created_at=_CREATED_AT,
    )


def run_scipy_linear_state_space_verification_benchmark() -> ExperimentRun:
    """Cross-check one fixed linear system against SciPy's public APIs."""
    import scipy
    from scipy.linalg import eigvals
    from scipy.signal import lsim

    matrix_a = np.array([[0.0, 1.0], [-4.0, -0.8]])
    matrix_b = np.array([[0.0], [1.0]])
    matrix_c = np.array([[1.0, 0.25]])
    matrix_d = np.array([[0.1]])
    initial_state = np.array([0.75, -0.25])
    time = np.array(
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
    control = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
        ]
    )
    eigenvalue_tolerance = 1.0e-12
    state_tolerance = 1.0e-10
    output_tolerance = 1.0e-10

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    flightlab_eigenvalues = np.asarray(system.eigenvalues())
    scipy_eigenvalues = np.asarray(eigvals(matrix_a, check_finite=True))
    if flightlab_eigenvalues.shape != (2,):
        raise ValueError("FlightLab benchmark eigenvalues must have shape (2,)")
    if scipy_eigenvalues.shape != (2,):
        raise ValueError("SciPy benchmark eigenvalues must have shape (2,)")
    if not np.all(np.isfinite(flightlab_eigenvalues)):
        raise ValueError(
            "FlightLab benchmark eigenvalues must contain only finite values"
        )
    if not np.all(np.isfinite(scipy_eigenvalues)):
        raise ValueError("SciPy benchmark eigenvalues must contain only finite values")
    flightlab_eigenvalues = flightlab_eigenvalues[
        np.argsort(flightlab_eigenvalues.imag)
    ]
    scipy_eigenvalues = scipy_eigenvalues[np.argsort(scipy_eigenvalues.imag)]

    flightlab_state, flightlab_output = system.simulate(
        initial_state, control[:, np.newaxis], time, method="exact"
    )
    scipy_time, scipy_output, scipy_state = lsim(
        (matrix_a, matrix_b, matrix_c, matrix_d),
        U=control,
        T=time,
        X0=initial_state,
        interp=False,
    )
    flightlab_state = np.asarray(flightlab_state)
    flightlab_output = np.asarray(flightlab_output)
    scipy_time = np.asarray(scipy_time)
    scipy_output = np.asarray(scipy_output)
    scipy_state = np.asarray(scipy_state)

    if flightlab_state.shape != (17, 2):
        raise ValueError(
            "FlightLab benchmark state trajectory must have shape (17, 2)"
        )
    if scipy_state.shape != (17, 2):
        raise ValueError("SciPy benchmark state trajectory must have shape (17, 2)")
    if flightlab_output.shape != (17, 1):
        raise ValueError(
            "FlightLab benchmark output trajectory must have shape (17, 1)"
        )
    if scipy_output.shape != (17,):
        raise ValueError("SciPy benchmark output trajectory must have shape (17,)")
    if scipy_time.shape != (17,):
        raise ValueError("SciPy benchmark time must have shape (17,)")

    if np.iscomplexobj(flightlab_state) or not np.all(
        np.isfinite(flightlab_state)
    ):
        raise ValueError(
            "FlightLab benchmark state trajectory must contain only finite real values"
        )
    if np.iscomplexobj(scipy_state) or not np.all(np.isfinite(scipy_state)):
        raise ValueError(
            "SciPy benchmark state trajectory must contain only finite real values"
        )
    if np.iscomplexobj(flightlab_output) or not np.all(
        np.isfinite(flightlab_output)
    ):
        raise ValueError(
            "FlightLab benchmark output trajectory must contain only finite real values"
        )
    if np.iscomplexobj(scipy_output) or not np.all(np.isfinite(scipy_output)):
        raise ValueError(
            "SciPy benchmark output trajectory must contain only finite real values"
        )
    if np.iscomplexobj(scipy_time) or not np.all(np.isfinite(scipy_time)):
        raise ValueError("SciPy benchmark time must contain only finite real values")
    if not np.array_equal(scipy_time, time):
        raise ValueError("SciPy benchmark time must exactly match the fixed time grid")

    eigenvalue_residual = float(
        np.max(np.abs(flightlab_eigenvalues - scipy_eigenvalues))
    )
    state_residual = float(np.max(np.abs(flightlab_state - scipy_state)))
    output_residual = float(
        np.max(np.abs(flightlab_output[:, 0] - scipy_output))
    )
    flightlab_initial_state_exact = bool(
        np.array_equal(flightlab_state[0], initial_state)
    )
    scipy_initial_state_exact = bool(np.array_equal(scipy_state[0], initial_state))
    passed = bool(
        flightlab_initial_state_exact
        and scipy_initial_state_exact
        and eigenvalue_residual <= eigenvalue_tolerance
        and state_residual <= state_tolerance
        and output_residual <= output_tolerance
    )

    metrics = response_metrics(time, flightlab_output[:, 0], scipy_output)
    if metrics.maximum_absolute_tracking_error != output_residual:
        raise ValueError("SciPy benchmark output residual evidence is inconsistent")

    return experiment_run(
        time=time,
        initial_state=initial_state,
        metrics=metrics,
        method="exact",
        system={
            "A": (0.0, 1.0, -4.0, -0.8),
            "A_shape": (2, 2),
            "B": (0.0, 1.0),
            "B_shape": (2, 1),
            "C": (1.0, 0.25),
            "C_shape": (1, 2),
            "D": (0.1,),
            "D_shape": (1, 1),
            "name": "independent_scipy_damped_oscillator_v1",
        },
        controller={"type": "none"},
        reference={
            "eigenvalue_api": "scipy.linalg.eigvals",
            "input_interpolation": "zero_order_hold",
            "input_trajectory": tuple(control),
            "library": "scipy",
            "library_version": scipy.__version__,
            "simulation_api": "scipy.signal.lsim",
            "state_shape": (17, 2),
            "state_trajectory": tuple(scipy_state.flat),
            "time_grid": tuple(time),
        },
        user_metadata={
            "eigenvalue_tolerance": eigenvalue_tolerance,
            "flightlab_initial_state_exact": flightlab_initial_state_exact,
            "maximum_absolute_eigenvalue_residual": eigenvalue_residual,
            "maximum_absolute_output_residual": output_residual,
            "maximum_absolute_state_residual": state_residual,
            "output_tolerance": output_tolerance,
            "passed": passed,
            "scipy_initial_state_exact": scipy_initial_state_exact,
            "state_tolerance": state_tolerance,
        },
        run_id="verification-linear-state-space-scipy-lsim-v1",
        created_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
    )
