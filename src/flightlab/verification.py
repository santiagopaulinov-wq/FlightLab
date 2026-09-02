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
