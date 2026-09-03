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


def run_nasa_gtm_longitudinal_modal_verification_benchmark() -> ExperimentRun:
    """Reproduce the published NASA GTM longitudinal modal quantities."""
    mass_matrix = np.array(
        [
            [11.1138, 0.0, 0.0, 0.0],
            [0.0, 11.1757, 0.0, 0.0],
            [0.0, 0.1310, 0.7841, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    stability_matrix = np.array(
        [
            [-0.0558, -0.4364, -0.7480, -0.4595],
            [-1.7284, -6.3068, 10.9544, -0.0306],
            [-0.0074, -1.7648, -0.3370, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    published_eigenvalues = np.array(
        [complex(-0.0042, 0.0763), complex(-0.5779, 1.4491)]
    )
    published_natural_frequencies = np.array([0.0764, 1.5601])
    published_damping_ratios = np.array([0.0545, 0.3704])
    eigenvalue_tolerance = 7.0e-4
    natural_frequency_tolerance = 2.0e-4
    damping_ratio_tolerance = 4.0e-4

    for name, matrix in (
        ("descriptor mass matrix", mass_matrix),
        ("descriptor stability matrix", stability_matrix),
    ):
        if matrix.shape != (4, 4):
            raise ValueError(f"GTM {name} must have shape (4, 4)")
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"GTM {name} must contain only finite real values")

    state_matrix = np.linalg.solve(mass_matrix, stability_matrix)
    if state_matrix.shape != (4, 4):
        raise ValueError("GTM standard state matrix must have shape (4, 4)")
    if np.iscomplexobj(state_matrix) or not np.all(np.isfinite(state_matrix)):
        raise ValueError(
            "GTM standard state matrix must contain only finite real values"
        )

    system = StateSpace(
        state_matrix, np.zeros((4, 1)), np.eye(4), np.zeros((4, 1))
    )
    eigenvalues = np.asarray(system.eigenvalues())
    properties = tuple(system.modal_properties())
    if eigenvalues.shape != (4,):
        raise ValueError("GTM benchmark eigenvalues must have shape (4,)")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("GTM benchmark eigenvalues must contain only finite values")
    if len(properties) != 4:
        raise ValueError("GTM benchmark modal properties must contain four members")

    positive_indices = np.flatnonzero(eigenvalues.imag > 0.0)
    negative_eigenvalues = eigenvalues[eigenvalues.imag < 0.0]
    if positive_indices.size != 2 or negative_eigenvalues.size != 2:
        raise ValueError("GTM benchmark eigenvalues must form two conjugate pairs")
    positive_indices = positive_indices[
        np.argsort(np.abs(eigenvalues[positive_indices]))
    ]
    matched_eigenvalues = eigenvalues[positive_indices]
    if any(
        not np.any(np.isclose(negative_eigenvalues, value.conjugate(), rtol=0.0, atol=1e-12))
        for value in matched_eigenvalues
    ):
        raise ValueError("GTM benchmark eigenvalues must form two conjugate pairs")

    natural_frequencies = []
    damping_ratios = []
    for index in positive_indices:
        prop = properties[int(index)]
        values = (prop.eigenvalue, prop.natural_frequency, prop.damping_ratio)
        if prop.natural_frequency is None or prop.damping_ratio is None:
            raise ValueError("GTM benchmark modal quantities must be oscillatory")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("GTM benchmark modal quantities must be finite")
        if prop.natural_frequency <= 0.0 or prop.damping_ratio <= 0.0:
            raise ValueError("GTM benchmark modal quantities must be positive")
        if prop.eigenvalue != eigenvalues[index]:
            raise ValueError("GTM benchmark modal evidence is internally inconsistent")
        natural_frequencies.append(prop.natural_frequency)
        damping_ratios.append(prop.damping_ratio)

    natural_frequencies = np.asarray(natural_frequencies)
    damping_ratios = np.asarray(damping_ratios)
    eigenvalue_residual = float(
        np.max(np.abs(matched_eigenvalues - published_eigenvalues))
    )
    natural_frequency_residual = float(
        np.max(np.abs(natural_frequencies - published_natural_frequencies))
    )
    damping_ratio_residual = float(
        np.max(np.abs(damping_ratios - published_damping_ratios))
    )
    passed = bool(
        eigenvalue_residual <= eigenvalue_tolerance
        and natural_frequency_residual <= natural_frequency_tolerance
        and damping_ratio_residual <= damping_ratio_tolerance
    )

    time = np.array([0.0, 1.0])
    metrics = response_metrics(time, np.zeros(2), np.zeros(2))
    return experiment_run(
        time=time,
        initial_state=np.zeros(4),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(state_matrix.flat),
            "A_shape": (4, 4),
            "B": (0.0, 0.0, 0.0, 0.0),
            "B_shape": (4, 1),
            "C": tuple(np.eye(4).flat),
            "C_shape": (4, 4),
            "D": (0.0, 0.0, 0.0, 0.0),
            "D_shape": (4, 1),
            "auxiliary_input_output_matrices": True,
            "descriptor_mass_matrix": tuple(mass_matrix.flat),
            "descriptor_stability_matrix": tuple(stability_matrix.flat),
            "name": "nasa_gtm_rigid_body_longitudinal_mach_0_8_v1",
            "transformation": "A = numpy.linalg.solve(M_r, S)",
        },
        controller={"type": "none"},
        reference={
            "doi": "10.2514/6.2013-4746",
            "evidence_classification": "computational_software_verification",
            "ntrs_document_id": "20140008923",
            "paper": "AIAA 2013-4746",
            "published_damping_ratios": (0.0545, 0.3704),
            "published_eigenvalues_imaginary": (0.0763, 1.4491),
            "published_eigenvalues_real": (-0.0042, -0.5779),
            "published_mode_order": ("phugoid", "short_period"),
            "published_natural_frequencies": (0.0764, 1.5601),
            "section": "V.A",
            "source_printed_page": 28,
            "state_order": ("Delta V / V", "Delta alpha", "q", "Delta theta"),
            "trim_alpha_deg": 3.8142,
            "trim_altitude_ft": 35000.0,
            "trim_elevator_deg": -6.1497,
            "trim_mach": 0.8,
            "trim_speed_ft_per_s": 778.2063,
            "trim_theta_deg": -3.8142,
            "trim_thrust_lb": 5617.0,
        },
        user_metadata={
            "damping_ratio_tolerance": damping_ratio_tolerance,
            "eigenvalue_tolerance": eigenvalue_tolerance,
            "maximum_absolute_damping_ratio_residual": damping_ratio_residual,
            "maximum_absolute_eigenvalue_residual": eigenvalue_residual,
            "maximum_absolute_natural_frequency_residual": natural_frequency_residual,
            "natural_frequency_tolerance": natural_frequency_tolerance,
            "passed": passed,
        },
        run_id="verification-nasa-gtm-longitudinal-modal-v1",
        created_at=datetime(2026, 9, 2, 18, tzinfo=UTC),
    )
