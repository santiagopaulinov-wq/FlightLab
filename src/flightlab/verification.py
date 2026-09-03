"""Independent verification benchmarks for FlightLab's numerical foundation."""

from datetime import UTC, datetime
from math import exp

import numpy as np

from flightlab.experiment import ExperimentRun, experiment_run
from flightlab.response import response_metrics
from flightlab.state_space import (
    BalancedTruncation,
    LuenbergerObserverInterconnection,
    ObserverBasedOutputFeedbackInterconnection,
    StateSpace,
)

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


def _nasa_unstable_roll_reference(angular_frequencies: np.ndarray) -> np.ndarray:
    return np.array(
        [
            0.78208 * (s**2 + 0.2175 * s + 0.5861)
            / (
                (s + 0.7599)
                * (s - 0.02004)
                * (s**2 + 0.1133 * s + 0.6375)
            )
            for frequency in angular_frequencies
            for s in (complex(0.0, frequency),)
        ],
        dtype=complex,
    )


def run_nasa_unstable_roll_frequency_response_verification_benchmark(
) -> ExperimentRun:
    """Verify frequency response against a published NASA roll model."""
    matrix_a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0097081024500,
                -0.4699353727332,
                -0.706097742,
                -0.85316,
            ],
        ]
    )
    matrix_b = np.array([[0.0], [0.0], [0.0], [1.0]])
    matrix_c = np.array([[0.458377088, 0.170102400, 0.78208, 0.0]])
    matrix_d = np.array([[0.0]])
    frequencies = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    tolerance = 1.0e-12

    for name, matrix, shape in (
        ("A", matrix_a, (4, 4)),
        ("B", matrix_b, (4, 1)),
        ("C", matrix_c, (1, 4)),
        ("D", matrix_d, (1, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(f"NASA roll benchmark matrix {name} must have shape {shape}")
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"NASA roll benchmark matrix {name} must contain only finite real values"
            )
    if frequencies.shape != (5,) or not np.array_equal(
        frequencies, np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    ):
        raise ValueError(
            "NASA roll benchmark frequencies must contain the fixed ordered values"
        )
    if np.iscomplexobj(frequencies) or not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "NASA roll benchmark frequencies must contain only finite real values"
        )

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    flightlab_response = np.asarray(system.frequency_response(frequencies))
    reference_response = np.asarray(_nasa_unstable_roll_reference(frequencies))
    if flightlab_response.shape != (5, 1, 1):
        raise ValueError(
            "NASA roll benchmark FlightLab response must have shape (5, 1, 1)"
        )
    if reference_response.shape != (5,):
        raise ValueError("NASA roll benchmark reference response must have shape (5,)")
    if not np.iscomplexobj(flightlab_response) or not np.all(
        np.isfinite(flightlab_response)
    ):
        raise ValueError(
            "NASA roll benchmark FlightLab response must contain only finite complex values"
        )
    if not np.iscomplexobj(reference_response) or not np.all(
        np.isfinite(reference_response)
    ):
        raise ValueError(
            "NASA roll benchmark reference response must contain only finite complex values"
        )

    flightlab_response = flightlab_response[:, 0, 0]
    reference_magnitudes = np.abs(reference_response)
    if np.any(reference_magnitudes <= 0.0):
        raise ValueError(
            "NASA roll benchmark reference response must contain no zero values"
        )
    complex_residual = float(
        np.max(np.abs(flightlab_response - reference_response))
    )
    magnitude_residual = float(
        np.max(np.abs(np.abs(flightlab_response) - reference_magnitudes))
    )
    phase_residual = float(
        np.max(np.abs(np.angle(flightlab_response / reference_response)))
    )
    residuals = (complex_residual, magnitude_residual, phase_residual)
    if not all(np.isfinite(residual) for residual in residuals):
        raise ValueError("NASA roll benchmark residuals must contain only finite values")
    passed = bool(all(residual <= tolerance for residual in residuals))

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(4),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (4, 4),
            "B": tuple(matrix_b.flat),
            "B_shape": (4, 1),
            "C": tuple(matrix_c.flat),
            "C_shape": (1, 4),
            "D": tuple(matrix_d.flat),
            "D_shape": (1, 1),
            "canonical_realization": "phase_variable_controllable",
            "denominator_coefficients_ascending": (
                -0.0097081024500,
                0.4699353727332,
                0.706097742,
                0.85316,
                1.0,
            ),
            "input_identity": "sidestick input",
            "input_units": "deg",
            "name": "nasa_unstable_roll_frequency_response_v1",
            "numerator_coefficients_ascending": (
                0.458377088,
                0.170102400,
                0.78208,
                0.0,
                0.0,
            ),
            "output_identity": "roll attitude",
            "output_units": "deg",
        },
        controller={"type": "none"},
        reference={
            "aircraft_description": "mid-size twin-engine commercial transport",
            "airspeed_kt": 150.0,
            "aiaa_paper": "2015-0655",
            "altitude_ft": 41000.0,
            "authors": (
                "Peter M. T. Zaal",
                "Alexandru Popovici",
                "Melinda A. Zavala",
            ),
            "denominator_factors": (
                "s + 0.7599",
                "s - 0.02004",
                "s^2 + 0.1133 s + 0.6375",
            ),
            "equation": "1",
            "evidence_classification": (
                "computational/software verification against a published analytical aircraft model"
            ),
            "flight_condition_role": "source provenance, not validation evidence",
            "gross_weight_lb": 185800.0,
            "model_stability": "unstable",
            "near_stall": True,
            "ntrs_document_id": "20160008914",
            "numerator_factor": "0.78208 (s^2 + 0.2175 s + 0.5861)",
            "paper_page": 3,
            "publication_date": "2015-01-05",
            "section": "III.A.1 Controlled Aircraft Dynamics",
            "source_pdf_url": (
                "https://ntrs.nasa.gov/api/citations/20160008914/downloads/20160008914.pdf"
            ),
            "source_record_url": "https://ntrs.nasa.gov/citations/20160008914",
            "title": "Effects of False Tilt Cues on the Training of Manual Roll Control Skills",
        },
        user_metadata={
            "angular_frequencies_rad_per_s": tuple(frequencies),
            "angular_frequency_units": "rad/s",
            "auxiliary_response_carrier": True,
            "complex_response_tolerance": tolerance,
            "flightlab_response_imaginary": tuple(flightlab_response.imag),
            "flightlab_response_real": tuple(flightlab_response.real),
            "magnitude_tolerance": tolerance,
            "maximum_absolute_complex_response_residual": complex_residual,
            "maximum_absolute_magnitude_residual": magnitude_residual,
            "maximum_absolute_phase_residual_rad": phase_residual,
            "passed": passed,
            "phase_tolerance_rad": tolerance,
            "reference_response_imaginary": tuple(reference_response.imag),
            "reference_response_real": tuple(reference_response.real),
        },
        run_id="verification-nasa-unstable-roll-frequency-response-v1",
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
    )


def run_mathworks_controllability_verification_benchmark() -> ExperimentRun:
    """Verify MIMO controllability against an official worked example."""
    matrix_a = np.array([[1.0, 1.0], [4.0, -2.0]])
    matrix_b = np.array([[1.0, -1.0], [1.0, -1.0]])
    matrix_c = np.eye(2)
    matrix_d = np.zeros((2, 2))
    reference_ab = np.array([[2.0, -2.0], [2.0, -2.0]])
    reference_controllability = np.array(
        [[1.0, -1.0, 2.0, -2.0], [1.0, -1.0, 2.0, -2.0]]
    )
    matrix_tolerance = 1.0e-12

    for name, matrix, shape in (
        ("A", matrix_a, (2, 2)),
        ("B", matrix_b, (2, 2)),
        ("C", matrix_c, (2, 2)),
        ("D", matrix_d, (2, 2)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks controllability benchmark matrix {name} "
                f"must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks controllability benchmark matrix {name} "
                "must contain only finite real values"
            )

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    controllability = np.asarray(system.controllability_matrix())
    if controllability.shape != (2, 4):
        raise ValueError(
            "MathWorks benchmark controllability matrix must have shape (2, 4)"
        )
    if np.iscomplexobj(controllability) or not np.all(
        np.isfinite(controllability)
    ):
        raise ValueError(
            "MathWorks benchmark controllability matrix must contain only "
            "finite real values"
        )
    rank = system.controllability_rank()
    fully_controllable = system.is_fully_controllable()
    if isinstance(rank, (bool, np.bool_)) or not isinstance(
        rank, (int, np.integer)
    ):
        raise ValueError(  # noqa: TRY004 - fixed evidence contract uses ValueError
            "MathWorks benchmark controllability rank must be an integer"
        )
    rank = int(rank)
    uncontrollable_state_count = 2 - rank
    if rank < 0 or rank > 2 or not 0 <= uncontrollable_state_count <= 2:
        raise ValueError(
            "MathWorks benchmark controllability rank and count must be in range"
        )
    if not isinstance(fully_controllable, (bool, np.bool_)):
        raise ValueError(  # noqa: TRY004 - fixed evidence contract uses ValueError
            "MathWorks benchmark full-controllability result must be Boolean"
        )
    fully_controllable = bool(fully_controllable)
    if fully_controllable != (rank == 2):
        raise ValueError(
            "MathWorks benchmark controllability evidence is internally inconsistent"
        )

    matrix_residual = float(
        np.max(np.abs(controllability - reference_controllability))
    )
    if not np.isfinite(matrix_residual):
        raise ValueError(
            "MathWorks benchmark controllability residual must be finite"
        )
    rank_matches = rank == 1
    uncontrollable_state_count_matches = uncontrollable_state_count == 1
    classification_matches = fully_controllable is False
    passed = bool(
        matrix_residual <= matrix_tolerance
        and rank_matches
        and uncontrollable_state_count_matches
        and classification_matches
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(2),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (2, 2),
            "B": tuple(matrix_b.flat),
            "B_shape": (2, 2),
            "C": tuple(matrix_c.flat),
            "C_shape": (2, 2),
            "D": tuple(matrix_d.flat),
            "D_shape": (2, 2),
            "auxiliary_output_matrices": True,
            "input_count": 2,
            "physical_units": "none assigned",
            "state_count": 2,
        },
        controller={"type": "none"},
        reference={
            "access_date": "2026-09-03",
            "controllability_formula": "Co = [B, A B, ..., A^(n-1) B]",
            "evidence_classification": (
                "computational/software verification against an official "
                "worked control-systems example and exact algebraic oracle"
            ),
            "product_documentation": "MathWorks Control System Toolbox Documentation",
            "published_conclusion": "system is not controllable",
            "published_full_controllability": False,
            "published_uncontrollable_state_count": 1,
            "section": "Check System Controllability",
            "source_title": "ctrb - Controllability of state-space model",
            "source_url": (
                "https://www.mathworks.com/help/control/ref/statespacemodel.ctrb.html"
            ),
        },
        user_metadata={
            "auxiliary_response_carrier": True,
            "classification_matches": classification_matches,
            "controllability_matrix": tuple(controllability.flat),
            "controllability_matrix_shape": (2, 4),
            "controllability_rank": rank,
            "fully_controllable": fully_controllable,
            "matrix_residual_tolerance": matrix_tolerance,
            "maximum_absolute_controllability_matrix_residual": matrix_residual,
            "passed": passed,
            "rank_matches": rank_matches,
            "reference_A_B": tuple(reference_ab.flat),
            "reference_controllability_matrix": tuple(
                reference_controllability.flat
            ),
            "reference_controllability_matrix_shape": (2, 4),
            "reference_rank": 1,
            "uncontrollable_state_count": uncontrollable_state_count,
            "uncontrollable_state_count_matches": (
                uncontrollable_state_count_matches
            ),
        },
        run_id="verification-mathworks-controllability-rank-v1",
        created_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
    )


def run_mathworks_siso_pole_placement_verification_benchmark() -> ExperimentRun:
    """Verify SISO pole placement against an official worked example."""
    matrix_a = np.array([[-1.0, -2.0], [1.0, 0.0]])
    matrix_b = np.array([[2.0], [0.0]])
    matrix_c = np.array([[0.0, 1.0]])
    matrix_d = np.array([[0.0]])
    desired_poles = np.array([-1.0, -2.0])
    reference_gain = np.array([[1.0, 0.0]])
    reference_closed_loop_a = np.array([[-3.0, -2.0], [1.0, 0.0]])
    reference_poles = np.array([-2.0, -1.0])
    gain_tolerance = 1.0e-12
    state_matrix_tolerance = 1.0e-12
    preservation_tolerance = 0.0
    pole_tolerance = 1.0e-12

    for name, matrix, shape in (
        ("A", matrix_a, (2, 2)),
        ("B", matrix_b, (2, 1)),
        ("C", matrix_c, (1, 2)),
        ("D", matrix_d, (1, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks pole-placement benchmark matrix {name} "
                f"must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks pole-placement benchmark matrix {name} "
                "must contain only finite real values"
            )
    if desired_poles.shape != (2,) or not np.array_equal(
        desired_poles, np.array([-1.0, -2.0])
    ):
        raise ValueError(
            "MathWorks pole-placement benchmark desired poles must contain "
            "the fixed ordered values"
        )
    if np.iscomplexobj(desired_poles) or not np.all(np.isfinite(desired_poles)):
        raise ValueError(
            "MathWorks pole-placement benchmark desired poles must contain "
            "only finite real values"
        )

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    gain = np.asarray(system.place_siso_poles(desired_poles))
    if gain.shape != (1, 2):
        raise ValueError(
            "MathWorks pole-placement benchmark gain must have shape (1, 2)"
        )
    if np.iscomplexobj(gain) or not np.all(np.isfinite(gain)):
        raise ValueError(
            "MathWorks pole-placement benchmark gain must contain only "
            "finite real values"
        )

    closed_loop = system.full_state_feedback(gain)
    for name, matrix, shape in (
        ("A", np.asarray(closed_loop.A), (2, 2)),
        ("B", np.asarray(closed_loop.B), (2, 1)),
        ("C", np.asarray(closed_loop.C), (1, 2)),
        ("D", np.asarray(closed_loop.D), (1, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks pole-placement benchmark closed-loop matrix {name} "
                f"must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks pole-placement benchmark closed-loop matrix {name} "
                "must contain only finite real values"
            )

    expected_closed_loop_a = np.array(
        [
            [-1.0 - 2.0 * gain[0, 0], -2.0 - 2.0 * gain[0, 1]],
            [1.0, 0.0],
        ]
    )
    closed_loop_consistent = bool(
        np.array_equal(closed_loop.A, expected_closed_loop_a)
    )
    if not closed_loop_consistent:
        raise ValueError(
            "MathWorks pole-placement benchmark gain and closed-loop A "
            "are internally inconsistent"
        )

    achieved_poles = np.asarray(closed_loop.eigenvalues())
    if achieved_poles.shape != (2,):
        raise ValueError(
            "MathWorks pole-placement benchmark achieved poles must have shape (2,)"
        )
    if not np.all(np.isfinite(achieved_poles)):
        raise ValueError(
            "MathWorks pole-placement benchmark achieved poles must be finite"
        )
    if np.iscomplexobj(achieved_poles) and np.any(achieved_poles.imag != 0.0):
        raise ValueError(
            "MathWorks pole-placement benchmark achieved poles must be real"
        )
    achieved_poles = np.sort(np.asarray(achieved_poles.real, dtype=float))

    gain_residual = float(np.max(np.abs(gain - reference_gain)))
    state_matrix_residual = float(
        np.max(np.abs(closed_loop.A - reference_closed_loop_a))
    )
    preservation_residual = float(
        max(
            np.max(np.abs(closed_loop.B - matrix_b)),
            np.max(np.abs(closed_loop.C - matrix_c)),
            np.max(np.abs(closed_loop.D - matrix_d)),
        )
    )
    pole_residual = float(np.max(np.abs(achieved_poles - reference_poles)))
    residuals = (
        gain_residual,
        state_matrix_residual,
        preservation_residual,
        pole_residual,
    )
    if not all(np.isfinite(residual) for residual in residuals):
        raise ValueError(
            "MathWorks pole-placement benchmark residuals must be finite"
        )
    desired_poles_achieved = bool(np.array_equal(achieved_poles, reference_poles))
    asymptotically_stable = bool(np.all(achieved_poles < 0.0))
    matrices_preserved = preservation_residual <= preservation_tolerance
    passed = bool(
        gain_residual <= gain_tolerance
        and state_matrix_residual <= state_matrix_tolerance
        and preservation_residual <= preservation_tolerance
        and pole_residual <= pole_tolerance
        and desired_poles_achieved
        and asymptotically_stable
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(2),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (2, 2),
            "B": tuple(matrix_b.flat),
            "B_shape": (2, 1),
            "C": tuple(matrix_c.flat),
            "C_shape": (1, 2),
            "D": tuple(matrix_d.flat),
            "D_shape": (1, 1),
            "input_count": 1,
            "output_count": 1,
            "physical_units": "none assigned",
            "state_count": 2,
        },
        controller={
            "closed_loop_convention": "A_cl = A - B K",
            "desired_poles_input_order": tuple(desired_poles),
            "feedback_convention": "u = -K x",
            "gain": tuple(gain.flat),
            "type": "static_full_state_feedback",
        },
        reference={
            "access_date": "2026-09-03",
            "coefficient_matching": (
                "1 + 2 k1 = 3; 2 + 2 k2 = 2; k1 = 1; k2 = 0"
            ),
            "evidence_classification": (
                "computational/software verification of deterministic controller "
                "synthesis and feedback interconnection against an official worked "
                "example and exact algebraic oracle"
            ),
            "product_documentation": "MathWorks Control System Toolbox Documentation",
            "reference_closed_loop_A": tuple(reference_closed_loop_a.flat),
            "reference_gain": tuple(reference_gain.flat),
            "reference_poles_sorted": tuple(reference_poles),
            "release_reference_pdf_url": (
                "https://www.mathworks.com/help/releases/r2024b/pdf_doc/control/control_ref.pdf"
            ),
            "section": "Pole Placement Design for Second-Order System",
            "source_conclusion": "closed-loop system is stable and nonoscillatory",
            "source_title": "place - Pole placement design",
            "source_url": "https://www.mathworks.com/help/control/ref/place.html",
            "target_characteristic_polynomial": "(s + 1)(s + 2) = s^2 + 3 s + 2",
        },
        user_metadata={
            "achieved_poles_sorted": tuple(achieved_poles),
            "asymptotically_stable": asymptotically_stable,
            "auxiliary_response_carrier": True,
            "closed_loop_A": tuple(closed_loop.A.flat),
            "closed_loop_B": tuple(closed_loop.B.flat),
            "closed_loop_C": tuple(closed_loop.C.flat),
            "closed_loop_D": tuple(closed_loop.D.flat),
            "closed_loop_consistent": closed_loop_consistent,
            "desired_poles_achieved": desired_poles_achieved,
            "gain_residual_tolerance": gain_tolerance,
            "matrices_preserved": matrices_preserved,
            "maximum_absolute_closed_loop_state_matrix_residual": (
                state_matrix_residual
            ),
            "maximum_absolute_gain_residual": gain_residual,
            "maximum_absolute_preserved_realization_residual": (
                preservation_residual
            ),
            "maximum_absolute_achieved_pole_residual": pole_residual,
            "passed": passed,
            "pole_residual_tolerance": pole_tolerance,
            "preserved_realization_tolerance": preservation_tolerance,
            "state_matrix_residual_tolerance": state_matrix_tolerance,
        },
        run_id="verification-mathworks-siso-pole-placement-v1",
        created_at=datetime(2026, 9, 4, tzinfo=UTC),
    )


def run_scipy_manual_controllability_gramian_verification_benchmark() -> ExperimentRun:
    """Verify a controllability Gramian against SciPy's published example."""
    matrix_a = np.array(
        [[-3.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [0.0, -5.0, -1.0]]
    )
    matrix_b = np.eye(3)
    matrix_c = np.eye(3)
    matrix_d = np.zeros((3, 3))
    source_q = np.eye(3)
    published_x = np.array(
        [
            [-0.75, 0.875, -3.75],
            [0.875, -1.375, 5.3125],
            [-3.75, 5.3125, -27.0625],
        ]
    )
    reference_gramian = np.array(
        [
            [0.75, -0.875, 3.75],
            [-0.875, 1.375, -5.3125],
            [3.75, -5.3125, 27.0625],
        ]
    )
    reference_principal_minors = np.array([0.75, 0.265625, 1.548828125])
    tolerance = 1.0e-12

    for name, matrix in (
        ("A", matrix_a),
        ("B", matrix_b),
        ("C", matrix_c),
        ("D", matrix_d),
    ):
        if matrix.shape != (3, 3):
            raise ValueError(
                f"SciPy Gramian benchmark matrix {name} must have shape (3, 3)"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"SciPy Gramian benchmark matrix {name} must contain only "
                "finite real values"
            )

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    gramian = np.asarray(system.controllability_gramian())
    if gramian.shape != (3, 3):
        raise ValueError("SciPy Gramian benchmark result must have shape (3, 3)")
    if np.iscomplexobj(gramian) or not np.all(np.isfinite(gramian)):
        raise ValueError(
            "SciPy Gramian benchmark result must contain only finite real values"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        lyapunov_left_hand_side = (
            matrix_a @ gramian + gramian @ matrix_a.T + matrix_b @ matrix_b.T
        )
        symmetry_difference = gramian - gramian.T
        a, b, c = gramian[0]
        d, e, f = gramian[1]
        g, h, i = gramian[2]
        principal_minors = np.array(
            [
                a,
                a * e - b * d,
                a * (e * i - f * h)
                - b * (d * i - f * g)
                + c * (d * h - e * g),
            ]
        )
    if not np.all(np.isfinite(principal_minors)):
        raise ValueError("SciPy Gramian benchmark principal minors must be finite")

    gramian_residual = float(np.max(np.abs(gramian - reference_gramian)))
    lyapunov_residual = float(np.max(np.abs(lyapunov_left_hand_side)))
    symmetry_residual = float(np.max(np.abs(symmetry_difference)))
    principal_minor_residual = float(
        np.max(np.abs(principal_minors - reference_principal_minors))
    )
    residuals = (
        gramian_residual,
        lyapunov_residual,
        symmetry_residual,
        principal_minor_residual,
    )
    if not all(np.isfinite(residual) for residual in residuals):
        raise ValueError("SciPy Gramian benchmark residuals must be finite")

    positive_definite = bool(np.all(principal_minors > 0.0))
    passed = bool(
        gramian_residual <= tolerance
        and lyapunov_residual <= tolerance
        and symmetry_residual <= tolerance
        and principal_minor_residual <= tolerance
        and positive_definite
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    if metrics.maximum_absolute_tracking_error != 0.0:
        raise ValueError("SciPy Gramian benchmark carrier evidence is inconsistent")

    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(3),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (3, 3),
            "B": tuple(matrix_b.flat),
            "B_shape": (3, 3),
            "C": tuple(matrix_c.flat),
            "C_shape": (3, 3),
            "D": tuple(matrix_d.flat),
            "D_shape": (3, 3),
            "auxiliary_output_matrices": True,
            "input_count": 3,
            "output_count": 3,
            "physical_units": "none assigned",
            "state_count": 3,
        },
        controller={"type": "none"},
        reference={
            "access_date": "2026-09-03",
            "algorithm": "Bartels-Stewart algorithm",
            "evidence_classification": (
                "computational/software verification of a continuous-time "
                "controllability Gramian and Lyapunov equation against an "
                "official worked example, exact sign mapping, and literal "
                "algebraic oracle"
            ),
            "example_title": "Given a and q solve for x",
            "flightlab_equation": "A Wc + Wc A^T + B B^T = 0",
            "manual": "SciPy v1.18.0 Manual",
            "published_A": tuple(matrix_a.flat),
            "published_A_shape": (3, 3),
            "published_Q": tuple(source_q.flat),
            "published_Q_shape": (3, 3),
            "published_X": tuple(published_x.flat),
            "published_X_shape": (3, 3),
            "published_equation_check_result": tuple(source_q.flat),
            "published_equation_convention": "A X + X A^H = Q",
            "reference_gramian": tuple(reference_gramian.flat),
            "reference_gramian_shape": (3, 3),
            "sign_mapping": "Wc_reference = -X_published; B B^T = Q = I",
            "source_function": "scipy.linalg.solve_continuous_lyapunov",
            "source_url": (
                "https://docs.scipy.org/doc/scipy-1.18.0/reference/generated/"
                "scipy.linalg.solve_continuous_lyapunov.html"
            ),
        },
        user_metadata={
            "auxiliary_response_carrier": True,
            "gramian_residual_tolerance": tolerance,
            "leading_principal_minor_residual_tolerance": tolerance,
            "lyapunov_equation_residual_tolerance": tolerance,
            "lyapunov_left_hand_side": tuple(lyapunov_left_hand_side.flat),
            "maximum_absolute_gramian_residual": gramian_residual,
            "maximum_absolute_leading_principal_minor_residual": (
                principal_minor_residual
            ),
            "maximum_absolute_lyapunov_equation_residual": lyapunov_residual,
            "maximum_absolute_symmetry_residual": symmetry_residual,
            "passed": passed,
            "positive_definite": positive_definite,
            "reference_leading_principal_minors": tuple(
                reference_principal_minors
            ),
            "returned_controllability_gramian": tuple(gramian.flat),
            "returned_controllability_gramian_shape": (3, 3),
            "returned_leading_principal_minors": tuple(principal_minors),
            "symmetry_difference": tuple(symmetry_difference.flat),
            "symmetry_residual_tolerance": tolerance,
        },
        run_id="verification-scipy-manual-controllability-gramian-v1",
        created_at=datetime(2026, 9, 4, 12, tzinfo=UTC),
    )


def run_mathworks_balanced_truncation_verification_benchmark() -> ExperimentRun:
    """Verify fixed balanced truncation against exact analytical oracles."""
    matrix_a = np.diag([-1.0, -2.0])
    matrix_b = np.diag([2.0, 1.0])
    matrix_c = np.diag([2.0, 1.0])
    matrix_d = np.zeros((2, 2))
    retained_order = 1
    frequencies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    reference_reduced_a = np.array([[-1.0]])
    reference_reduced_b = np.array([[2.0, 0.0]])
    reference_reduced_c = np.array([[2.0], [0.0]])
    reference_reduced_d = np.zeros((2, 2))
    reference_projection = np.array([[1.0, 0.0]])
    reference_reconstruction = np.array([[1.0], [0.0]])
    reference_transformation = np.eye(2)
    reference_retained_hsv = np.array([2.0])
    reference_discarded_hsv = np.array([0.25])
    reference_error_bound = 0.5
    reference_error_singular_values = np.column_stack(
        (1.0 / np.sqrt(frequencies * frequencies + 4.0), np.zeros(5))
    )
    tolerance = 1.0e-12

    for name, matrix, shape in (
        ("A", matrix_a, (2, 2)),
        ("B", matrix_b, (2, 2)),
        ("C", matrix_c, (2, 2)),
        ("D", matrix_d, (2, 2)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks balanced-truncation matrix {name} must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks balanced-truncation matrix {name} must contain only "
                "finite real values"
            )
    if frequencies.shape != (5,) or not np.array_equal(
        frequencies, np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    ):
        raise ValueError(
            "MathWorks balanced-truncation frequencies must contain five fixed values"
        )

    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    truncation = system.balanced_truncation(retained_order)
    if not isinstance(truncation, BalancedTruncation):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks balanced-truncation result must be a BalancedTruncation"
        )
    if isinstance(truncation.retained_order, (bool, np.bool_)) or not isinstance(
        truncation.retained_order, (int, np.integer)
    ):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks balanced-truncation retained order must be an integer"
        )
    if int(truncation.retained_order) != retained_order:
        raise ValueError(
            "MathWorks balanced-truncation retained order must equal one"
        )

    reduced = truncation.system
    if not isinstance(reduced, StateSpace):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks balanced-truncation reduced system must be a StateSpace"
        )
    returned_arrays = {
        "reduced A": (np.asarray(reduced.A), (1, 1)),
        "reduced B": (np.asarray(reduced.B), (1, 2)),
        "reduced C": (np.asarray(reduced.C), (2, 1)),
        "reduced D": (np.asarray(reduced.D), (2, 2)),
        "projection": (np.asarray(truncation.projection), (1, 2)),
        "reconstruction": (np.asarray(truncation.reconstruction), (2, 1)),
        "balanced transformation": (
            np.asarray(truncation.balanced_transformation),
            (2, 2),
        ),
        "retained Hankel singular values": (
            np.asarray(truncation.retained_hankel_singular_values),
            (1,),
        ),
        "discarded Hankel singular values": (
            np.asarray(truncation.discarded_hankel_singular_values),
            (1,),
        ),
    }
    for name, (values, shape) in returned_arrays.items():
        if values.shape != shape:
            raise ValueError(
                f"MathWorks balanced-truncation {name} must have shape {shape}"
            )
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"MathWorks balanced-truncation {name} must contain only "
                "finite real values"
            )

    retained_hsv = returned_arrays["retained Hankel singular values"][0]
    discarded_hsv = returned_arrays["discarded Hankel singular values"][0]
    returned_hsv = np.concatenate((retained_hsv, discarded_hsv))
    if np.any(returned_hsv < 0.0):
        raise ValueError(
            "MathWorks balanced-truncation Hankel singular values must be nonnegative"
        )
    if np.any(returned_hsv[:-1] < returned_hsv[1:]):
        raise ValueError(
            "MathWorks balanced-truncation Hankel singular values must be ordered"
        )

    error_bound = truncation.a_priori_error_bound
    if isinstance(error_bound, (bool, np.bool_)) or not np.isscalar(error_bound):
        raise ValueError(
            "MathWorks balanced-truncation error bound must be a real scalar"
        )
    if np.iscomplexobj(error_bound) or not np.isfinite(error_bound):
        raise ValueError(
            "MathWorks balanced-truncation error bound must be finite and real"
        )
    error_bound = float(error_bound)
    if error_bound < 0.0:
        raise ValueError(
            "MathWorks balanced-truncation error bound must be nonnegative"
        )

    sampled_error_singular_values = np.asarray(
        system.balanced_truncation_frequency_response_error_singular_values(
            retained_order, frequencies
        )
    )
    if sampled_error_singular_values.shape != (5, 2):
        raise ValueError(
            "MathWorks balanced-truncation sampled singular values must have "
            "shape (5, 2)"
        )
    if np.iscomplexobj(sampled_error_singular_values) or not np.all(
        np.isfinite(sampled_error_singular_values)
    ):
        raise ValueError(
            "MathWorks balanced-truncation sampled singular values must contain "
            "only finite real values"
        )
    if np.any(sampled_error_singular_values < 0.0):
        raise ValueError(
            "MathWorks balanced-truncation sampled singular values must be nonnegative"
        )
    if np.any(
        sampled_error_singular_values[:, :-1]
        < sampled_error_singular_values[:, 1:]
    ):
        raise ValueError(
            "MathWorks balanced-truncation sampled singular values must be ordered"
        )

    realization_residual = float(
        max(
            np.max(np.abs(reduced.A - reference_reduced_a)),
            np.max(np.abs(reduced.B - reference_reduced_b)),
            np.max(np.abs(reduced.C - reference_reduced_c)),
            np.max(np.abs(reduced.D - reference_reduced_d)),
        )
    )
    coordinate_map_residual = float(
        max(
            np.max(np.abs(truncation.projection - reference_projection)),
            np.max(np.abs(truncation.reconstruction - reference_reconstruction)),
            np.max(
                np.abs(
                    truncation.balanced_transformation
                    - reference_transformation
                )
            ),
        )
    )
    hankel_residual = float(
        max(
            np.max(np.abs(retained_hsv - reference_retained_hsv)),
            np.max(np.abs(discarded_hsv - reference_discarded_hsv)),
        )
    )
    error_bound_residual = float(abs(error_bound - reference_error_bound))
    sampled_error_residual = float(
        np.max(
            np.abs(
                sampled_error_singular_values
                - reference_error_singular_values
            )
        )
    )
    bound_margin = float(
        reference_error_bound - np.max(sampled_error_singular_values)
    )
    residuals_and_margin = (
        realization_residual,
        coordinate_map_residual,
        hankel_residual,
        error_bound_residual,
        sampled_error_residual,
        bound_margin,
    )
    if not all(np.isfinite(value) for value in residuals_and_margin):
        raise ValueError(
            "MathWorks balanced-truncation residuals and margin must be finite"
        )

    all_samples_within_bound = bool(
        np.all(sampled_error_singular_values <= error_bound + tolerance)
    )
    dc_attains_bound = bool(
        abs(sampled_error_singular_values[0, 0] - error_bound) <= tolerance
    )
    passed = bool(
        realization_residual <= tolerance
        and coordinate_map_residual <= tolerance
        and hankel_residual <= tolerance
        and error_bound_residual <= tolerance
        and sampled_error_residual <= tolerance
        and all_samples_within_bound
        and dc_attains_bound
        and bound_margin >= -tolerance
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    if metrics.maximum_absolute_tracking_error != 0.0:
        raise ValueError(
            "MathWorks balanced-truncation carrier evidence is inconsistent"
        )

    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(2),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (2, 2),
            "B": tuple(matrix_b.flat),
            "B_shape": (2, 2),
            "C": tuple(matrix_c.flat),
            "C_shape": (2, 2),
            "D": tuple(matrix_d.flat),
            "D_shape": (2, 2),
            "input_count": 2,
            "output_count": 2,
            "physical_units": "none assigned",
            "state_count": 2,
        },
        controller={"type": "none"},
        reference={
            "absolute_error_algorithm_statement": (
                "absolute-error balanced truncation reduces the stable part"
            ),
            "absolute_error_bound_formula": (
                "||G_s - G_r||_infinity <= 2 * sum(sigma_j, j=r+1,...,n)"
            ),
            "access_date": "2026-09-03",
            "evidence_classification": (
                "computational/software verification of continuous-time balanced "
                "truncation, Hankel-energy ordering, reduced realization, and the "
                "absolute input/output error bound against an authoritative "
                "algorithm contract and exact analytical oracles"
            ),
            "full_transfer_matrix": "diag([4/(s+1), 1/(s+2)])",
            "hsv_description": (
                "state contributions to input/output behavior in balanced coordinates"
            ),
            "info_error_bound_description": (
                "bounds the absolute approximation error for the retained order"
            ),
            "page_sections": ("Description", "Output Arguments: info", "Algorithms"),
            "page_title": "balred - (Not recommended) Model order reduction",
            "product_documentation": "MathWorks Control System Toolbox Documentation",
            "reduced_transfer_matrix": "diag([4/(s+1), 0])",
            "scalar_gramian_integral": (
                "integral_0^infinity exp(-2 a t) q^2 dt = q^2 / (2 a)"
            ),
            "source_url": (
                "https://www.mathworks.com/help/control/ref/"
                "dynamicsystem.balred.html"
            ),
            "transfer_error_matrix": "diag([0, 1/(s+2)])",
        },
        user_metadata={
            "absolute_a_priori_error_bound_reference": reference_error_bound,
            "all_sampled_errors_within_bound": all_samples_within_bound,
            "angular_frequencies_rad_per_s": tuple(frequencies),
            "angular_frequency_units": "rad/s",
            "auxiliary_response_carrier": True,
            "bound_comparison_tolerance": tolerance,
            "bound_satisfaction_margin": bound_margin,
            "coordinate_map_residual_tolerance": tolerance,
            "dc_error_attains_returned_bound": dc_attains_bound,
            "error_bound_residual_tolerance": tolerance,
            "hankel_evidence_residual_tolerance": tolerance,
            "maximum_absolute_coordinate_map_residual": coordinate_map_residual,
            "maximum_absolute_error_bound_residual": error_bound_residual,
            "maximum_absolute_hankel_evidence_residual": hankel_residual,
            "maximum_absolute_reduction_realization_residual": (
                realization_residual
            ),
            "maximum_absolute_sampled_error_singular_value_residual": (
                sampled_error_residual
            ),
            "passed": passed,
            "reconstruction": tuple(truncation.reconstruction.flat),
            "reconstruction_shape": (2, 1),
            "reconstruction_reference": tuple(reference_reconstruction.flat),
            "reduced_A": tuple(reduced.A.flat),
            "reduced_A_shape": (1, 1),
            "reduced_A_reference": tuple(reference_reduced_a.flat),
            "reduced_B": tuple(reduced.B.flat),
            "reduced_B_shape": (1, 2),
            "reduced_B_reference": tuple(reference_reduced_b.flat),
            "reduced_C": tuple(reduced.C.flat),
            "reduced_C_shape": (2, 1),
            "reduced_C_reference": tuple(reference_reduced_c.flat),
            "reduced_D": tuple(reduced.D.flat),
            "reduced_D_shape": (2, 2),
            "reduced_D_reference": tuple(reference_reduced_d.flat),
            "reduction_realization_residual_tolerance": tolerance,
            "retained_order": int(truncation.retained_order),
            "returned_a_priori_error_bound": error_bound,
            "returned_balanced_transformation": tuple(
                truncation.balanced_transformation.flat
            ),
            "returned_balanced_transformation_shape": (2, 2),
            "returned_discarded_hankel_singular_values": tuple(discarded_hsv),
            "returned_discarded_hankel_singular_values_shape": (1,),
            "returned_projection": tuple(truncation.projection.flat),
            "returned_projection_shape": (1, 2),
            "returned_retained_hankel_singular_values": tuple(retained_hsv),
            "returned_retained_hankel_singular_values_shape": (1,),
            "sampled_error_singular_value_reference": tuple(
                reference_error_singular_values.flat
            ),
            "sampled_error_singular_value_residual_tolerance": tolerance,
            "sampled_error_singular_values": tuple(
                sampled_error_singular_values.flat
            ),
            "sampled_error_singular_values_shape": (5, 2),
            "transformation_reference": tuple(reference_transformation.flat),
            "projection_reference": tuple(reference_projection.flat),
            "hankel_singular_values_reference": (2.0, 0.25),
            "controllability_gramian_reference": (2.0, 0.0, 0.0, 0.25),
            "observability_gramian_reference": (2.0, 0.0, 0.0, 0.25),
        },
        run_id="verification-mathworks-balanced-truncation-v1",
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
    )


def run_mathworks_luenberger_observer_verification_benchmark() -> ExperimentRun:
    """Verify observer synthesis and interconnection against MathWorks' example."""
    matrix_a = np.array([[-1.0, -0.75], [1.0, 0.0]])
    matrix_b = np.array([[1.0], [0.0]])
    matrix_c = np.array([[1.0, 1.0]])
    matrix_d = np.array([[0.0]])
    desired_poles = np.array([-2.0, -3.0])
    reference_gain = np.array([[17.0 / 3.0], [-5.0 / 3.0]])
    reference_error_a = np.array(
        [[-20.0 / 3.0, -77.0 / 12.0], [8.0 / 3.0, 5.0 / 3.0]]
    )
    reference_coefficients = np.array([1.0, 5.0, 6.0])
    reference_poles = np.array([-3.0, -2.0])
    reference_augmented_a = np.array(
        [
            [-1.0, -3.0 / 4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [17.0 / 3.0, 17.0 / 3.0, -20.0 / 3.0, -77.0 / 12.0],
            [-5.0 / 3.0, -5.0 / 3.0, 8.0 / 3.0, 5.0 / 3.0],
        ]
    )
    reference_augmented_b = np.array([[1.0], [0.0], [1.0], [0.0]])
    reference_augmented_c = np.array(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    reference_augmented_d = np.zeros((3, 1))
    numerical_tolerance = 1.0e-12
    exact_tolerance = 0.0

    for name, matrix, shape in (
        ("A", matrix_a, (2, 2)),
        ("B", matrix_b, (2, 1)),
        ("C", matrix_c, (1, 2)),
        ("D", matrix_d, (1, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks observer benchmark matrix {name} must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks observer benchmark matrix {name} must contain only "
                "finite real values"
            )
    if desired_poles.shape != (2,) or not np.array_equal(
        desired_poles, np.array([-2.0, -3.0])
    ):
        raise ValueError(
            "MathWorks observer benchmark desired poles must contain the fixed order"
        )

    source_copies = tuple(
        matrix.copy() for matrix in (matrix_a, matrix_b, matrix_c, matrix_d)
    )
    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    gain = np.asarray(system.place_siso_observer_poles(desired_poles))
    if gain.shape != (2, 1):
        raise ValueError("MathWorks observer benchmark gain must have shape (2, 1)")
    if np.iscomplexobj(gain) or not np.all(np.isfinite(gain)):
        raise ValueError(
            "MathWorks observer benchmark gain must contain only finite real values"
        )

    interconnection = system.luenberger_observer(gain)
    if not isinstance(interconnection, LuenbergerObserverInterconnection):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks observer benchmark result must be a "
            "LuenbergerObserverInterconnection"
        )
    stored_gain = np.asarray(interconnection.observer_gain)
    if stored_gain.shape != (2, 1):
        raise ValueError(
            "MathWorks observer benchmark stored gain must have shape (2, 1)"
        )
    if np.iscomplexobj(stored_gain) or not np.all(np.isfinite(stored_gain)):
        raise ValueError(
            "MathWorks observer benchmark stored gain must contain only finite "
            "real values"
        )
    augmented = interconnection.system
    if not isinstance(augmented, StateSpace):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks observer benchmark augmented system must be a StateSpace"
        )
    for name, matrix, shape in (
        ("A", np.asarray(augmented.A), (4, 4)),
        ("B", np.asarray(augmented.B), (4, 1)),
        ("C", np.asarray(augmented.C), (3, 4)),
        ("D", np.asarray(augmented.D), (3, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks observer benchmark augmented {name} must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks observer benchmark augmented {name} must contain only "
                "finite real values"
            )

    source_preservation_residual = float(
        max(
            np.max(np.abs(matrix - original))
            for matrix, original in zip(
                (system.A, system.B, system.C, system.D),
                source_copies,
                strict=True,
            )
        )
    )
    if source_preservation_residual != exact_tolerance:
        raise ValueError("MathWorks observer benchmark source matrices were mutated")

    stored_correction = stored_gain @ matrix_c
    stored_error_a = matrix_a - stored_correction
    expected_augmented_a = np.block(
        [
            [matrix_a, np.zeros((2, 2))],
            [stored_correction, stored_error_a],
        ]
    )
    expected_augmented_b = np.vstack((matrix_b, matrix_b))
    expected_augmented_c = np.block(
        [[matrix_c, np.zeros((1, 2))], [np.zeros((2, 2)), np.eye(2)]]
    )
    expected_augmented_d = np.vstack((matrix_d, np.zeros((2, 1))))
    if not all(
        np.array_equal(actual, expected)
        for actual, expected in (
            (augmented.A, expected_augmented_a),
            (augmented.B, expected_augmented_b),
            (augmented.C, expected_augmented_c),
            (augmented.D, expected_augmented_d),
        )
    ):
        raise ValueError(
            "MathWorks observer benchmark interconnection is internally inconsistent"
        )

    error_a = matrix_a - gain @ matrix_c
    trace = float(error_a[0, 0] + error_a[1, 1])
    determinant = float(
        error_a[0, 0] * error_a[1, 1]
        - error_a[0, 1] * error_a[1, 0]
    )
    coefficients = np.array([1.0, -trace, determinant])
    discriminant = float(coefficients[1] ** 2 - 4.0 * coefficients[2])
    if not all(
        np.isfinite(value)
        for value in (*coefficients, trace, determinant, discriminant)
    ):
        raise ValueError(
            "MathWorks observer benchmark derived coefficients must be finite"
        )
    if discriminant < 0.0:
        raise ValueError(
            "MathWorks observer benchmark quadratic discriminant must be nonnegative"
        )
    square_root_discriminant = float(np.sqrt(discriminant))
    achieved_poles = np.sort(
        np.array(
            [
                (-coefficients[1] - square_root_discriminant) / 2.0,
                (-coefficients[1] + square_root_discriminant) / 2.0,
            ]
        )
    )
    if not np.all(np.isfinite(achieved_poles)):
        raise ValueError("MathWorks observer benchmark achieved poles must be finite")

    gain_residual = float(np.max(np.abs(gain - reference_gain)))
    gain_preservation_residual = float(np.max(np.abs(gain - stored_gain)))
    error_a_residual = float(np.max(np.abs(error_a - reference_error_a)))
    coefficient_residual = float(
        np.max(np.abs(coefficients - reference_coefficients))
    )
    pole_residual = float(np.max(np.abs(achieved_poles - reference_poles)))
    augmented_residual = float(
        max(
            np.max(np.abs(augmented.A - reference_augmented_a)),
            np.max(np.abs(augmented.B - reference_augmented_b)),
            np.max(np.abs(augmented.C - reference_augmented_c)),
            np.max(np.abs(augmented.D - reference_augmented_d)),
        )
    )
    residuals = (
        gain_residual,
        gain_preservation_residual,
        error_a_residual,
        coefficient_residual,
        pole_residual,
        augmented_residual,
        source_preservation_residual,
    )
    if not all(np.isfinite(residual) for residual in residuals):
        raise ValueError("MathWorks observer benchmark residuals must be finite")

    discriminant_nonnegative = discriminant >= 0.0
    error_dynamics_stable = bool(np.all(achieved_poles < 0.0))
    source_matrices_preserved = source_preservation_residual <= exact_tolerance
    passed = bool(
        gain_residual <= numerical_tolerance
        and gain_preservation_residual <= exact_tolerance
        and error_a_residual <= numerical_tolerance
        and coefficient_residual <= numerical_tolerance
        and pole_residual <= numerical_tolerance
        and augmented_residual <= numerical_tolerance
        and source_matrices_preserved
        and discriminant_nonnegative
        and error_dynamics_stable
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    if metrics.maximum_absolute_tracking_error != 0.0:
        raise ValueError("MathWorks observer carrier evidence is inconsistent")

    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(2),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (2, 2),
            "B": tuple(matrix_b.flat),
            "B_shape": (2, 1),
            "C": tuple(matrix_c.flat),
            "C_shape": (1, 2),
            "D": tuple(matrix_d.flat),
            "D_shape": (1, 1),
            "input_count": 1,
            "output_count": 1,
            "physical_units": "none assigned",
            "state_count": 2,
        },
        controller={
            "desired_observer_poles_input_order": tuple(desired_poles),
            "observer_gain": tuple(gain.flat),
            "type": "full_order_luenberger_observer",
        },
        reference={
            "access_date": "2026-09-03",
            "duality_statement": (
                "transpose A and substitute C' for B in pole placement"
            ),
            "evidence_classification": (
                "computational/software verification of deterministic full-order "
                "state-observer synthesis, estimation-error pole assignment, and "
                "Luenberger plant/observer interconnection against an authoritative "
                "worked example and exact algebraic oracles"
            ),
            "example_title": "Pole Placement Observer Design",
            "gain_command": "L = place(A',C',[-2,-3])'",
            "only_output_measured": True,
            "observer_matrix_commands": (
                "At = A-L*C",
                "Bt = [B,L]",
                "Ct = [C;eye(2)]",
            ),
            "page_title": "place - Pole placement design",
            "product_documentation": "MathWorks Control System Toolbox Documentation",
            "published_A": tuple(matrix_a.flat),
            "published_B": tuple(matrix_b.flat),
            "published_C": tuple(matrix_c.flat),
            "published_D": tuple(matrix_d.flat),
            "published_desired_observer_poles": tuple(desired_poles),
            "source_url": "https://www.mathworks.com/help/control/ref/place.html",
        },
        user_metadata={
            "achieved_error_poles_sorted": tuple(achieved_poles),
            "augmented_A": tuple(augmented.A.flat),
            "augmented_A_reference": tuple(reference_augmented_a.flat),
            "augmented_A_shape": (4, 4),
            "augmented_B": tuple(augmented.B.flat),
            "augmented_B_reference": tuple(reference_augmented_b.flat),
            "augmented_B_shape": (4, 1),
            "augmented_C": tuple(augmented.C.flat),
            "augmented_C_reference": tuple(reference_augmented_c.flat),
            "augmented_C_shape": (3, 4),
            "augmented_D": tuple(augmented.D.flat),
            "augmented_D_reference": tuple(reference_augmented_d.flat),
            "augmented_D_shape": (3, 1),
            "augmented_realization_residual_tolerance": numerical_tolerance,
            "auxiliary_response_carrier": True,
            "characteristic_coefficient_residual_tolerance": numerical_tolerance,
            "characteristic_coefficients": tuple(coefficients),
            "characteristic_coefficients_reference": tuple(
                reference_coefficients
            ),
            "characteristic_determinant": determinant,
            "characteristic_discriminant": discriminant,
            "characteristic_discriminant_nonnegative": discriminant_nonnegative,
            "characteristic_trace": trace,
            "error_dynamics_asymptotically_stable": error_dynamics_stable,
            "error_state_A": tuple(error_a.flat),
            "error_state_A_reference": tuple(reference_error_a.flat),
            "error_state_matrix_residual_tolerance": numerical_tolerance,
            "estimation_error_definition": "e = x - x_hat",
            "estimation_error_dynamics": "e_dot = (A - L C) e",
            "maximum_absolute_achieved_error_pole_residual": pole_residual,
            "maximum_absolute_augmented_realization_residual": augmented_residual,
            "maximum_absolute_characteristic_coefficient_residual": (
                coefficient_residual
            ),
            "maximum_absolute_error_state_matrix_residual": error_a_residual,
            "maximum_absolute_observer_gain_preservation_residual": (
                gain_preservation_residual
            ),
            "maximum_absolute_observer_gain_residual": gain_residual,
            "maximum_absolute_source_matrix_preservation_residual": (
                source_preservation_residual
            ),
            "observer_convention": (
                "x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)"
            ),
            "observer_gain_preservation_tolerance": exact_tolerance,
            "observer_gain_reference": tuple(reference_gain.flat),
            "observer_gain_residual_tolerance": numerical_tolerance,
            "observer_gain_stored_by_interconnection": tuple(stored_gain.flat),
            "observer_pole_residual_tolerance": numerical_tolerance,
            "output_order": ("y", "x_hat"),
            "passed": passed,
            "source_matrices_preserved": source_matrices_preserved,
            "source_matrix_preservation_tolerance": exact_tolerance,
            "state_order": ("x", "x_hat"),
        },
        run_id="verification-mathworks-luenberger-observer-v1",
        created_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
    )


def run_mathworks_separation_principle_verification_benchmark() -> ExperimentRun:
    """Verify dynamic output feedback and the separation principle."""
    matrix_a = np.array([[-1.0, -3.0 / 4.0], [1.0, 0.0]])
    matrix_b = np.array([[1.0], [0.0]])
    matrix_c = np.array([[1.0, 1.0]])
    matrix_d = np.array([[0.0]])
    state_feedback_gain = np.array([[2.0, 5.0 / 4.0]])
    observer_gain = np.array([[17.0 / 3.0], [-5.0 / 3.0]])

    reference_feedback = np.array([[2.0, 5.0 / 4.0], [0.0, 0.0]])
    reference_controller_a = np.array([[-3.0, -2.0], [1.0, 0.0]])
    reference_correction = np.array(
        [[17.0 / 3.0, 17.0 / 3.0], [-5.0 / 3.0, -5.0 / 3.0]]
    )
    reference_error_a = np.array(
        [[-20.0 / 3.0, -77.0 / 12.0], [8.0 / 3.0, 5.0 / 3.0]]
    )
    reference_bottom_right = np.array(
        [[-26.0 / 3.0, -23.0 / 3.0], [8.0 / 3.0, 5.0 / 3.0]]
    )
    reference_augmented_a = np.array(
        [
            [-1.0, -3.0 / 4.0, -2.0, -5.0 / 4.0],
            [1.0, 0.0, 0.0, 0.0],
            [17.0 / 3.0, 17.0 / 3.0, -26.0 / 3.0, -23.0 / 3.0],
            [-5.0 / 3.0, -5.0 / 3.0, 8.0 / 3.0, 5.0 / 3.0],
        ]
    )
    reference_augmented_b = np.array([[1.0], [0.0], [1.0], [0.0]])
    reference_augmented_c = np.array([[1.0, 1.0, 0.0, 0.0]])
    reference_augmented_d = np.array([[0.0]])
    coordinate_map = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ]
    )
    reference_separated_a = np.block(
        [
            [reference_controller_a, reference_feedback],
            [np.zeros((2, 2)), reference_error_a],
        ]
    )
    reference_separated_b = np.array([[1.0], [0.0], [0.0], [0.0]])
    reference_separated_c = np.array([[1.0, 1.0, 0.0, 0.0]])
    reference_controller_coefficients = np.array([1.0, 3.0, 2.0])
    reference_observer_coefficients = np.array([1.0, 5.0, 6.0])
    reference_poles = np.array([-3.0, -2.0, -2.0, -1.0])
    reference_combined_coefficients = np.array([1.0, 8.0, 23.0, 28.0, 12.0])
    numerical_tolerance = 1.0e-12
    exact_tolerance = 0.0

    for name, matrix, shape in (
        ("A", matrix_a, (2, 2)),
        ("B", matrix_b, (2, 1)),
        ("C", matrix_c, (1, 2)),
        ("D", matrix_d, (1, 1)),
        ("K", state_feedback_gain, (1, 2)),
        ("L", observer_gain, (2, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks separation benchmark {name} must have shape {shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks separation benchmark {name} must contain only finite "
                "real values"
            )

    source_copies = tuple(
        matrix.copy()
        for matrix in (
            matrix_a,
            matrix_b,
            matrix_c,
            matrix_d,
            state_feedback_gain,
            observer_gain,
        )
    )
    system = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
    interconnection = system.observer_based_output_feedback(
        state_feedback_gain, observer_gain
    )
    if not isinstance(interconnection, ObserverBasedOutputFeedbackInterconnection):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks separation benchmark result must be an "
            "ObserverBasedOutputFeedbackInterconnection"
        )

    stored_k = np.asarray(interconnection.state_feedback_gain)
    stored_l = np.asarray(interconnection.observer_gain)
    for name, gain, shape in (
        ("stored K", stored_k, (1, 2)),
        ("stored L", stored_l, (2, 1)),
    ):
        if gain.shape != shape:
            raise ValueError(
                f"MathWorks separation benchmark {name} must have shape {shape}"
            )
        if np.iscomplexobj(gain) or not np.all(np.isfinite(gain)):
            raise ValueError(
                f"MathWorks separation benchmark {name} must contain only finite "
                "real values"
            )

    augmented = interconnection.system
    if not isinstance(augmented, StateSpace):
        raise ValueError(  # noqa: TRY004 - frozen evidence contract uses ValueError
            "MathWorks separation benchmark augmented system must be a StateSpace"
        )
    for name, matrix, shape in (
        ("A", np.asarray(augmented.A), (4, 4)),
        ("B", np.asarray(augmented.B), (4, 1)),
        ("C", np.asarray(augmented.C), (1, 4)),
        ("D", np.asarray(augmented.D), (1, 1)),
    ):
        if matrix.shape != shape:
            raise ValueError(
                f"MathWorks separation benchmark augmented {name} must have shape "
                f"{shape}"
            )
        if np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"MathWorks separation benchmark augmented {name} must contain "
                "only finite real values"
            )

    expected_feedback = system.B @ stored_k
    expected_correction = stored_l @ system.C
    expected_augmented_a = np.block(
        [
            [system.A, -expected_feedback],
            [
                expected_correction,
                system.A - expected_feedback - expected_correction,
            ],
        ]
    )
    expected_augmented_b = np.vstack((system.B, system.B))
    expected_augmented_c = np.hstack((system.C, -system.D @ stored_k))
    if not all(
        np.array_equal(actual, expected)
        for actual, expected in (
            (augmented.A, expected_augmented_a),
            (augmented.B, expected_augmented_b),
            (augmented.C, expected_augmented_c),
            (augmented.D, system.D),
        )
    ):
        raise ValueError(
            "MathWorks separation benchmark interconnection is internally "
            "inconsistent"
        )

    source_residual = float(
        max(
            np.max(np.abs(matrix - original))
            for matrix, original in zip(
                (
                    system.A,
                    system.B,
                    system.C,
                    system.D,
                    state_feedback_gain,
                    observer_gain,
                ),
                source_copies,
                strict=True,
            )
        )
    )
    if source_residual != exact_tolerance:
        raise ValueError("MathWorks separation benchmark source inputs were mutated")

    controller_a = system.A - system.B @ stored_k
    error_a = system.A - stored_l @ system.C
    separated_a = coordinate_map @ augmented.A @ coordinate_map
    separated_b = coordinate_map @ augmented.B
    separated_c = augmented.C @ coordinate_map
    involution = coordinate_map @ coordinate_map

    def characteristic_coefficients(matrix):
        trace = float(matrix[0, 0] + matrix[1, 1])
        determinant = float(
            matrix[0, 0] * matrix[1, 1]
            - matrix[0, 1] * matrix[1, 0]
        )
        coefficients = np.array([1.0, -trace, determinant])
        discriminant = float(coefficients[1] ** 2 - 4.0 * coefficients[2])
        if not all(np.isfinite(value) for value in (*coefficients, discriminant)):
            raise ValueError(
                "MathWorks separation benchmark characteristic evidence must be "
                "finite"
            )
        if discriminant < 0.0:
            raise ValueError(
                "MathWorks separation benchmark quadratic discriminant must be "
                "nonnegative"
            )
        root = float(np.sqrt(discriminant))
        poles = np.sort(
            np.array(
                [
                    (-coefficients[1] - root) / 2.0,
                    (-coefficients[1] + root) / 2.0,
                ]
            )
        )
        if not np.all(np.isfinite(poles)):
            raise ValueError(
                "MathWorks separation benchmark derived poles must be finite"
            )
        return coefficients, discriminant, poles

    controller_coefficients, controller_discriminant, controller_poles = (
        characteristic_coefficients(controller_a)
    )
    observer_coefficients, observer_discriminant, observer_poles = (
        characteristic_coefficients(error_a)
    )
    separated_poles = np.sort(np.concatenate((controller_poles, observer_poles)))
    combined_coefficients = np.array(
        [
            controller_coefficients[0] * observer_coefficients[0],
            controller_coefficients[0] * observer_coefficients[1]
            + controller_coefficients[1] * observer_coefficients[0],
            controller_coefficients[0] * observer_coefficients[2]
            + controller_coefficients[1] * observer_coefficients[1]
            + controller_coefficients[2] * observer_coefficients[0],
            controller_coefficients[1] * observer_coefficients[2]
            + controller_coefficients[2] * observer_coefficients[1],
            controller_coefficients[2] * observer_coefficients[2],
        ]
    )
    if not np.all(np.isfinite(combined_coefficients)):
        raise ValueError(
            "MathWorks separation benchmark combined polynomial must be finite"
        )

    gain_residual = float(
        max(
            np.max(np.abs(stored_k - state_feedback_gain)),
            np.max(np.abs(stored_l - observer_gain)),
        )
    )
    augmented_residual = float(
        max(
            np.max(np.abs(augmented.A - reference_augmented_a)),
            np.max(np.abs(augmented.B - reference_augmented_b)),
            np.max(np.abs(augmented.C - reference_augmented_c)),
            np.max(np.abs(augmented.D - reference_augmented_d)),
        )
    )
    separation_residual = float(
        max(
            np.max(np.abs(separated_a - reference_separated_a)),
            np.max(np.abs(separated_b - reference_separated_b)),
            np.max(np.abs(separated_c - reference_separated_c)),
        )
    )
    involution_residual = float(np.max(np.abs(involution - np.eye(4))))
    zero_block_residual = float(np.max(np.abs(separated_a[2:, :2])))
    coefficient_residual = float(
        max(
            np.max(
                np.abs(
                    controller_coefficients - reference_controller_coefficients
                )
            ),
            np.max(
                np.abs(observer_coefficients - reference_observer_coefficients)
            ),
        )
    )
    pole_residual = float(np.max(np.abs(separated_poles - reference_poles)))
    polynomial_residual = float(
        np.max(np.abs(combined_coefficients - reference_combined_coefficients))
    )
    residuals = (
        gain_residual,
        augmented_residual,
        separation_residual,
        involution_residual,
        zero_block_residual,
        coefficient_residual,
        pole_residual,
        polynomial_residual,
        source_residual,
    )
    if not all(np.isfinite(value) for value in residuals):
        raise ValueError("MathWorks separation benchmark residuals must be finite")

    gains_preserved = gain_residual <= exact_tolerance
    source_inputs_preserved = source_residual <= exact_tolerance
    coordinate_map_involutory = involution_residual <= exact_tolerance
    discriminants_nonnegative = bool(
        controller_discriminant >= 0.0 and observer_discriminant >= 0.0
    )
    controller_poles_stable = bool(np.all(controller_poles < 0.0))
    observer_poles_stable = bool(np.all(observer_poles < 0.0))
    pole_union_multiplicity_matches = bool(
        np.max(np.abs(separated_poles - reference_poles)) <= numerical_tolerance
    )
    passed = bool(
        gains_preserved
        and augmented_residual <= numerical_tolerance
        and separation_residual <= numerical_tolerance
        and coordinate_map_involutory
        and zero_block_residual <= numerical_tolerance
        and coefficient_residual <= numerical_tolerance
        and pole_residual <= numerical_tolerance
        and polynomial_residual <= numerical_tolerance
        and source_inputs_preserved
        and discriminants_nonnegative
        and controller_poles_stable
        and observer_poles_stable
        and pole_union_multiplicity_matches
    )

    carrier_time = np.array([0.0, 1.0])
    metrics = response_metrics(carrier_time, np.zeros(2), np.zeros(2))
    if metrics.maximum_absolute_tracking_error != 0.0:
        raise ValueError("MathWorks separation carrier evidence is inconsistent")

    return experiment_run(
        time=carrier_time,
        initial_state=np.zeros(2),
        metrics=metrics,
        method="exact",
        system={
            "A": tuple(matrix_a.flat),
            "A_shape": (2, 2),
            "B": tuple(matrix_b.flat),
            "B_shape": (2, 1),
            "C": tuple(matrix_c.flat),
            "C_shape": (1, 2),
            "D": tuple(matrix_d.flat),
            "D_shape": (1, 1),
            "input_count": 1,
            "output_count": 1,
            "physical_units": "none assigned",
            "state_count": 2,
        },
        controller={
            "observer_gain_L": tuple(observer_gain.flat),
            "state_feedback_gain_K": tuple(state_feedback_gain.flat),
            "type": "observer_based_dynamic_output_feedback",
        },
        reference={
            "access_date": "2026-09-03",
            "block_triangular_dynamics": "[[A-BK, BK], [0, A-LC]]",
            "complete_closed_loop_poles": (
                "union of eigenvalues of A-BK and A-LC with multiplicity"
            ),
            "controller_dynamics": "A-BK",
            "dynamic_output_feedback": "u = -K xi",
            "error_coordinate_definition": "e = x - xi",
            "evidence_classification": (
                "computational/software verification of observer-based dynamic "
                "output-feedback interconnection and the continuous-time "
                "separation principle against an authoritative control-design "
                "contract and exact algebraic oracles"
            ),
            "observer_equation": (
                "xi_dot = A xi + B u + L(y - C xi - D u)"
            ),
            "page_title": "Pole Placement",
            "product_documentation": "MathWorks Control System Toolbox Documentation",
            "section_titles": (
                "State-Feedback Gain Selection",
                "State Estimator Design",
            ),
            "source_url": (
                "https://www.mathworks.com/help/control/getstart/"
                "pole-placement.html"
            ),
            "state_feedback_convention": "u = -Kx",
        },
        user_metadata={
            "augmented_A": tuple(augmented.A.flat),
            "augmented_A_reference": tuple(reference_augmented_a.flat),
            "augmented_A_shape": (4, 4),
            "augmented_B": tuple(augmented.B.flat),
            "augmented_B_reference": tuple(reference_augmented_b.flat),
            "augmented_B_shape": (4, 1),
            "augmented_C": tuple(augmented.C.flat),
            "augmented_C_reference": tuple(reference_augmented_c.flat),
            "augmented_C_shape": (1, 4),
            "augmented_D": tuple(augmented.D.flat),
            "augmented_D_reference": tuple(reference_augmented_d.flat),
            "augmented_D_shape": (1, 1),
            "augmented_realization_residual_tolerance": numerical_tolerance,
            "auxiliary_response_carrier": True,
            "combined_characteristic_coefficients": tuple(combined_coefficients),
            "combined_characteristic_coefficients_reference": tuple(
                reference_combined_coefficients
            ),
            "combined_polynomial_residual_tolerance": numerical_tolerance,
            "characteristic_coefficient_residual_tolerance": numerical_tolerance,
            "controller_characteristic_coefficients": tuple(
                controller_coefficients
            ),
            "controller_characteristic_discriminant": controller_discriminant,
            "controller_state_A": tuple(controller_a.flat),
            "controller_state_A_reference": tuple(reference_controller_a.flat),
            "controller_poles_asymptotically_stable": controller_poles_stable,
            "controller_poles_sorted": tuple(controller_poles),
            "coordinate_map_S": tuple(coordinate_map.flat),
            "coordinate_map_S_shape": (4, 4),
            "coordinate_map_involutory": coordinate_map_involutory,
            "feedback_product_BK_reference": tuple(reference_feedback.flat),
            "gain_preservation_tolerance": exact_tolerance,
            "gains_preserved": gains_preserved,
            "involution_residual_tolerance": exact_tolerance,
            "maximum_absolute_augmented_realization_residual": augmented_residual,
            "maximum_absolute_characteristic_coefficient_residual": (
                coefficient_residual
            ),
            "maximum_absolute_combined_polynomial_residual": polynomial_residual,
            "maximum_absolute_gain_preservation_residual": gain_residual,
            "maximum_absolute_involution_residual": involution_residual,
            "maximum_absolute_separated_pole_residual": pole_residual,
            "maximum_absolute_separation_coordinate_residual": separation_residual,
            "maximum_absolute_source_input_preservation_residual": source_residual,
            "maximum_absolute_zero_lower_left_block_residual": zero_block_residual,
            "observer_characteristic_coefficients": tuple(observer_coefficients),
            "observer_characteristic_discriminant": observer_discriminant,
            "observer_correction_LC_reference": tuple(reference_correction.flat),
            "observer_error_A": tuple(error_a.flat),
            "observer_error_A_reference": tuple(reference_error_a.flat),
            "observer_poles_asymptotically_stable": observer_poles_stable,
            "observer_poles_sorted": tuple(observer_poles),
            "passed": passed,
            "pole_union_multiplicity_matches": pole_union_multiplicity_matches,
            "separated_pole_residual_tolerance": numerical_tolerance,
            "separated_A": tuple(separated_a.flat),
            "separated_A_reference": tuple(reference_separated_a.flat),
            "separated_B": tuple(separated_b.flat),
            "separated_B_reference": tuple(reference_separated_b.flat),
            "separated_C": tuple(separated_c.flat),
            "separated_C_reference": tuple(reference_separated_c.flat),
            "separated_poles_sorted": tuple(separated_poles),
            "separation_coordinate_residual_tolerance": numerical_tolerance,
            "source_input_preservation_tolerance": exact_tolerance,
            "source_inputs_preserved": source_inputs_preserved,
            "state_order": ("x", "x_hat"),
            "state_order_after_coordinate_map": ("x", "e"),
            "stored_observer_gain_L": tuple(stored_l.flat),
            "stored_state_feedback_gain_K": tuple(stored_k.flat),
            "observer_bottom_right_A_reference": tuple(
                reference_bottom_right.flat
            ),
            "zero_lower_left_block_residual_tolerance": numerical_tolerance,
        },
        run_id="verification-mathworks-separation-principle-v1",
        created_at=datetime(2026, 9, 6, tzinfo=UTC),
    )
