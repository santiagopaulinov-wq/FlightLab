import numpy as np

from flightlab.longitudinal import LongitudinalModel
from flightlab.state_space import StateSpace


def valid_parameters():
    return {
        "trim_speed": 10,
        "trim_pitch": 0,
        "gravity": 9,
        "x_u": 1,
        "x_w": 2,
        "x_q": 3,
        "x_delta_e": 4,
        "z_u": 5,
        "z_w": 6,
        "z_q": 7,
        "z_delta_e": 8,
        "m_u": 9,
        "m_w": 10,
        "m_q": 11,
        "m_delta_e": 12,
    }


def test_longitudinal_model_builds_expected_state_space_matrices():
    model = LongitudinalModel(**valid_parameters())

    system = model.to_state_space()

    assert isinstance(system, StateSpace)
    np.testing.assert_array_equal(
        system.A,
        [[1, 2, 3, -9], [5, 6, 17, 0], [9, 10, 11, 0], [0, 0, 1, 0]],
    )
    np.testing.assert_array_equal(system.B, [[4], [8], [12], [0]])
    np.testing.assert_array_equal(system.C, np.eye(4))
    np.testing.assert_array_equal(system.D, np.zeros((4, 1)))
    assert system.eigenvalues().shape == (4,)
    modes = system.modal_properties()
    assert len(modes) == 4
    np.testing.assert_allclose([mode.eigenvalue for mode in modes], system.eigenvalues())
    eigenvalues = system.eigenvalues()
    eigenvectors = system.right_eigenvectors()
    left_eigenvectors = system.left_eigenvectors()
    biorthogonal_modes = system.biorthogonal_modes()
    participation = system.participation_factors()
    modal_input = system.modal_input_influence()
    assert eigenvectors.shape == (4, 4)
    assert left_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.right_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.left_eigenvectors.shape == (4, 4)
    assert participation.shape == (4, 4)
    assert modal_input.shape == (4, 1)
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(np.isfinite(eigenvectors.real))
    assert np.all(np.isfinite(eigenvectors.imag))
    assert np.all(np.isfinite(left_eigenvectors.real))
    assert np.all(np.isfinite(left_eigenvectors.imag))
    assert np.all(np.isfinite(biorthogonal_modes.right_eigenvectors.real))
    assert np.all(np.isfinite(biorthogonal_modes.right_eigenvectors.imag))
    assert np.all(np.isfinite(biorthogonal_modes.left_eigenvectors.real))
    assert np.all(np.isfinite(biorthogonal_modes.left_eigenvectors.imag))
    assert np.all(np.isfinite(participation.real))
    assert np.all(np.isfinite(participation.imag))
    assert np.all(np.isfinite(modal_input.real))
    assert np.all(np.isfinite(modal_input.imag))
    np.testing.assert_array_equal(biorthogonal_modes.eigenvalues, eigenvalues)
    np.testing.assert_allclose(
        participation,
        biorthogonal_modes.right_eigenvectors
        * np.conj(biorthogonal_modes.left_eigenvectors),
    )
    np.testing.assert_allclose(np.sum(participation, axis=0), np.ones(4))
    np.testing.assert_allclose(
        modal_input, biorthogonal_modes.left_eigenvectors.conj().T @ system.B
    )
    assert LongitudinalModel.INPUT_ORDER == ("delta_e",)
    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index]
        np.testing.assert_allclose(system.A @ vector, eigenvalue * vector)
        left_vector = left_eigenvectors[:, index]
        np.testing.assert_allclose(
            left_vector.conj().T @ system.A,
            eigenvalue * left_vector.conj().T,
        )
        scaled_right = biorthogonal_modes.right_eigenvectors[:, index]
        scaled_left = biorthogonal_modes.left_eigenvectors[:, index]
        np.testing.assert_allclose(scaled_left.conj().T @ scaled_right, 1.0)
        np.testing.assert_allclose(
            system.A @ scaled_right, eigenvalue * scaled_right
        )
        np.testing.assert_allclose(
            scaled_left.conj().T @ system.A,
            eigenvalue * scaled_left.conj().T,
        )
    assert isinstance(system.is_asymptotically_stable(), bool)
    assert system.rk4_step(np.zeros(4), np.zeros(1), 0.01).shape == (4,)


def test_longitudinal_model_declares_state_input_and_output_ordering():
    assert LongitudinalModel.STATE_ORDER == ("u", "w", "q", "theta")
    assert LongitudinalModel.INPUT_ORDER == ("delta_e",)
    assert LongitudinalModel.OUTPUT_ORDER == LongitudinalModel.STATE_ORDER


def test_longitudinal_model_simulates_small_elevator_step():
    model = LongitudinalModel(
        trim_speed=10.0,
        trim_pitch=0.0,
        gravity=9.81,
        x_u=-0.4,
        x_w=-0.2,
        x_q=0.1,
        x_delta_e=0.2,
        z_u=-0.3,
        z_w=-1.8,
        z_q=-1.7,
        z_delta_e=0.5,
        m_u=0.9,
        m_w=-0.9,
        m_q=-2.7,
        m_delta_e=-1.0,
    )
    time = np.linspace(0.0, 0.5, 11)

    states, outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01]), time
    )

    assert states.shape == (time.size, 4)
    assert outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    assert np.any(np.abs(states[1:]) > 0.0)
    np.testing.assert_allclose(outputs, states)
    np.testing.assert_allclose(
        np.diff(states[:, 3]), np.diff(time) * states[:-1, 2]
    )

    rk4_states, rk4_outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01]), time, method="rk4"
    )
    assert rk4_states.shape == (time.size, 4)
    assert np.all(np.isfinite(rk4_states))
    np.testing.assert_allclose(rk4_outputs, rk4_states)

    zero_input_states, zero_input_outputs = model.to_state_space().zero_input_response(
        np.array([0.1, 0.0, 0.0, 0.01]), time, method="rk4"
    )
    assert zero_input_states.shape == (time.size, 4)
    assert np.all(np.isfinite(zero_input_states))
    np.testing.assert_allclose(zero_input_outputs, zero_input_states)

    forced_states, forced_outputs = model.to_state_space().forced_response(
        np.array([0.01]), time
    )
    assert forced_states.shape == (time.size, 4)
    assert forced_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(forced_states))
    assert np.all(np.isfinite(forced_outputs))
    assert np.any(np.abs(forced_states[1:]) > 0.0)
    np.testing.assert_allclose(forced_outputs, forced_states)

    step_states, step_outputs = model.to_state_space().step_response(
        np.array([0.01]), time
    )
    assert step_states.shape == (time.size, 4)
    assert step_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(step_states))
    assert np.all(np.isfinite(step_outputs))
    assert np.any(np.abs(step_states[1:]) > 0.0)
    np.testing.assert_allclose(step_outputs, step_states)

    impulse_states, impulse_outputs = model.to_state_space().impulse_response(
        np.array([0.01]), time
    )
    assert impulse_states.shape == (time.size, 4)
    assert impulse_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(impulse_states))
    assert np.all(np.isfinite(impulse_outputs))
    assert np.any(np.abs(impulse_states[1:]) > 0.0)
    np.testing.assert_allclose(impulse_outputs, impulse_states)


def test_longitudinal_model_rejects_non_finite_parameter():
    parameters = valid_parameters()
    parameters["m_q"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "m_q must be a finite scalar"):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_infinite_parameter():
    parameters = valid_parameters()
    parameters["trim_speed"] = np.inf

    with np.testing.assert_raises_regex(
        ValueError, "trim_speed must be a finite scalar"
    ):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_array_parameter():
    parameters = valid_parameters()
    parameters["x_u"] = np.array(1.0)

    with np.testing.assert_raises_regex(ValueError, "x_u must be a finite scalar"):
        LongitudinalModel(**parameters)


def test_longitudinal_model_rejects_non_scalar_parameter():
    parameters = valid_parameters()
    parameters["z_w"] = [1.0]

    with np.testing.assert_raises_regex(ValueError, "z_w must be a finite scalar"):
        LongitudinalModel(**parameters)
