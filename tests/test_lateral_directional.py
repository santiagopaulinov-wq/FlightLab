import numpy as np

from flightlab.lateral_directional import LateralDirectionalModel
from flightlab.state_space import StateSpace


def valid_parameters():
    return {
        "trim_speed": 10,
        "trim_pitch": 0,
        "gravity": 9,
        "y_v": 1,
        "y_p": 2,
        "y_r": 3,
        "y_delta_a": 4,
        "y_delta_r": 5,
        "l_v": 6,
        "l_p": 7,
        "l_r": 8,
        "l_delta_a": 9,
        "l_delta_r": 10,
        "n_v": 11,
        "n_p": 12,
        "n_r": 13,
        "n_delta_a": 14,
        "n_delta_r": 15,
    }


def test_lateral_directional_model_builds_expected_state_space_matrices():
    model = LateralDirectionalModel(**valid_parameters())

    system = model.to_state_space()

    assert isinstance(system, StateSpace)
    np.testing.assert_array_equal(
        system.A,
        [[1, 2, -7, 9], [6, 7, 8, 0], [11, 12, 13, 0], [0, 1, 0, 0]],
    )
    np.testing.assert_array_equal(
        system.B, [[4, 5], [9, 10], [14, 15], [0, 0]]
    )
    np.testing.assert_array_equal(system.C, np.eye(4))
    np.testing.assert_array_equal(system.D, np.zeros((4, 2)))
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
    modal_output = system.modal_output_influence()
    assert eigenvectors.shape == (4, 4)
    assert left_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.right_eigenvectors.shape == (4, 4)
    assert biorthogonal_modes.left_eigenvectors.shape == (4, 4)
    assert participation.shape == (4, 4)
    assert modal_input.shape == (4, 2)
    assert modal_output.shape == (4, 4)
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
    assert np.all(np.isfinite(modal_output.real))
    assert np.all(np.isfinite(modal_output.imag))
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
    np.testing.assert_allclose(
        modal_output, system.C @ biorthogonal_modes.right_eigenvectors
    )
    assert LateralDirectionalModel.INPUT_ORDER == ("delta_a", "delta_r")
    assert LateralDirectionalModel.OUTPUT_ORDER == LateralDirectionalModel.STATE_ORDER
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
    assert system.rk4_step(np.zeros(4), np.zeros(2), 0.01).shape == (4,)


def test_lateral_directional_model_declares_state_input_and_output_ordering():
    assert LateralDirectionalModel.STATE_ORDER == ("v", "p", "r", "phi")
    assert LateralDirectionalModel.INPUT_ORDER == ("delta_a", "delta_r")
    assert LateralDirectionalModel.OUTPUT_ORDER == LateralDirectionalModel.STATE_ORDER


def test_lateral_directional_model_simulates_small_aileron_step():
    model = LateralDirectionalModel(
        trim_speed=10.0,
        trim_pitch=0.1,
        gravity=9.81,
        y_v=-0.9,
        y_p=0.4,
        y_r=7.0,
        y_delta_a=0.2,
        y_delta_r=0.1,
        l_v=-0.8,
        l_p=-2.9,
        l_r=0.3,
        l_delta_a=0.5,
        l_delta_r=-0.2,
        n_v=0.3,
        n_p=0.3,
        n_r=-1.5,
        n_delta_a=-0.1,
        n_delta_r=0.4,
    )
    time = np.linspace(0.0, 0.5, 11)

    states, outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01, 0.0]), time
    )

    assert states.shape == (time.size, 4)
    assert outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    assert np.any(np.abs(states[1:]) > 0.0)
    np.testing.assert_allclose(outputs, states)
    phi_rate = states[:-1, 1] + np.tan(model.trim_pitch) * states[:-1, 2]
    np.testing.assert_allclose(np.diff(states[:, 3]), np.diff(time) * phi_rate)

    rk4_states, rk4_outputs = model.to_state_space().simulate(
        np.zeros(4), np.array([0.01, 0.0]), time, method="rk4"
    )
    assert rk4_states.shape == (time.size, 4)
    assert np.all(np.isfinite(rk4_states))
    np.testing.assert_allclose(rk4_outputs, rk4_states)

    zero_input_states, zero_input_outputs = model.to_state_space().zero_input_response(
        np.array([0.1, 0.0, 0.0, 0.01]), time
    )
    assert zero_input_states.shape == (time.size, 4)
    assert np.all(np.isfinite(zero_input_states))
    np.testing.assert_allclose(zero_input_outputs, zero_input_states)

    forced_states, forced_outputs = model.to_state_space().forced_response(
        np.array([0.01, 0.0]), time
    )
    assert forced_states.shape == (time.size, 4)
    assert forced_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(forced_states))
    assert np.all(np.isfinite(forced_outputs))
    assert np.any(np.abs(forced_states[1:]) > 0.0)
    np.testing.assert_allclose(forced_outputs, forced_states)

    step_states, step_outputs = model.to_state_space().step_response(
        np.array([0.01, 0.0]), time
    )
    assert step_states.shape == (time.size, 4)
    assert step_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(step_states))
    assert np.all(np.isfinite(step_outputs))
    assert np.any(np.abs(step_states[1:]) > 0.0)
    np.testing.assert_allclose(step_outputs, step_states)

    impulse_states, impulse_outputs = model.to_state_space().impulse_response(
        np.array([0.01, 0.0]), time
    )
    assert impulse_states.shape == (time.size, 4)
    assert impulse_outputs.shape == (time.size, 4)
    assert np.all(np.isfinite(impulse_states))
    assert np.all(np.isfinite(impulse_outputs))
    assert np.any(np.abs(impulse_states[1:]) > 0.0)
    np.testing.assert_allclose(impulse_outputs, impulse_states)


def test_lateral_directional_model_rejects_invalid_parameter():
    parameters = valid_parameters()
    parameters["n_r"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "n_r must be a finite scalar"):
        LateralDirectionalModel(**parameters)
