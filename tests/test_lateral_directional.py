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


def test_lateral_directional_model_rejects_invalid_parameter():
    parameters = valid_parameters()
    parameters["n_r"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "n_r must be a finite scalar"):
        LateralDirectionalModel(**parameters)
