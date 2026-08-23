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
