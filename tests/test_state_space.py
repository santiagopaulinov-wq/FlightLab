import numpy as np
import pytest

from flightlab.state_space import StateSpace


def valid_matrices():
    return (
        np.array([[0, 1], [-2, -3]]),
        np.array([[0], [1]]),
        np.array([[1, 0]]),
        np.array([[0]]),
    )


def test_valid_construction():
    A, B, C, D = valid_matrices()
    system = StateSpace(A, B, C, D)

    assert system.A.dtype == float
    assert system.B.dtype == float
    assert system.C.dtype == float
    assert system.D.dtype == float
    assert all(matrix.ndim == 2 for matrix in (system.A, system.B, system.C, system.D))


def test_dimensions():
    system = StateSpace(*valid_matrices())

    assert system.n_states == 2
    assert system.n_inputs == 1
    assert system.n_outputs == 1


def test_invalid_A():
    _, B, C, D = valid_matrices()

    with pytest.raises(ValueError):
        StateSpace(np.zeros((2, 3)), B, C, D)


def test_invalid_B():
    A, _, C, D = valid_matrices()

    with pytest.raises(ValueError):
        StateSpace(A, np.zeros((3, 1)), C, D)


def test_invalid_C():
    A, B, _, D = valid_matrices()

    with pytest.raises(ValueError):
        StateSpace(A, B, np.zeros((1, 3)), D)


def test_invalid_D():
    A, B, C, _ = valid_matrices()

    with pytest.raises(ValueError):
        StateSpace(A, B, C, np.zeros((2, 1)))


def test_state_derivative():
    system = StateSpace(*valid_matrices())

    result = system.state_derivative([1, 2], [3])

    np.testing.assert_allclose(result, [2, -5])


def test_output():
    system = StateSpace(*valid_matrices())

    result = system.output([1, 2], [3])

    np.testing.assert_allclose(result, [1])


def test_euler_step():
    system = StateSpace(*valid_matrices())

    result = system.euler_step([1, 2], [3], 0.1)

    np.testing.assert_allclose(result, [1.2, 1.5])


@pytest.mark.parametrize("dt", [0, -0.1, np.inf, [0.1]])
def test_euler_step_requires_positive_finite_scalar_dt(dt):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="dt must be a finite positive scalar"):
        system.euler_step([1, 2], [3], dt)


def test_simulate_constant_input_over_time_grid():
    system = StateSpace(*valid_matrices())

    state_trajectory, _ = system.simulate([1, 2], [3], [0, 0.1, 0.3])

    np.testing.assert_allclose(state_trajectory, [[1, 2], [1.2, 1.5], [1.5, 0.72]])


def test_simulate_returns_state_and_output_trajectories():
    A, B, _, _ = valid_matrices()
    system = StateSpace(A, B, [[1, 0], [0, 1]], [[0.5], [-1]])

    states, outputs = system.simulate([1, 2], [3], [0, 0.1, 0.3])

    assert states.shape == (3, 2)
    assert outputs.shape == (3, 2)
    np.testing.assert_allclose(outputs, [[2.5, -1], [2.7, -1.5], [3, -2.28]])


def test_simulate_time_varying_inputs():
    A, B, C, _ = valid_matrices()
    system = StateSpace(A, B, C, [[2]])

    states, outputs = system.simulate([1, 0], [[0], [1], [2]], [0, 0.1, 0.2])

    np.testing.assert_allclose(states, [[1, 0], [1, -0.2], [0.98, -0.24]])
    np.testing.assert_allclose(outputs, [[1], [3], [4.98]])


def test_simulate_constant_input_backward_compatibility():
    system = StateSpace(*valid_matrices())

    states, outputs = system.simulate([1, 2], [3], [0, 0.1])

    np.testing.assert_allclose(states, [[1, 2], [1.2, 1.5]])
    np.testing.assert_allclose(outputs, [[1], [1.2]])


@pytest.mark.parametrize(
    "u",
    [3, [3, 4], [[3], [4]], [[3, 4], [5, 6], [7, 8]], [[[3]], [[4]], [[5]]]],
)
def test_simulate_rejects_invalid_input_trajectory_shapes(u):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="u must have shape"):
        system.simulate([1, 2], u, [0, 0.1, 0.2])


@pytest.mark.parametrize("time", [[], 0, [0, np.inf], [0, 0.1, 0.1], [0, -0.1]])
def test_simulate_requires_valid_time_grid(time):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError):
        system.simulate([1, 2], [3], time)


@pytest.mark.parametrize(
    ("x0", "u", "message"),
    [([[1, 2]], [3], "x must be a 1D vector")],
)
def test_simulate_validates_state_and_input_dimensions(x0, u, message):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match=message):
        system.simulate(x0, u, [0])


@pytest.mark.parametrize("method_name", ["state_derivative", "output"])
def test_invalid_state_vector_dimensions(method_name):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="x must be a 1D vector"):
        getattr(system, method_name)([[1, 2]], [3])


@pytest.mark.parametrize("method_name", ["state_derivative", "output"])
def test_invalid_input_vector_dimensions(method_name):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="u must be a 1D vector"):
        getattr(system, method_name)([1, 2], [[3]])
