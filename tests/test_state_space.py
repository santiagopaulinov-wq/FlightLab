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


def test_eigenvalues_and_asymptotic_stability_for_stable_system():
    system = StateSpace(*valid_matrices())

    np.testing.assert_allclose(np.sort(system.eigenvalues()), [-2, -1])
    assert system.is_asymptotically_stable() is True


def test_asymptotic_stability_rejects_unstable_system():
    _, B, C, D = valid_matrices()
    system = StateSpace([[0, 1], [2, 1]], B, C, D)

    assert np.any(system.eigenvalues().real > 0.0)
    assert system.is_asymptotically_stable() is False


def test_asymptotic_stability_rejects_marginal_system():
    _, B, C, D = valid_matrices()
    system = StateSpace([[0, -1], [1, 0]], B, C, D)

    np.testing.assert_allclose(system.eigenvalues().real, 0.0, atol=1e-15)
    assert system.is_asymptotically_stable() is False


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


def test_rk4_step_matches_scalar_analytical_solution():
    system = StateSpace([[-2.0]], [[0.0]], [[1.0]], [[0.0]])

    result = system.rk4_step([1.0], [0.0], 0.1)

    np.testing.assert_allclose(result, [np.exp(-0.2)], rtol=2e-5)


def test_rk4_step_is_more_accurate_than_euler_for_scalar_system():
    system = StateSpace([[-2.0]], [[0.0]], [[1.0]], [[0.0]])
    exact = np.exp(-0.2)

    euler_error = abs(system.euler_step([1.0], [0.0], 0.1)[0] - exact)
    rk4_error = abs(system.rk4_step([1.0], [0.0], 0.1)[0] - exact)

    assert rk4_error < euler_error


def test_rk4_step_for_multi_state_system():
    system = StateSpace(
        [[0.0, 1.0], [-1.0, 0.0]],
        [[0.0], [0.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )
    dt = 0.1

    result = system.rk4_step([1.0, 0.0], [0.0], dt)

    np.testing.assert_allclose(
        result,
        [1.0 - dt**2 / 2.0 + dt**4 / 24.0, -dt + dt**3 / 6.0],
    )


def test_rk4_step_with_nonzero_control_input():
    system = StateSpace([[0.0]], [[2.0]], [[1.0]], [[0.0]])

    result = system.rk4_step([1.0], [3.0], 0.25)

    np.testing.assert_allclose(result, [2.5])


@pytest.mark.parametrize(
    ("x", "u", "message"),
    [
        ([[1.0, 2.0]], [3.0], "x must be a 1D vector"),
        ([1.0, 2.0], [[3.0]], "u must be a 1D vector"),
    ],
)
def test_rk4_step_rejects_invalid_state_and_input_shapes(x, u, message):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match=message):
        system.rk4_step(x, u, 0.1)


@pytest.mark.parametrize("dt", [0, -0.1, np.inf, [0.1]])
def test_rk4_step_requires_positive_finite_scalar_dt(dt):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="dt must be a finite positive scalar"):
        system.rk4_step([1.0, 2.0], [3.0], dt)


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


def test_simulate_explicit_euler_matches_default_exactly():
    system = StateSpace(*valid_matrices())
    arguments = ([1.0, 2.0], [3.0], [0.0, 0.1, 0.3])

    default_states, default_outputs = system.simulate(*arguments)
    euler_states, euler_outputs = system.simulate(*arguments, method="euler")

    np.testing.assert_array_equal(euler_states, default_states)
    np.testing.assert_array_equal(euler_outputs, default_outputs)


def test_simulate_rk4_matches_scalar_analytical_trajectory():
    system = StateSpace([[-1.0]], [[0.0]], [[1.0]], [[0.0]])
    time = np.array([0.0, 0.1, 0.2, 0.3])

    states, outputs = system.simulate([1.0], [0.0], time, method="rk4")

    expected = np.exp(-time)[:, np.newaxis]
    np.testing.assert_allclose(states, expected, rtol=3e-7)
    np.testing.assert_allclose(outputs, expected, rtol=3e-7)


def test_simulate_rk4_is_more_accurate_than_euler():
    system = StateSpace([[-1.0]], [[0.0]], [[1.0]], [[0.0]])
    time = np.linspace(0.0, 1.0, 6)
    exact_final_state = np.exp(-1.0)

    euler_states, _ = system.simulate([1.0], [0.0], time, method="euler")
    rk4_states, _ = system.simulate([1.0], [0.0], time, method="rk4")

    assert abs(rk4_states[-1, 0] - exact_final_state) < abs(
        euler_states[-1, 0] - exact_final_state
    )


def test_simulate_rk4_with_constant_input():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    time = np.array([0.0, 0.1, 0.2])

    states, _ = system.simulate([0.0], [2.0], time, method="rk4")

    expected = (2.0 * (1.0 - np.exp(-time)))[:, np.newaxis]
    np.testing.assert_allclose(states, expected, rtol=1e-6, atol=1e-8)


def test_simulate_rk4_uses_left_sampled_time_varying_input():
    system = StateSpace([[0.0]], [[1.0]], [[1.0]], [[0.0]])

    states, _ = system.simulate(
        [0.0], [[1.0], [2.0], [100.0]], [0.0, 0.5, 1.0], method="rk4"
    )

    np.testing.assert_allclose(states, [[0.0], [0.5], [1.5]])


@pytest.mark.parametrize("method", ["bogus", "Euler", None])
def test_simulate_rejects_unsupported_integration_method(method):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="method must be 'euler' or 'rk4'"):
        system.simulate([1.0, 2.0], [3.0], [0.0, 0.1], method=method)


def test_zero_input_response_for_scalar_decay_system():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])

    states, outputs = system.zero_input_response([2.0], [0.0, 0.1, 0.2])

    np.testing.assert_allclose(states, [[2.0], [1.8], [1.62]])
    np.testing.assert_allclose(outputs, states)


def test_zero_input_response_has_nonzero_transient_from_nonzero_initial_state():
    system = StateSpace([[-2.0]], [[1.0]], [[1.0]], [[0.0]])

    states, _ = system.zero_input_response([1.0], [0.0, 0.1, 0.2], method="rk4")

    assert np.all(states > 0.0)
    assert np.any(states[1:] != 0.0)


def test_zero_input_response_converges_toward_zero_for_stable_system():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    time = np.linspace(0.0, 5.0, 51)

    states, _ = system.zero_input_response([3.0], time, method="rk4")

    assert abs(states[-1, 0]) < abs(states[0, 0])
    assert abs(states[-1, 0]) < 0.03


def test_zero_input_response_euler_matches_direct_simulation():
    system = StateSpace(*valid_matrices())
    time = [0.0, 0.1, 0.3]

    response = system.zero_input_response([1.0, 2.0], time, method="euler")
    direct = system.simulate([1.0, 2.0], [0.0], time, method="euler")

    for actual, expected in zip(response, direct, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_zero_input_response_rk4_matches_direct_simulation():
    system = StateSpace(*valid_matrices())
    time = [0.0, 0.1, 0.3]

    response = system.zero_input_response([1.0, 2.0], time, method="rk4")
    direct = system.simulate([1.0, 2.0], [0.0], time, method="rk4")

    for actual, expected in zip(response, direct, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_zero_input_response_for_multi_state_system():
    system = StateSpace(
        [[0.0, 1.0], [-1.0, -1.0]],
        [[1.0], [0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0], [0.0]],
    )

    states, outputs = system.zero_input_response(
        [1.0, -0.5], [0.0, 0.1, 0.2], method="rk4"
    )

    assert states.shape == (3, 2)
    assert outputs.shape == (3, 2)
    np.testing.assert_allclose(outputs, states)


def test_zero_input_response_constructs_zero_for_multiple_inputs():
    system = StateSpace(
        [[0.0]],
        [[1.0, -2.0, 4.0]],
        [[1.0]],
        [[3.0, 5.0, 7.0]],
    )

    states, outputs = system.zero_input_response([2.0], [0.0, 0.5, 1.0])

    np.testing.assert_array_equal(states, [[2.0], [2.0], [2.0]])
    np.testing.assert_array_equal(outputs, states)


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
