import numpy as np
import pytest

from flightlab.state_space import (
    BalancedRealization,
    BalancedTruncation,
    ModalFamily,
    ModalProperties,
    ModalStateCharacterization,
    NonstablePBHDiagnostic,
    StateSpace,
    StructuralAnalysis,
)


def valid_matrices():
    return (
        np.array([[0, 1], [-2, -3]]),
        np.array([[0], [1]]),
        np.array([[1, 0]]),
        np.array([[0]]),
    )


def stable_minimal_balancing_matrices():
    return (
        np.array([[-2.0, 1.0], [-1.0, -3.0]]),
        np.array([[1.0, 0.5], [0.25, 1.0]]),
        np.array([[1.0, 0.2], [-0.4, 1.0]]),
        np.array([[0.1, 0.0], [0.0, -0.2]]),
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


def test_frequency_response_matches_first_order_siso_analytic_value():
    system = StateSpace([[-2.0]], [[3.0]], [[4.0]], [[0.5]])
    frequency = 1.5

    response = system.frequency_response(frequency)

    expected = 12.0 / (2.0 + 1j * frequency) + 0.5
    assert response.shape == (1, 1)
    assert response.dtype == complex
    np.testing.assert_allclose(response, [[expected]])


def test_frequency_response_at_dc_matches_static_gain():
    system = StateSpace(
        [[-2.0, 1.0], [0.0, -3.0]],
        [[1.0], [2.0]],
        [[4.0, -1.0]],
        [[0.25]],
    )

    response = system.frequency_response(0.0)
    expected = -system.C @ np.linalg.solve(system.A, system.B) + system.D

    np.testing.assert_allclose(response, expected)


def test_frequency_response_vector_preserves_shape_order_and_scalar_values():
    system = StateSpace([[-2.0]], [[3.0]], [[4.0]], [[0.5]])
    frequencies = np.array([3.0, 0.0, 1.5])

    response = system.frequency_response(frequencies)

    assert response.shape == (3, 1, 1)
    for index, frequency in enumerate(frequencies):
        np.testing.assert_array_equal(
            response[index], system.frequency_response(frequency)
        )


def test_frequency_response_mimo_includes_direct_feedthrough():
    A = np.diag([-1.0, -2.0])
    B = np.eye(2)
    C = np.array([[1.0, 2.0], [3.0, 4.0]])
    D = np.array([[0.25, -0.5], [1.5, 2.0]])
    system = StateSpace(A, B, C, D)
    frequency = 2.5

    response = system.frequency_response(frequency)
    dynamic_response = np.column_stack(
        (C[:, 0] / (1.0 + 1j * frequency), C[:, 1] / (2.0 + 1j * frequency))
    )

    assert response.shape == (2, 2)
    np.testing.assert_allclose(response, dynamic_response + D)
    direct_only = StateSpace(A, B, np.zeros((2, 2)), D)
    np.testing.assert_array_equal(
        direct_only.frequency_response(frequency), D.astype(complex)
    )


@pytest.mark.parametrize(
    ("B", "C", "D", "scalar_shape", "vector_shape"),
    [
        (
            np.empty((2, 0)),
            np.eye(2),
            np.empty((2, 0)),
            (2, 0),
            (3, 2, 0),
        ),
        (
            np.eye(2),
            np.empty((0, 2)),
            np.empty((0, 2)),
            (0, 2),
            (3, 0, 2),
        ),
        (
            np.empty((2, 0)),
            np.empty((0, 2)),
            np.empty((0, 0)),
            (0, 0),
            (3, 0, 0),
        ),
    ],
)
def test_frequency_response_preserves_empty_channel_shapes(
    B, C, D, scalar_shape, vector_shape
):
    system = StateSpace(np.diag([-1.0, -2.0]), B, C, D)

    scalar_response = system.frequency_response(1.0)
    vector_response = system.frequency_response([0.0, 1.0, 2.0])

    assert scalar_response.shape == scalar_shape
    assert vector_response.shape == vector_shape
    assert scalar_response.dtype == complex
    assert vector_response.dtype == complex


def test_frequency_response_accepts_nonstable_system_away_from_poles():
    system = StateSpace([[1.0]], [[2.0]], [[3.0]], [[0.0]])

    response = system.frequency_response(2.0)

    assert system.is_asymptotically_stable() is False
    np.testing.assert_allclose(response, [[6.0 / (-1.0 + 2.0j)]])


def test_frequency_response_rejects_frequency_at_imaginary_axis_pole():
    system = StateSpace(
        [[0.0, -1.0], [1.0, 0.0]],
        [[1.0], [0.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )

    with pytest.raises(
        ValueError,
        match=r"frequency response is undefined at angular frequency 1\.0 rad/s",
    ):
        system.frequency_response(1.0)


@pytest.mark.parametrize("frequencies", [np.nan, np.inf, -np.inf, [0.0, np.nan]])
def test_frequency_response_rejects_nonfinite_frequencies(frequencies):
    system = StateSpace(*valid_matrices())

    with pytest.raises(
        ValueError, match="angular frequencies must contain only finite values"
    ):
        system.frequency_response(frequencies)


@pytest.mark.parametrize("frequencies", [1.0 + 0.0j, [0.0, 1.0j], "one"])
def test_frequency_response_rejects_nonreal_frequencies(frequencies):
    system = StateSpace(*valid_matrices())

    with pytest.raises(TypeError, match="angular frequencies must be real"):
        system.frequency_response(frequencies)


@pytest.mark.parametrize("frequencies", [np.empty((0,)), [[0.0, 1.0]]])
def test_frequency_response_rejects_invalid_frequency_dimensions(frequencies):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="angular frequenc"):
        system.frequency_response(frequencies)


def test_frequency_response_singular_value_equals_siso_response_magnitude():
    system = StateSpace([[-2.0]], [[3.0]], [[4.0]], [[0.5]])
    frequency = 1.5

    values = system.frequency_response_singular_values(frequency)

    assert values.shape == (1,)
    assert values.dtype == float
    assert np.all(values >= 0.0)
    np.testing.assert_allclose(
        values, [abs(system.frequency_response(frequency)[0, 0])]
    )


def test_frequency_response_singular_values_match_analytic_diagonal_mimo():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.diag([1.0, 4.0]),
        np.zeros((2, 2)),
    )

    values = system.frequency_response_singular_values(0.0)

    np.testing.assert_allclose(values, [2.0, 1.0])
    assert np.all(np.diff(values) <= 0.0)


def test_frequency_response_singular_values_match_numpy_svd_for_general_mimo():
    system = StateSpace(
        [[-2.0, 1.0], [-1.0, -3.0]],
        [[1.0, 0.5], [0.25, 1.0]],
        [[1.0, 0.2], [-0.4, 1.0], [0.5, -0.3]],
        [[0.1, 0.0], [0.0, -0.2], [0.3, 0.4]],
    )
    frequencies = np.array([3.0, 0.0, 1.5])

    values = system.frequency_response_singular_values(frequencies)
    expected = np.linalg.svd(
        system.frequency_response(frequencies), compute_uv=False
    )

    assert values.shape == (3, 2)
    np.testing.assert_allclose(values, expected)


def test_frequency_response_singular_values_preserve_scalar_and_vector_order():
    system = StateSpace([[-2.0]], [[3.0]], [[4.0]], [[0.5]])
    frequencies = np.array([3.0, 0.0, 1.5])

    scalar_values = system.frequency_response_singular_values(3.0)
    vector_values = system.frequency_response_singular_values(frequencies)

    assert scalar_values.shape == (1,)
    assert vector_values.shape == (3, 1)
    for index, frequency in enumerate(frequencies):
        np.testing.assert_array_equal(
            vector_values[index],
            system.frequency_response_singular_values(frequency),
        )


def test_frequency_response_singular_values_preserve_repeated_multiplicity():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.zeros((2, 2)),
        2.0 * np.eye(2),
    )

    values = system.frequency_response_singular_values(4.0)

    np.testing.assert_array_equal(values, [2.0, 2.0])


@pytest.mark.parametrize(
    ("B", "C", "D"),
    [
        (np.empty((2, 0)), np.eye(2), np.empty((2, 0))),
        (np.eye(2), np.empty((0, 2)), np.empty((0, 2))),
        (np.empty((2, 0)), np.empty((0, 2)), np.empty((0, 0))),
    ],
)
def test_frequency_response_singular_values_preserve_empty_channels(B, C, D):
    system = StateSpace(np.diag([-1.0, -2.0]), B, C, D)

    scalar_values = system.frequency_response_singular_values(1.0)
    vector_values = system.frequency_response_singular_values([0.0, 1.0, 2.0])

    assert scalar_values.shape == (0,)
    assert vector_values.shape == (3, 0)
    assert scalar_values.dtype == float
    assert vector_values.dtype == float


def test_frequency_response_singular_values_include_direct_feedthrough():
    D = np.array([[3.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.ones((2, 3)),
        np.zeros((2, 2)),
        D,
    )

    values = system.frequency_response_singular_values([0.0, 5.0])

    np.testing.assert_allclose(values, [[3.0, 2.0], [3.0, 2.0]])


def test_frequency_response_singular_values_preserve_pole_error():
    system = StateSpace(
        [[0.0, -1.0], [1.0, 0.0]],
        [[1.0], [0.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )

    with pytest.raises(
        ValueError,
        match=r"frequency response is undefined at angular frequency 1\.0 rad/s",
    ):
        system.frequency_response_singular_values(1.0)


@pytest.mark.parametrize(
    ("frequencies", "error_type", "message"),
    [
        (np.nan, ValueError, "must contain only finite values"),
        (1.0 + 0.0j, TypeError, "must be real numeric values"),
        ([[0.0, 1.0]], ValueError, "must be a real scalar or 1D array"),
    ],
)
def test_frequency_response_singular_values_preserve_frequency_validation(
    frequencies, error_type, message
):
    system = StateSpace(*valid_matrices())

    with pytest.raises(error_type, match=message):
        system.frequency_response_singular_values(frequencies)


def test_fully_controllable_system_has_full_controllability_rank():
    system = StateSpace(*valid_matrices())

    matrix = system.controllability_matrix()

    np.testing.assert_array_equal(matrix, [[0.0, 1.0], [1.0, -3.0]])
    assert system.controllability_rank() == 2
    assert system.is_fully_controllable() is True


def test_uncontrollable_system_has_deficient_controllability_rank():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    matrix = system.controllability_matrix()

    np.testing.assert_array_equal(matrix, [[1.0, -1.0], [0.0, 0.0]])
    assert system.controllability_rank() == 1
    assert system.is_fully_controllable() is False


def test_controllability_gramian_matches_diagonal_analytic_solution():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.diag([2.0, 4.0]),
        np.eye(2),
        np.zeros((2, 2)),
    )

    gramian = system.controllability_gramian()

    np.testing.assert_allclose(gramian, np.diag([2.0, 4.0]))
    assert gramian.dtype == float


def test_controllability_gramian_is_symmetric_with_small_lyapunov_residual():
    system = StateSpace(*valid_matrices())

    gramian = system.controllability_gramian()
    residual = system.A @ gramian + gramian @ system.A.T + system.B @ system.B.T

    np.testing.assert_allclose(gramian, gramian.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(residual, np.zeros((2, 2)), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("nonstable_eigenvalue", [0.0, 1.0])
def test_controllability_gramian_rejects_nonstable_system(nonstable_eigenvalue):
    system = StateSpace(
        np.diag([-1.0, nonstable_eigenvalue]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )

    with pytest.raises(
        ValueError,
        match="controllability Gramian requires an asymptotically stable system",
    ):
        system.controllability_gramian()


def test_controllability_gramian_is_zero_without_input_channels():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.empty((2, 0)),
        np.eye(2),
        np.empty((2, 0)),
    )

    gramian = system.controllability_gramian()

    assert gramian.shape == (2, 2)
    np.testing.assert_array_equal(gramian, np.zeros((2, 2)))


def test_fully_controllable_unstable_system_is_stabilizable():
    system = StateSpace(
        [[0.0, 1.0], [2.0, 1.0]],
        [[0.0], [1.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    assert system.is_fully_controllable() is True
    assert np.any(system.eigenvalues().real > 0.0)
    assert system.is_stabilizable() is True


def test_uncontrollable_asymptotically_stable_system_is_stabilizable():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    assert system.is_fully_controllable() is False
    assert system.is_asymptotically_stable() is True
    assert system.is_stabilizable() is True


def test_uncontrollable_unstable_mode_is_not_stabilizable():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    assert system.is_fully_controllable() is False
    assert system.is_stabilizable() is False


def test_uncontrollable_neutral_mode_is_not_stabilizable():
    system = StateSpace(
        np.diag([-1.0, 0.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    assert system.is_fully_controllable() is False
    assert system.is_stabilizable() is False


def test_only_stable_modes_may_be_uncontrollable_in_multistate_system():
    system = StateSpace(
        np.diag([-3.0, -1.0, 0.0, 2.0]),
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        np.eye(4),
        np.zeros((4, 2)),
    )

    assert system.controllability_rank() == 2
    assert system.is_stabilizable() is True


@pytest.mark.parametrize(
    ("matrix_index", "nonfinite_value"),
    [(0, np.nan), (1, np.inf), (2, -np.inf), (3, np.nan)],
)
def test_state_space_rejects_nonfinite_system_data(matrix_index, nonfinite_value):
    matrices = [matrix.astype(float) for matrix in valid_matrices()]
    matrices[matrix_index].flat[0] = nonfinite_value

    with pytest.raises(
        ValueError, match="A, B, C, and D must contain only finite values"
    ):
        StateSpace(*matrices)


def test_fully_observable_system_has_full_observability_rank():
    system = StateSpace(*valid_matrices())

    matrix = system.observability_matrix()

    np.testing.assert_array_equal(matrix, [[1.0, 0.0], [0.0, 1.0]])
    assert system.observability_rank() == 2
    assert system.is_fully_observable() is True


def test_unobservable_system_has_deficient_observability_rank():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    matrix = system.observability_matrix()

    np.testing.assert_array_equal(matrix, [[1.0, 0.0], [-1.0, 0.0]])
    assert system.observability_rank() == 1
    assert system.is_fully_observable() is False


def test_observability_gramian_matches_diagonal_analytic_solution():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.diag([2.0, 4.0]),
        np.zeros((2, 2)),
    )

    gramian = system.observability_gramian()

    np.testing.assert_allclose(gramian, np.diag([2.0, 4.0]))
    assert gramian.dtype == float


def test_observability_gramian_is_symmetric_with_small_lyapunov_residual():
    system = StateSpace(*valid_matrices())

    gramian = system.observability_gramian()
    residual = system.A.T @ gramian + gramian @ system.A + system.C.T @ system.C

    np.testing.assert_allclose(gramian, gramian.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(residual, np.zeros((2, 2)), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("nonstable_eigenvalue", [0.0, 1.0])
def test_observability_gramian_rejects_nonstable_system(nonstable_eigenvalue):
    system = StateSpace(
        np.diag([-1.0, nonstable_eigenvalue]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )

    with pytest.raises(
        ValueError,
        match="observability Gramian requires an asymptotically stable system",
    ):
        system.observability_gramian()


def test_observability_gramian_is_zero_without_output_channels():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.empty((0, 2)),
        np.empty((0, 2)),
    )

    gramian = system.observability_gramian()

    assert gramian.shape == (2, 2)
    np.testing.assert_array_equal(gramian, np.zeros((2, 2)))


def test_hankel_singular_values_match_ordered_diagonal_analytic_solution():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.diag([2.0, 4.0]),
        np.diag([3.0, 5.0]),
        np.zeros((2, 2)),
    )

    values = system.hankel_singular_values()

    np.testing.assert_allclose(values, [5.0, 3.0])
    assert values.shape == (2,)
    assert values.dtype == float
    assert np.all(values >= 0.0)


def test_hankel_singular_values_preserve_repeated_value_multiplicity():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.diag([np.sqrt(2.0), 2.0]),
        np.diag([np.sqrt(2.0), 2.0]),
        np.zeros((2, 2)),
    )

    np.testing.assert_allclose(system.hankel_singular_values(), [1.0, 1.0])


@pytest.mark.parametrize("empty_channel", ["input", "output"])
def test_hankel_singular_values_are_zero_with_empty_channel(empty_channel):
    B = np.empty((2, 0)) if empty_channel == "input" else np.eye(2)
    C = np.empty((0, 2)) if empty_channel == "output" else np.eye(2)
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        B,
        C,
        np.empty((C.shape[0], B.shape[1])),
    )

    np.testing.assert_array_equal(system.hankel_singular_values(), np.zeros(2))


def test_hankel_singular_values_are_invariant_under_state_similarity():
    system = StateSpace(*valid_matrices())
    transformation = np.array([[2.0, 1.0], [0.0, 1.0]])
    inverse_transformation = np.linalg.inv(transformation)
    transformed_system = StateSpace(
        transformation @ system.A @ inverse_transformation,
        transformation @ system.B,
        system.C @ inverse_transformation,
        system.D,
    )

    np.testing.assert_allclose(
        transformed_system.hankel_singular_values(),
        system.hankel_singular_values(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("nonstable_eigenvalue", [0.0, 1.0])
def test_hankel_singular_values_reject_nonstable_system(nonstable_eigenvalue):
    system = StateSpace(
        np.diag([-1.0, nonstable_eigenvalue]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )

    with pytest.raises(
        ValueError,
        match="controllability Gramian requires an asymptotically stable system",
    ):
        system.hankel_singular_values()


def test_unreachable_stable_state_produces_one_zero_hankel_singular_value():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    values = system.hankel_singular_values()

    assert system.controllability_rank() == 1
    assert system.observability_rank() == 2
    assert np.count_nonzero(values == 0.0) == (
        system.n_states - system.controllability_rank()
    )
    np.testing.assert_allclose(values, [0.5, 0.0])


def test_unobservable_stable_state_produces_one_zero_hankel_singular_value():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    values = system.hankel_singular_values()

    assert system.controllability_rank() == 2
    assert system.observability_rank() == 1
    assert np.count_nonzero(values == 0.0) == (
        system.n_states - system.observability_rank()
    )
    np.testing.assert_allclose(values, [0.5, 0.0])


def test_distinct_unreachable_and_unobservable_states_produce_two_zero_values():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        np.zeros((2, 2)),
    )

    values = system.hankel_singular_values()
    controllability_deficiency = system.n_states - system.controllability_rank()
    observability_deficiency = system.n_states - system.observability_rank()

    assert controllability_deficiency == 1
    assert observability_deficiency == 1
    assert np.count_nonzero(values == 0.0) == (
        controllability_deficiency + observability_deficiency
    )
    np.testing.assert_allclose(values, [0.5, 0.0, 0.0])


def test_stable_minimal_realization_has_strictly_positive_hankel_values():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.eye(3),
        np.eye(3),
        np.zeros((3, 3)),
    )

    values = system.hankel_singular_values()

    assert system.controllability_rank() == system.n_states
    assert system.observability_rank() == system.n_states
    assert system.is_minimal_realization() is True
    assert np.all(values > 0.0)
    np.testing.assert_allclose(values, [0.5, 0.25, 1.0 / 6.0])


def test_stable_minimal_system_produces_full_order_balanced_realization():
    system = StateSpace(*stable_minimal_balancing_matrices())

    result = system.balanced_realization()

    assert isinstance(result, BalancedRealization)
    assert isinstance(result.system, StateSpace)
    assert result.transformation.shape == (system.n_states, system.n_states)
    assert np.linalg.matrix_rank(result.transformation) == system.n_states
    assert all(
        np.isrealobj(matrix)
        for matrix in (
            result.transformation,
            result.system.A,
            result.system.B,
            result.system.C,
            result.system.D,
        )
    )


def test_balanced_realization_follows_x_equals_transformation_z_convention():
    system = StateSpace(*stable_minimal_balancing_matrices())

    balanced, transformation = system.balanced_realization()

    np.testing.assert_allclose(
        balanced.A, np.linalg.solve(transformation, system.A @ transformation)
    )
    np.testing.assert_allclose(
        balanced.B, np.linalg.solve(transformation, system.B)
    )
    np.testing.assert_allclose(balanced.C, system.C @ transformation)
    np.testing.assert_array_equal(balanced.D, system.D)


def test_balanced_realization_preserves_input_output_behavior():
    system = StateSpace(*stable_minimal_balancing_matrices())
    balanced, transformation = system.balanced_realization()
    original_initial_state = np.array([0.7, -0.3])
    balanced_initial_state = np.linalg.solve(
        transformation, original_initial_state
    )
    time = np.array([0.0, 0.07, 0.19, 0.34, 0.58])
    inputs = np.array(
        [[0.2, -0.1], [0.0, 0.3], [-0.4, 0.2], [0.1, 0.0], [0.5, -0.2]]
    )

    original_states, original_outputs = system.simulate(
        original_initial_state, inputs, time, method="exact"
    )
    balanced_states, balanced_outputs = balanced.simulate(
        balanced_initial_state, inputs, time, method="exact"
    )

    np.testing.assert_allclose(
        balanced_states @ transformation.T,
        original_states,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        balanced_outputs, original_outputs, rtol=1e-12, atol=1e-12
    )


def test_balanced_gramians_equal_diagonal_hankel_singular_values():
    system = StateSpace(*stable_minimal_balancing_matrices())
    balanced = system.balanced_realization().system
    expected = np.diag(system.hankel_singular_values())

    controllability_gramian = balanced.controllability_gramian()
    observability_gramian = balanced.observability_gramian()

    np.testing.assert_allclose(
        controllability_gramian, observability_gramian, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        controllability_gramian, expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        balanced.hankel_singular_values(),
        system.hankel_singular_values(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_balanced_realization_preserves_dimensions_and_eigenvalues():
    system = StateSpace(*stable_minimal_balancing_matrices())
    balanced = system.balanced_realization().system

    assert balanced.n_states == system.n_states
    assert balanced.n_inputs == system.n_inputs
    assert balanced.n_outputs == system.n_outputs
    np.testing.assert_allclose(
        np.sort_complex(balanced.eigenvalues()),
        np.sort_complex(system.eigenvalues()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_balanced_realization_rejects_nonstable_system():
    system = StateSpace(
        np.diag([-1.0, 0.0]), np.eye(2), np.eye(2), np.zeros((2, 2))
    )

    with pytest.raises(
        ValueError,
        match="balanced realization requires an asymptotically stable system",
    ):
        system.balanced_realization()


@pytest.mark.parametrize("deficiency", ["unreachable", "unobservable"])
def test_balanced_realization_rejects_nonminimal_system(deficiency):
    B = [[1.0], [0.0]] if deficiency == "unreachable" else [[1.0], [1.0]]
    C = [[1.0, 1.0]] if deficiency == "unreachable" else [[1.0, 0.0]]
    system = StateSpace(np.diag([-1.0, -2.0]), B, C, np.zeros((1, 1)))

    assert system.is_minimal_realization() is False
    with pytest.raises(
        ValueError, match="balanced realization requires a minimal realization"
    ):
        system.balanced_realization()


def test_balanced_realization_rejects_singular_gramian_factor(monkeypatch):
    system = StateSpace(*stable_minimal_balancing_matrices())
    monkeypatch.setattr(
        system, "controllability_gramian", lambda: np.diag([1.0, 0.0])
    )

    with pytest.raises(
        ValueError,
        match="requires numerically positive-definite Gramian factors",
    ):
        system.balanced_realization()


def test_balanced_truncation_uses_exact_leading_balanced_blocks():
    system = StateSpace(*stable_minimal_balancing_matrices())
    balanced = system.balanced_realization().system

    result = system.balanced_truncation(1)

    assert isinstance(result, BalancedTruncation)
    assert result.retained_order == 1
    np.testing.assert_array_equal(result.system.A, balanced.A[:1, :1])
    np.testing.assert_array_equal(result.system.B, balanced.B[:1, :])
    np.testing.assert_array_equal(result.system.C, balanced.C[:, :1])
    np.testing.assert_array_equal(result.system.D, balanced.D)
    np.testing.assert_array_equal(result.system.D, system.D)


def test_balanced_truncation_maps_follow_documented_coordinate_convention():
    system = StateSpace(*stable_minimal_balancing_matrices())
    result = system.balanced_truncation(1)
    original_state = np.array([0.7, -0.3])
    balanced_state = np.linalg.solve(
        result.balanced_transformation, original_state
    )

    reduced_state = result.projection @ original_state
    reconstructed_state = result.reconstruction @ reduced_state
    expected_reconstruction = (
        result.balanced_transformation @ np.array([balanced_state[0], 0.0])
    )

    np.testing.assert_allclose(reduced_state, balanced_state[:1])
    np.testing.assert_allclose(reconstructed_state, expected_reconstruction)
    np.testing.assert_allclose(
        result.projection @ result.reconstruction, np.eye(1)
    )


def test_balanced_truncation_returns_reduced_system_dimensions():
    system = StateSpace(*stable_minimal_balancing_matrices())

    result = system.balanced_truncation(1)

    assert result.system.A.shape == (1, 1)
    assert result.system.B.shape == (1, system.n_inputs)
    assert result.system.C.shape == (system.n_outputs, 1)
    assert result.system.D.shape == (system.n_outputs, system.n_inputs)
    assert result.projection.shape == (1, system.n_states)
    assert result.reconstruction.shape == (system.n_states, 1)


def test_balanced_truncation_retains_largest_hankel_singular_values():
    system = StateSpace(*stable_minimal_balancing_matrices())

    result = system.balanced_truncation(1)
    all_values = system.hankel_singular_values()

    assert np.all(np.diff(all_values) <= 0.0)
    np.testing.assert_array_equal(
        result.retained_hankel_singular_values, all_values[:1]
    )
    np.testing.assert_allclose(
        result.retained_hankel_singular_values,
        np.diag(system.balanced_realization().system.controllability_gramian())[:1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_balanced_truncation_error_bound_matches_analytic_diagonal_system():
    hankel_values = np.array([3.0, 0.5, 0.125])
    decay_rates = np.array([1.0, 2.0, 4.0])
    channel_gains = np.sqrt(2.0 * decay_rates * hankel_values)
    system = StateSpace(
        -np.diag(decay_rates),
        np.diag(channel_gains),
        np.diag(channel_gains),
        np.zeros((3, 3)),
    )

    result = system.balanced_truncation(1)

    np.testing.assert_allclose(system.hankel_singular_values(), hankel_values)
    np.testing.assert_allclose(
        result.discarded_hankel_singular_values, [0.5, 0.125]
    )
    assert result.a_priori_error_bound == pytest.approx(2.0 * (0.5 + 0.125))


def test_balanced_truncation_error_bound_does_not_increase_with_retained_order():
    hankel_values = np.array([3.0, 0.5, 0.125])
    decay_rates = np.array([1.0, 2.0, 4.0])
    channel_gains = np.sqrt(2.0 * decay_rates * hankel_values)
    system = StateSpace(
        -np.diag(decay_rates),
        np.diag(channel_gains),
        np.diag(channel_gains),
        np.zeros((3, 3)),
    )

    order_one_bound = system.balanced_truncation(1).a_priori_error_bound
    order_two_bound = system.balanced_truncation(2).a_priori_error_bound

    assert order_two_bound <= order_one_bound
    assert order_two_bound == pytest.approx(2.0 * 0.125)


def test_balanced_truncation_error_bound_is_finite_real_and_nonnegative():
    system = StateSpace(*stable_minimal_balancing_matrices())

    bound = system.balanced_truncation(1).a_priori_error_bound

    assert isinstance(bound, float)
    assert np.isreal(bound)
    assert np.isfinite(bound)
    assert bound >= 0.0


def test_zero_discarded_sum_is_full_order_fact_not_valid_truncation():
    system = StateSpace(*stable_minimal_balancing_matrices())
    values = system.hankel_singular_values()

    assert float(2.0 * np.sum(values[system.n_states :])) == 0.0
    assert system.balanced_truncation(1).a_priori_error_bound > 0.0
    with pytest.raises(
        ValueError, match="retained order must satisfy 1 <= r < n_states"
    ):
        system.balanced_truncation(system.n_states)


@pytest.mark.parametrize("retained_order", [0, -1, 2, 3])
def test_balanced_truncation_rejects_orders_outside_reduced_range(retained_order):
    system = StateSpace(*stable_minimal_balancing_matrices())

    with pytest.raises(
        ValueError, match="retained order must satisfy 1 <= r < n_states"
    ):
        system.balanced_truncation(retained_order)


@pytest.mark.parametrize("retained_order", [1.0, "1", None, True])
def test_balanced_truncation_rejects_noninteger_orders(retained_order):
    system = StateSpace(*stable_minimal_balancing_matrices())

    with pytest.raises(TypeError, match="retained order must be an integer"):
        system.balanced_truncation(retained_order)


def test_balanced_truncation_rejects_nonstable_system():
    system = StateSpace(
        np.diag([-1.0, 0.0]), np.eye(2), np.eye(2), np.zeros((2, 2))
    )

    with pytest.raises(
        ValueError,
        match="balanced realization requires an asymptotically stable system",
    ):
        system.balanced_truncation(1)


@pytest.mark.parametrize("deficiency", ["unreachable", "unobservable"])
def test_balanced_truncation_rejects_nonminimal_system(deficiency):
    B = [[1.0], [0.0]] if deficiency == "unreachable" else [[1.0], [1.0]]
    C = [[1.0, 1.0]] if deficiency == "unreachable" else [[1.0, 0.0]]
    system = StateSpace(np.diag([-1.0, -2.0]), B, C, np.zeros((1, 1)))

    with pytest.raises(
        ValueError, match="balanced realization requires a minimal realization"
    ):
        system.balanced_truncation(1)


def test_balanced_truncation_is_stable_and_approximates_small_discarded_state():
    discarded_hankel_value = 1e-3
    system = StateSpace(
        np.diag([-1.0, -10.0]),
        np.diag([np.sqrt(2.0), np.sqrt(20.0 * discarded_hankel_value)]),
        np.diag([np.sqrt(2.0), np.sqrt(20.0 * discarded_hankel_value)]),
        np.zeros((2, 2)),
    )
    time = np.linspace(0.0, 5.0, 101)
    step_input = np.ones(2)

    result = system.balanced_truncation(1)
    _, full_outputs = system.step_response(step_input, time, method="exact")
    _, reduced_outputs = result.system.step_response(
        step_input, time, method="exact"
    )

    assert result.system.is_asymptotically_stable() is True
    np.testing.assert_allclose(result.retained_hankel_singular_values, [1.0])
    np.testing.assert_allclose(
        result.discarded_hankel_singular_values, [discarded_hankel_value]
    )
    assert result.a_priori_error_bound == pytest.approx(
        2.0 * discarded_hankel_value
    )
    assert np.max(np.abs(reduced_outputs - full_outputs)) <= (
        result.a_priori_error_bound
    )


def test_fully_observable_unstable_system_is_detectable():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        np.eye(2),
        [[1.0, 1.0]],
        np.zeros((1, 2)),
    )

    assert system.is_fully_observable() is True
    assert np.any(system.eigenvalues().real > 0.0)
    assert system.is_detectable() is True


def test_unobservable_asymptotically_stable_system_is_detectable():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    assert system.is_fully_observable() is False
    assert system.is_asymptotically_stable() is True
    assert system.is_detectable() is True


def test_unobservable_unstable_mode_is_not_detectable():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    assert system.is_fully_observable() is False
    assert system.is_detectable() is False


def test_unobservable_neutral_mode_is_not_detectable():
    system = StateSpace(
        np.diag([-1.0, 0.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    assert system.is_fully_observable() is False
    assert system.is_detectable() is False


def test_only_stable_modes_may_be_unobservable_in_multistate_system():
    system = StateSpace(
        np.diag([-3.0, -1.0, 0.0, 2.0]),
        np.eye(4),
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        np.zeros((2, 4)),
    )

    assert system.observability_rank() == 2
    assert system.is_detectable() is True


@pytest.mark.parametrize(
    ("controllable", "observable", "expected"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_minimal_realization_requires_controllability_and_observability(
    controllable, observable, expected
):
    B = [[1.0], [1.0]] if controllable else [[1.0], [0.0]]
    C = [[1.0, 1.0]] if observable else [[1.0, 0.0]]
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        B,
        C,
        np.zeros((1, 1)),
    )

    assert system.is_fully_controllable() is controllable
    assert system.is_fully_observable() is observable
    assert system.is_minimal_realization() is expected


@pytest.mark.parametrize(
    ("A", "B", "C", "expected"),
    [
        (
            [[0.0, 1.0], [-2.0, -3.0]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            StructuralAnalysis(True, True, True, True, True),
        ),
        (
            np.diag([-1.0, -2.0]),
            [[1.0], [0.0]],
            np.eye(2),
            StructuralAnalysis(False, True, False, True, True),
        ),
        (
            np.diag([-1.0, -2.0]),
            np.eye(2),
            [[1.0, 0.0]],
            StructuralAnalysis(True, False, False, True, True),
        ),
        (
            np.diag([-1.0, 1.0]),
            [[1.0], [0.0]],
            [[1.0, 0.0]],
            StructuralAnalysis(False, False, False, False, False),
        ),
    ],
)
def test_structural_analysis_agrees_with_existing_checks(A, B, C, expected):
    system = StateSpace(A, B, C, np.zeros((len(C), np.asarray(B).shape[1])))

    analysis = system.structural_analysis()

    assert isinstance(analysis, StructuralAnalysis)
    assert analysis == expected
    assert analysis.controllable is system.is_fully_controllable()
    assert analysis.observable is system.is_fully_observable()
    assert analysis.minimal is system.is_minimal_realization()
    assert analysis.stabilizable is system.is_stabilizable()
    assert analysis.detectable is system.is_detectable()


def test_structural_analysis_is_immutable():
    analysis = StateSpace(*valid_matrices()).structural_analysis()

    with pytest.raises(AttributeError):
        analysis.minimal = False


def test_nonstable_pbh_diagnostic_reports_controllability_failure_only():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        [[1.0], [0.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )

    assert system.nonstable_pbh_diagnostics() == (
        NonstablePBHDiagnostic(2.0, True, False),
    )


def test_nonstable_pbh_diagnostic_reports_observability_failure_only():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        np.eye(2),
        [[1.0, 0.0]],
        np.zeros((1, 2)),
    )

    assert system.nonstable_pbh_diagnostics() == (
        NonstablePBHDiagnostic(2.0, False, True),
    )


def test_nonstable_pbh_diagnostic_reports_both_failures():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        [[1.0], [0.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )

    assert system.nonstable_pbh_diagnostics() == (
        NonstablePBHDiagnostic(2.0, True, True),
    )


def test_nonstable_pbh_diagnostic_includes_neutral_failure():
    system = StateSpace(
        np.diag([-1.0, 0.0]),
        [[1.0], [0.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )

    assert system.nonstable_pbh_diagnostics() == (
        NonstablePBHDiagnostic(0.0, True, True),
    )


def test_nonstable_pbh_diagnostics_omit_stable_failures():
    system = StateSpace(
        np.diag([-2.0, 1.0]),
        [[0.0], [1.0]],
        [[0.0, 1.0]],
        [[0.0]],
    )

    assert system.is_fully_controllable() is False
    assert system.is_fully_observable() is False
    assert system.nonstable_pbh_diagnostics() == ()


def test_nonstable_pbh_diagnostics_omit_nonstable_modes_that_pass():
    system = StateSpace(
        np.diag([-1.0, 2.0]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )

    assert system.is_stabilizable() is True
    assert system.is_detectable() is True
    assert system.nonstable_pbh_diagnostics() == ()


def test_nonstable_pbh_diagnostics_are_immutable_and_deterministic():
    system = StateSpace(
        np.diag([2.0, 0.0, -1.0]),
        np.zeros((3, 1)),
        np.zeros((1, 3)),
        [[0.0]],
    )

    diagnostics = system.nonstable_pbh_diagnostics()

    assert diagnostics == (
        NonstablePBHDiagnostic(2.0, True, True),
        NonstablePBHDiagnostic(0.0, True, True),
    )
    with pytest.raises(AttributeError):
        diagnostics[0].controllability_failed = False
    with pytest.raises(TypeError):
        diagnostics[0] = diagnostics[1]


def test_nonstable_pbh_diagnostics_preserve_unstable_complex_pair_order():
    system = StateSpace(
        [[1.0, -2.0], [2.0, 1.0]],
        np.zeros((2, 1)),
        np.zeros((1, 2)),
        [[0.0]],
    )

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()

    assert len(diagnostics) == 2
    np.testing.assert_array_equal(
        [diagnostic.eigenvalue for diagnostic in diagnostics], eigenvalues
    )
    assert eigenvalues[0] != eigenvalues[1]
    assert eigenvalues[0] == np.conj(eigenvalues[1])
    assert all(
        diagnostic.controllability_failed
        and diagnostic.observability_failed
        for diagnostic in diagnostics
    )


def test_nonstable_pbh_diagnostics_preserve_repeated_eigenvalue_multiplicity():
    system = StateSpace(
        np.diag([2.0, 2.0, -1.0]),
        np.zeros((3, 1)),
        np.zeros((1, 3)),
        [[0.0]],
    )

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()

    assert diagnostics == (
        NonstablePBHDiagnostic(eigenvalues[0], True, True),
        NonstablePBHDiagnostic(eigenvalues[1], True, True),
    )
    np.testing.assert_array_equal(eigenvalues, [2.0, 2.0, -1.0])


def test_nonstable_pbh_diagnostics_preserve_defective_eigenvalue_multiplicity():
    system = StateSpace(
        [[2.0, 1.0], [0.0, 2.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()

    np.testing.assert_array_equal(eigenvalues, [2.0, 2.0])
    assert np.linalg.matrix_rank(system.A - eigenvalues[0] * np.eye(2)) == 1
    assert diagnostics == tuple(
        NonstablePBHDiagnostic(eigenvalue, True, False)
        for eigenvalue in eigenvalues
    )


def test_nonstable_pbh_diagnostics_distinguish_all_nonstable_outcomes():
    system = StateSpace(
        np.diag([1.0, 2.0, 3.0, 4.0]),
        [[0.0], [1.0], [0.0], [1.0]],
        [[1.0, 0.0, 0.0, 1.0]],
        [[0.0]],
    )

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()

    np.testing.assert_array_equal(eigenvalues, [1.0, 2.0, 3.0, 4.0])
    assert diagnostics == (
        NonstablePBHDiagnostic(eigenvalues[0], True, False),
        NonstablePBHDiagnostic(eigenvalues[1], False, True),
        NonstablePBHDiagnostic(eigenvalues[2], True, True),
    )
    assert all(diagnostic.eigenvalue != eigenvalues[3] for diagnostic in diagnostics)


@pytest.mark.parametrize(
    ("B", "C", "expected_stabilizable", "expected_detectable"),
    [
        ([[0.0], [1.0]], [[0.0, 1.0]], True, True),
        ([[1.0], [0.0]], [[0.0, 1.0]], False, True),
        ([[0.0], [1.0]], [[1.0, 0.0]], True, False),
        ([[1.0], [0.0]], [[1.0, 0.0]], False, False),
    ],
)
def test_nonstable_pbh_diagnostics_agree_with_structural_predicates(
    B, C, expected_stabilizable, expected_detectable
):
    system = StateSpace(np.diag([-1.0, 2.0]), B, C, [[0.0]])

    diagnostics = system.nonstable_pbh_diagnostics()
    has_controllability_failure = any(
        diagnostic.controllability_failed for diagnostic in diagnostics
    )
    has_observability_failure = any(
        diagnostic.observability_failed for diagnostic in diagnostics
    )

    assert system.is_stabilizable() is expected_stabilizable
    assert system.is_detectable() is expected_detectable
    assert system.is_stabilizable() is (not has_controllability_failure)
    assert system.is_detectable() is (not has_observability_failure)
    if expected_stabilizable and expected_detectable:
        assert diagnostics == ()


@pytest.mark.parametrize(
    ("B", "C", "controllability_failed", "observability_failed"),
    [
        ([[0.0], [1.0]], [[0.0, 1.0]], False, False),
        ([[1.0], [0.0]], [[0.0, 1.0]], True, False),
        ([[0.0], [1.0]], [[1.0, 0.0]], False, True),
        ([[1.0], [0.0]], [[1.0, 0.0]], True, True),
    ],
)
def test_neutral_pbh_diagnostics_agree_with_structural_predicates(
    B, C, controllability_failed, observability_failed
):
    system = StateSpace(np.diag([-1.0, 0.0]), B, C, [[0.0]])

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()
    expected_diagnostics = (
        (NonstablePBHDiagnostic(eigenvalues[1], controllability_failed, observability_failed),)
        if controllability_failed or observability_failed
        else ()
    )

    np.testing.assert_array_equal(eigenvalues, [-1.0, 0.0])
    assert diagnostics == expected_diagnostics
    assert system.is_stabilizable() is (not controllability_failed)
    assert system.is_detectable() is (not observability_failed)
    assert system.is_stabilizable() is (
        not any(diagnostic.controllability_failed for diagnostic in diagnostics)
    )
    assert system.is_detectable() is (
        not any(diagnostic.observability_failed for diagnostic in diagnostics)
    )


@pytest.mark.parametrize(
    ("B", "C", "controllability_failed", "observability_failed"),
    [
        ([[1.0], [0.0]], [[1.0, 0.0]], False, False),
        (np.zeros((2, 1)), [[1.0, 0.0]], True, False),
        ([[1.0], [0.0]], np.zeros((1, 2)), False, True),
        (np.zeros((2, 1)), np.zeros((1, 2)), True, True),
    ],
)
def test_purely_imaginary_pbh_diagnostics_agree_with_structural_predicates(
    B, C, controllability_failed, observability_failed
):
    system = StateSpace([[0.0, -2.0], [2.0, 0.0]], B, C, [[0.0]])

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()
    expected_diagnostics = (
        tuple(
            NonstablePBHDiagnostic(
                eigenvalue, controllability_failed, observability_failed
            )
            for eigenvalue in eigenvalues
        )
        if controllability_failed or observability_failed
        else ()
    )

    np.testing.assert_array_equal(eigenvalues.real, [0.0, 0.0])
    np.testing.assert_allclose(np.abs(eigenvalues.imag), [2.0, 2.0])
    assert eigenvalues[0] == np.conj(eigenvalues[1])
    assert diagnostics == expected_diagnostics
    assert system.is_stabilizable() is (not controllability_failed)
    assert system.is_detectable() is (not observability_failed)
    assert system.is_stabilizable() is (
        not any(diagnostic.controllability_failed for diagnostic in diagnostics)
    )
    assert system.is_detectable() is (
        not any(diagnostic.observability_failed for diagnostic in diagnostics)
    )


@pytest.mark.parametrize(
    ("B", "C", "D", "controllability_failed", "observability_failed"),
    [
        (np.empty((3, 0)), np.eye(3), np.empty((3, 0)), True, False),
        (np.eye(3), np.empty((0, 3)), np.empty((0, 3)), False, True),
        (np.empty((3, 0)), np.empty((0, 3)), np.empty((0, 0)), True, True),
    ],
)
def test_empty_channels_report_ordered_nonstable_pbh_failures(
    B, C, D, controllability_failed, observability_failed
):
    system = StateSpace(np.diag([-1.0, 0.0, 2.0]), B, C, D)

    eigenvalues = system.eigenvalues()
    diagnostics = system.nonstable_pbh_diagnostics()

    assert system.B.shape == B.shape
    assert system.C.shape == C.shape
    assert system.D.shape == D.shape
    np.testing.assert_array_equal(eigenvalues, [-1.0, 0.0, 2.0])
    assert diagnostics == tuple(
        NonstablePBHDiagnostic(
            eigenvalue, controllability_failed, observability_failed
        )
        for eigenvalue in eigenvalues[1:]
    )
    assert system.is_stabilizable() is (not controllability_failed)
    assert system.is_detectable() is (not observability_failed)


def test_stable_empty_channel_system_has_no_nonstable_pbh_failures():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.empty((2, 0)),
        np.empty((0, 2)),
        np.empty((0, 0)),
    )

    assert system.B.shape == (2, 0)
    assert system.C.shape == (0, 2)
    assert system.D.shape == (0, 0)
    assert system.is_asymptotically_stable() is True
    assert system.is_fully_controllable() is False
    assert system.is_fully_observable() is False
    assert system.is_stabilizable() is True
    assert system.is_detectable() is True
    assert system.nonstable_pbh_diagnostics() == ()


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


def test_modal_properties_for_stable_complex_conjugate_pair():
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    modes = system.modal_properties()

    assert len(modes) == 2
    np.testing.assert_allclose([mode.eigenvalue for mode in modes], system.eigenvalues())
    for mode in modes:
        assert mode.natural_frequency == pytest.approx(np.sqrt(5.0))
        assert mode.damping_ratio == pytest.approx(1.0 / np.sqrt(5.0))
        assert mode.damped_natural_frequency == pytest.approx(2.0)
        assert mode.period == pytest.approx(np.pi)
        assert mode.time_constant == pytest.approx(1.0)
    assert modes[0][1:] == pytest.approx(modes[1][1:])


def test_modal_properties_for_unstable_complex_conjugate_pair():
    system = StateSpace(
        [[1.0, -2.0], [2.0, 1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    modes = system.modal_properties()

    for mode in modes:
        assert mode.natural_frequency == pytest.approx(np.sqrt(5.0))
        assert mode.damping_ratio == pytest.approx(-1.0 / np.sqrt(5.0))
        assert mode.damped_natural_frequency == pytest.approx(2.0)
        assert mode.period == pytest.approx(np.pi)
        assert mode.time_constant == pytest.approx(-1.0)
    assert modes[0][1:] == pytest.approx(modes[1][1:])


@pytest.mark.parametrize(
    ("eigenvalue", "expected_time_constant"),
    [(-2.0, 0.5), (4.0, -0.25)],
)
def test_modal_properties_for_real_eigenvalue(eigenvalue, expected_time_constant):
    system = StateSpace([[eigenvalue]], [[0.0]], [[1.0]], [[0.0]])

    mode = system.modal_properties()[0]

    assert mode.eigenvalue == complex(eigenvalue)
    assert mode.time_constant == pytest.approx(expected_time_constant)
    assert mode.natural_frequency is None
    assert mode.damping_ratio is None
    assert mode.damped_natural_frequency is None
    assert mode.period is None


def test_modal_properties_for_zero_eigenvalue_are_not_applicable():
    system = StateSpace([[0.0]], [[0.0]], [[1.0]], [[0.0]])

    mode = system.modal_properties()[0]

    assert mode.eigenvalue == 0j
    assert mode.natural_frequency is None
    assert mode.damping_ratio is None
    assert mode.damped_natural_frequency is None
    assert mode.period is None
    assert mode.time_constant is None


def test_right_eigenvectors_for_diagonal_matrix_have_expected_modal_subspaces():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    eigenvectors = system.right_eigenvectors()

    assert eigenvectors.shape == (3, 3)
    np.testing.assert_allclose(np.abs(eigenvectors), np.eye(3))
    np.testing.assert_allclose(np.linalg.norm(eigenvectors, axis=0), np.ones(3))


def test_right_eigenvectors_for_non_diagonal_system_satisfy_eigenpair_equation():
    system = StateSpace(
        [[2.0, 1.0], [0.0, 3.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    eigenvalues = system.eigenvalues()
    eigenvectors = system.right_eigenvectors()

    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index]
        np.testing.assert_allclose(system.A @ vector, eigenvalue * vector)
    np.testing.assert_allclose(np.linalg.norm(eigenvectors, axis=0), np.ones(2))


def test_right_eigenvectors_preserve_complex_conjugate_modes_and_ordering():
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    eigenvalues = system.eigenvalues()
    eigenvectors = system.right_eigenvectors()
    modal_eigenvalues = np.array(
        [mode.eigenvalue for mode in system.modal_properties()]
    )

    assert np.iscomplexobj(eigenvectors)
    np.testing.assert_array_equal(modal_eigenvalues, eigenvalues)
    np.testing.assert_allclose(eigenvalues[0], np.conj(eigenvalues[1]))
    np.testing.assert_allclose(
        np.abs(eigenvectors[:, 0]), np.abs(eigenvectors[:, 1])
    )
    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index]
        np.testing.assert_allclose(system.A @ vector, eigenvalue * vector)
    np.testing.assert_allclose(np.linalg.norm(eigenvectors, axis=0), np.ones(2))


def test_left_eigenvectors_for_diagonal_matrix_have_expected_modal_subspaces():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    eigenvectors = system.left_eigenvectors()

    assert eigenvectors.shape == (3, 3)
    assert np.all(np.isfinite(eigenvectors.real))
    assert np.all(np.isfinite(eigenvectors.imag))
    np.testing.assert_allclose(np.abs(eigenvectors), np.eye(3))
    np.testing.assert_allclose(np.linalg.norm(eigenvectors, axis=0), np.ones(3))


def test_left_eigenvectors_differ_from_right_for_nonsymmetric_system():
    system = StateSpace(
        [[2.0, 1.0], [0.0, 3.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    eigenvalues = system.eigenvalues()
    left_eigenvectors = system.left_eigenvectors()
    right_eigenvectors = system.right_eigenvectors()

    assert not np.allclose(np.abs(left_eigenvectors), np.abs(right_eigenvectors))
    for index, eigenvalue in enumerate(eigenvalues):
        vector = left_eigenvectors[:, index]
        np.testing.assert_allclose(
            vector.conj().T @ system.A, eigenvalue * vector.conj().T
        )
    np.testing.assert_allclose(
        np.linalg.norm(left_eigenvectors, axis=0), np.ones(2)
    )


def test_left_eigenvectors_preserve_complex_modes_and_modal_ordering():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    eigenvalues = system.eigenvalues()
    left_eigenvectors = system.left_eigenvectors()
    modal_eigenvalues = np.array(
        [mode.eigenvalue for mode in system.modal_properties()]
    )

    assert left_eigenvectors.shape == (2, 2)
    assert np.iscomplexobj(left_eigenvectors)
    assert np.all(np.isfinite(left_eigenvectors.real))
    assert np.all(np.isfinite(left_eigenvectors.imag))
    np.testing.assert_array_equal(modal_eigenvalues, eigenvalues)
    np.testing.assert_allclose(eigenvalues[0], np.conj(eigenvalues[1]))
    np.testing.assert_allclose(
        np.abs(left_eigenvectors[:, 0]), np.abs(left_eigenvectors[:, 1])
    )
    for index, eigenvalue in enumerate(eigenvalues):
        vector = left_eigenvectors[:, index]
        np.testing.assert_allclose(
            vector.conj().T @ system.A, eigenvalue * vector.conj().T
        )
    np.testing.assert_allclose(
        np.linalg.norm(left_eigenvectors, axis=0), np.ones(2)
    )


def test_biorthogonal_modes_for_diagonal_real_system():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    modes = system.biorthogonal_modes()

    assert modes.right_eigenvectors.shape == (3, 3)
    assert modes.left_eigenvectors.shape == (3, 3)
    np.testing.assert_array_equal(modes.eigenvalues, system.eigenvalues())
    np.testing.assert_allclose(
        modes.left_eigenvectors.conj().T @ modes.right_eigenvectors, np.eye(3)
    )


def test_biorthogonal_modes_for_nonsymmetric_diagonalizable_system():
    system = StateSpace(
        [[2.0, 1.0], [0.0, 3.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    modes = system.biorthogonal_modes()

    assert not np.allclose(
        np.abs(modes.left_eigenvectors), np.abs(modes.right_eigenvectors)
    )
    np.testing.assert_allclose(
        modes.left_eigenvectors.conj().T @ modes.right_eigenvectors, np.eye(2)
    )
    for index, eigenvalue in enumerate(modes.eigenvalues):
        right_vector = modes.right_eigenvectors[:, index]
        left_vector = modes.left_eigenvectors[:, index]
        np.testing.assert_allclose(
            system.A @ right_vector, eigenvalue * right_vector
        )
        np.testing.assert_allclose(
            left_vector.conj().T @ system.A,
            eigenvalue * left_vector.conj().T,
        )


def test_biorthogonal_modes_preserve_complex_values_and_modal_ordering():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    modes = system.biorthogonal_modes()
    modal_eigenvalues = np.array(
        [mode.eigenvalue for mode in system.modal_properties()]
    )

    assert np.iscomplexobj(modes.right_eigenvectors)
    assert np.iscomplexobj(modes.left_eigenvectors)
    np.testing.assert_array_equal(modes.eigenvalues, modal_eigenvalues)
    for index, eigenvalue in enumerate(modes.eigenvalues):
        right_vector = modes.right_eigenvectors[:, index]
        left_vector = modes.left_eigenvectors[:, index]
        np.testing.assert_allclose(left_vector.conj().T @ right_vector, 1.0)
        np.testing.assert_allclose(
            system.A @ right_vector, eigenvalue * right_vector
        )
        np.testing.assert_allclose(
            left_vector.conj().T @ system.A,
            eigenvalue * left_vector.conj().T,
        )


def test_biorthogonal_modes_reject_numerically_zero_paired_product(monkeypatch):
    system = StateSpace(np.diag([-1.0, -2.0]), [[0.0], [0.0]], np.eye(2), [[0.0], [0.0]])
    monkeypatch.setattr(system, "right_eigenvectors", lambda: np.eye(2))
    monkeypatch.setattr(
        system, "left_eigenvectors", lambda: np.array([[0.0, 1.0], [1.0, 0.0]])
    )

    with pytest.raises(ValueError, match="inner product is too close to zero"):
        system.biorthogonal_modes()


def test_participation_factors_for_diagonal_real_system():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    participation = system.participation_factors()

    assert participation.shape == (3, 3)
    np.testing.assert_allclose(participation, np.eye(3))
    np.testing.assert_allclose(np.sum(participation, axis=0), np.ones(3))


def test_participation_factors_match_biorthogonal_modes_for_nonsymmetric_system():
    system = StateSpace(
        [[1.0, 1.0], [-2.0, 4.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    modes = system.biorthogonal_modes()
    participation = system.participation_factors()
    expected = modes.right_eigenvectors * np.conj(modes.left_eigenvectors)

    np.testing.assert_array_equal(modes.eigenvalues, system.eigenvalues())
    np.testing.assert_allclose(
        [mode.eigenvalue for mode in system.modal_properties()], modes.eigenvalues
    )
    np.testing.assert_allclose(participation, expected)
    assert np.all(np.abs(participation) > 0.0)
    np.testing.assert_allclose(np.sum(participation, axis=0), np.ones(2))


def test_participation_factors_preserve_complex_conjugate_modes():
    system = StateSpace(
        [[-1.0, -3.0, 1.0], [2.0, -1.0, 2.0], [1.0, 0.0, -4.0]],
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    modes = system.biorthogonal_modes()
    participation = system.participation_factors()

    assert participation.shape == (3, 3)
    assert np.iscomplexobj(participation)
    assert np.any(np.abs(participation.imag) > 0.0)
    np.testing.assert_allclose(
        participation,
        modes.right_eigenvectors * np.conj(modes.left_eigenvectors),
    )
    np.testing.assert_allclose(np.sum(participation, axis=0), np.ones(3))


def test_modal_state_characterization_for_diagonal_system():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    characterizations = system.modal_state_characterization()
    properties = system.modal_properties()

    assert len(characterizations) == 3
    np.testing.assert_array_equal(
        [result.eigenvalue for result in characterizations], system.eigenvalues()
    )
    for index, result in enumerate(characterizations):
        assert result.modal_properties == properties[index]
        assert result.participation_magnitudes.shape == (3,)
        assert np.all(np.isfinite(result.participation_magnitudes))
        assert np.all(result.participation_magnitudes >= 0.0)
        assert np.sum(result.participation_magnitudes) == pytest.approx(1.0)
        np.testing.assert_allclose(
            result.participation_magnitudes, np.eye(3)[:, index]
        )
        assert result.dominant_state_indices == (index,)


def test_modal_state_characterization_for_coupled_system():
    system = StateSpace(
        [[1.0, 1.0], [-2.0, 4.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    characterizations = system.modal_state_characterization()
    participation = np.abs(system.participation_factors())
    expected = participation / np.sum(participation, axis=0)

    assert len(characterizations) == 2
    for index, result in enumerate(characterizations):
        np.testing.assert_allclose(result.participation_magnitudes, expected[:, index])
        assert np.all(result.participation_magnitudes > 0.0)
        assert result.dominant_state_indices


def test_modal_state_characterization_reports_tied_dominant_states():
    system = StateSpace(
        [[-2.0, 1.0], [1.0, -2.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    characterizations = system.modal_state_characterization()

    for result in characterizations:
        np.testing.assert_allclose(result.participation_magnitudes, [0.5, 0.5])
        assert result.dominant_state_indices == (0, 1)


def test_modal_state_characterization_is_invariant_to_modal_vector_scaling(
    monkeypatch,
):
    system = StateSpace(
        [[1.0, 1.0], [-2.0, 4.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )
    baseline = system.modal_state_characterization()
    modes = system.biorthogonal_modes()
    scaling = np.array([2.0 + 1.0j, -0.5 + 0.75j])
    scaled_modes = modes._replace(
        right_eigenvectors=modes.right_eigenvectors * scaling[np.newaxis, :],
        left_eigenvectors=modes.left_eigenvectors
        / np.conj(scaling)[np.newaxis, :],
    )
    monkeypatch.setattr(system, "biorthogonal_modes", lambda: scaled_modes)

    scaled = system.modal_state_characterization()

    for baseline_result, scaled_result in zip(baseline, scaled, strict=True):
        np.testing.assert_allclose(
            scaled_result.participation_magnitudes,
            baseline_result.participation_magnitudes,
        )
        assert scaled_result.dominant_state_indices == (
            baseline_result.dominant_state_indices
        )


def test_modal_state_characterization_is_consistent_for_conjugate_pair():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    first, second = system.modal_state_characterization()

    np.testing.assert_allclose(first.eigenvalue, np.conj(second.eigenvalue))
    np.testing.assert_allclose(
        first.participation_magnitudes, second.participation_magnitudes
    )
    assert first.dominant_state_indices == second.dominant_state_indices
    assert first.modal_properties.natural_frequency == pytest.approx(
        second.modal_properties.natural_frequency
    )
    assert first.modal_properties.damping_ratio == pytest.approx(
        second.modal_properties.damping_ratio
    )


def test_modal_families_groups_one_conjugate_pair():
    system = StateSpace(
        [[-0.5, -2.0], [2.0, -0.5]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    (family,) = system.modal_families()

    assert family.is_oscillatory is True
    assert family.multiplicity == 2
    np.testing.assert_allclose(family.eigenvalues[0], np.conj(family.eigenvalues[1]))


def test_modal_families_groups_multiple_conjugate_pairs():
    system = StateSpace(
        [
            [-1.0, -2.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -3.0, -4.0],
            [0.0, 0.0, 4.0, -3.0],
        ],
        np.zeros((4, 1)),
        np.eye(4),
        np.zeros((4, 1)),
    )

    families = system.modal_families()

    assert len(families) == 2
    assert all(family.is_oscillatory for family in families)
    assert [family.multiplicity for family in families] == [2, 2]
    assert sorted(
        abs(family.eigenvalues[0].imag) for family in families
    ) == pytest.approx([2.0, 4.0])


def test_modal_families_keeps_each_real_mode_isolated():
    system = StateSpace(
        np.diag([-3.0, -1.0, 0.5]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    families = system.modal_families()

    assert len(families) == 3
    assert all(not family.is_oscillatory for family in families)
    assert [family.multiplicity for family in families] == [1, 1, 1]
    np.testing.assert_array_equal(
        [family.eigenvalues[0] for family in families], system.eigenvalues()
    )


def test_modal_families_mixed_spectrum_preserves_canonical_order_and_members():
    system = StateSpace(
        [[-2.0, 0.0, 0.0], [0.0, -0.5, -2.0], [0.0, 2.0, -0.5]],
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )
    families = system.modal_families()

    flattened_eigenvalues = [
        member.eigenvalue for family in families for member in family.members
    ]
    assert sorted(flattened_eigenvalues, key=str) == sorted(
        system.eigenvalues(), key=str
    )
    assert sorted(family.multiplicity for family in families) == [1, 2]
    assert sorted(family.is_oscillatory for family in families) == [False, True]


def test_modal_families_accepts_numerically_perturbed_conjugates(monkeypatch):
    system = StateSpace([[0.0]], [[0.0]], [[1.0]], [[0.0]])

    def characterization(eigenvalue):
        properties = ModalProperties(eigenvalue, 2.0, 0.25, 1.9, 3.0, 2.0)
        return ModalStateCharacterization(eigenvalue, properties, np.ones(1), (0,))

    modes = (
        characterization(-0.5 + 2.0j),
        characterization(-0.50000004 - 2.0000001j),
    )
    monkeypatch.setattr(system, "modal_state_characterization", lambda: modes)

    (family,) = system.modal_families()

    assert family.members == modes
    assert family.is_oscillatory is True


def test_modal_families_deterministically_follow_first_member_order(monkeypatch):
    system = StateSpace([[0.0]], [[0.0]], [[1.0]], [[0.0]])

    def characterization(eigenvalue):
        properties = ModalProperties(eigenvalue, None, None, None, None, None)
        return ModalStateCharacterization(eigenvalue, properties, np.ones(1), (0,))

    modes = tuple(
        characterization(value)
        for value in (-4.0, -1.0 + 3.0j, -2.0, -1.0 - 3.0j)
    )
    monkeypatch.setattr(system, "modal_state_characterization", lambda: modes)

    families = system.modal_families()

    assert [family.eigenvalues for family in families] == [
        (-4.0,),
        (-1.0 + 3.0j, -1.0 - 3.0j),
        (-2.0,),
    ]


def test_modal_families_rejects_unmatched_complex_mode(monkeypatch):
    system = StateSpace([[0.0]], [[0.0]], [[1.0]], [[0.0]])
    eigenvalue = -1.0 + 2.0j
    properties = ModalProperties(eigenvalue, 2.2, 0.4, 2.0, 3.1, 1.0)
    mode = ModalStateCharacterization(eigenvalue, properties, np.ones(1), (0,))
    monkeypatch.setattr(
        system, "modal_state_characterization", lambda: (mode,)
    )

    with pytest.raises(RuntimeError, match="could not match conjugate mode"):
        system.modal_families()


def test_modal_family_state_participation_real_family_matches_member():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    summaries = system.modal_family_state_participation()

    for summary in summaries:
        assert summary.family.multiplicity == 1
        np.testing.assert_array_equal(
            summary.participation_magnitudes,
            summary.family.members[0].participation_magnitudes,
        )


def test_modal_family_state_participation_averages_conjugate_members():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    (summary,) = system.modal_family_state_participation()

    expected = np.mean(
        [
            member.participation_magnitudes
            for member in summary.family.members
        ],
        axis=0,
    )
    assert summary.family.is_oscillatory is True
    np.testing.assert_allclose(summary.participation_magnitudes, expected)
    assert np.sum(summary.participation_magnitudes) == pytest.approx(1.0)


def test_modal_family_state_participation_preserves_family_and_state_order():
    system = StateSpace(
        np.diag([-3.0, -1.0, -2.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_state_participation()

    np.testing.assert_array_equal(
        [summary.family.eigenvalues[0] for summary in summaries],
        system.eigenvalues(),
    )
    np.testing.assert_array_equal(
        [summary.participation_magnitudes for summary in summaries], np.eye(3)
    )
    assert [summary.dominant_state_indices for summary in summaries] == [
        (0,),
        (1,),
        (2,),
    ]


def test_modal_family_state_participation_handles_dominant_tie():
    system = StateSpace(
        [[-2.0, 1.0], [1.0, -2.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    summaries = system.modal_family_state_participation()

    for summary in summaries:
        np.testing.assert_allclose(summary.participation_magnitudes, [0.5, 0.5])
        assert summary.dominant_state_indices == (0, 1)


def test_modal_family_state_participation_matches_participation_factors():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )
    participation = np.abs(system.participation_factors())
    member_participation = participation / np.sum(participation, axis=0)

    (summary,) = system.modal_family_state_participation()

    np.testing.assert_allclose(
        summary.participation_magnitudes,
        np.mean(member_participation, axis=1),
    )


def test_modal_family_state_participation_is_finite_real_and_nonnegative():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_state_participation()

    assert len(summaries) == len(system.modal_families())
    for summary in summaries:
        assert summary.participation_magnitudes.shape == (system.n_states,)
        assert np.isrealobj(summary.participation_magnitudes)
        assert np.all(np.isfinite(summary.participation_magnitudes))
        assert np.all(summary.participation_magnitudes >= 0.0)
        assert np.sum(summary.participation_magnitudes) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("eigenvalue", "stability", "time_constant"),
    [(-2.0, "decaying", 0.5), (4.0, "growing", -0.25)],
)
def test_modal_family_dynamic_summary_for_real_family(
    eigenvalue, stability, time_constant
):
    system = StateSpace([[eigenvalue]], [[0.0]], [[1.0]], [[0.0]])

    (summary,) = system.modal_family_dynamic_summaries()

    assert summary.family.members[0].modal_properties == system.modal_properties()[0]
    assert summary.is_oscillatory is False
    assert summary.real_part == pytest.approx(eigenvalue)
    assert summary.stability == stability
    assert summary.time_constant == pytest.approx(time_constant)
    assert summary.natural_frequency is None
    assert summary.damping_ratio is None
    assert summary.damped_natural_frequency is None
    assert summary.period is None


@pytest.mark.parametrize(
    ("real_part", "stability", "damping_sign"),
    [(-1.0, "decaying", 1.0), (1.0, "growing", -1.0)],
)
def test_modal_family_dynamic_summary_for_oscillatory_family(
    real_part, stability, damping_sign
):
    system = StateSpace(
        [[real_part, -2.0], [2.0, real_part]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )

    (summary,) = system.modal_family_dynamic_summaries()

    expected_frequency = np.sqrt(real_part**2 + 4.0)
    assert summary.is_oscillatory is True
    assert summary.real_part == pytest.approx(real_part)
    assert summary.stability == stability
    assert summary.natural_frequency == pytest.approx(expected_frequency)
    assert summary.damping_ratio == pytest.approx(
        damping_sign / expected_frequency
    )
    assert summary.damped_natural_frequency == pytest.approx(2.0)
    assert summary.period == pytest.approx(np.pi)
    assert summary.time_constant == pytest.approx(-1.0 / real_part)


@pytest.mark.parametrize("real_part", [-1e-10, 0.0, 1e-10])
def test_modal_family_dynamic_summary_neutral_stability_tolerance(real_part):
    system = StateSpace([[real_part]], [[0.0]], [[1.0]], [[0.0]])

    (summary,) = system.modal_family_dynamic_summaries()

    assert summary.stability == "neutral"
    assert summary.real_part == pytest.approx(real_part)


def test_modal_family_dynamic_summary_preserves_family_order():
    system = StateSpace(
        np.diag([-3.0, 2.0, -1.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_dynamic_summaries()

    np.testing.assert_array_equal(
        [summary.family.eigenvalues[0] for summary in summaries],
        system.eigenvalues(),
    )
    assert [summary.stability for summary in summaries] == [
        "decaying",
        "growing",
        "decaying",
    ]


def test_modal_family_dynamic_summary_validates_conjugate_invariants(monkeypatch):
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )
    family = system.modal_families()[0]
    inconsistent_properties = family.members[1].modal_properties._replace(
        natural_frequency=family.members[0].modal_properties.natural_frequency + 0.1
    )
    inconsistent_member = family.members[1]._replace(
        modal_properties=inconsistent_properties
    )
    inconsistent_family = ModalFamily(
        (family.members[0], inconsistent_member), is_oscillatory=True
    )
    monkeypatch.setattr(system, "modal_families", lambda: (inconsistent_family,))

    with pytest.raises(ValueError, match="inconsistent natural_frequency"):
        system.modal_family_dynamic_summaries()


def test_modal_family_dynamic_summary_defined_values_are_finite():
    system = StateSpace(
        [[-1.0, 0.0, 0.0], [0.0, -2.0, -3.0], [0.0, 3.0, -2.0]],
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_dynamic_summaries()

    for summary in summaries:
        values = (
            summary.real_part,
            summary.natural_frequency,
            summary.damping_ratio,
            summary.damped_natural_frequency,
            summary.period,
            summary.time_constant,
        )
        assert all(value is None or np.isfinite(value) for value in values)


def test_modal_input_influence_for_diagonal_system_preserves_input_columns():
    inputs = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        inputs,
        np.eye(3),
        np.zeros((3, 2)),
    )

    influence = system.modal_input_influence()

    assert influence.shape == (3, 2)
    assert np.iscomplexobj(influence)
    np.testing.assert_allclose(influence, inputs)


def test_modal_input_influence_matches_biorthogonal_left_vectors():
    system = StateSpace(
        [[1.0, 1.0], [-2.0, 4.0]],
        [[1.0, -2.0], [3.0, 4.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )

    modes = system.biorthogonal_modes()
    influence = system.modal_input_influence()

    np.testing.assert_array_equal(modes.eigenvalues, system.eigenvalues())
    np.testing.assert_allclose(
        [mode.eigenvalue for mode in system.modal_properties()], modes.eigenvalues
    )
    np.testing.assert_allclose(
        system.participation_factors(),
        modes.right_eigenvectors * np.conj(modes.left_eigenvectors),
    )
    np.testing.assert_allclose(influence, modes.left_eigenvectors.conj().T @ system.B)
    assert np.all(np.isfinite(influence.real))
    assert np.all(np.isfinite(influence.imag))


def test_modal_input_influence_preserves_complex_conjugate_modes():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )

    modes = system.biorthogonal_modes()
    influence = system.modal_input_influence()

    assert influence.shape == (2, 2)
    assert np.iscomplexobj(influence)
    assert np.any(np.abs(influence.imag) > 0.0)
    np.testing.assert_allclose(influence, modes.left_eigenvectors.conj().T @ system.B)
    np.testing.assert_allclose(influence[0], np.conj(influence[1]))


def test_modal_family_input_influence_real_family_matches_member_magnitude():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        [[1.0, -2.0], [3.0, 4.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )
    modal_influence = system.modal_input_influence()

    summaries = system.modal_family_input_influence()

    for row, summary in enumerate(summaries):
        np.testing.assert_allclose(
            summary.influence_magnitudes, np.abs(modal_influence[row])
        )


def test_modal_family_input_influence_averages_conjugate_member_magnitudes():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [3.0, 4.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )
    modal_influence = system.modal_input_influence()

    (summary,) = system.modal_family_input_influence()

    np.testing.assert_allclose(
        summary.influence_magnitudes,
        np.mean(np.abs(modal_influence), axis=0),
    )
    assert summary.family.is_oscillatory is True


def test_modal_family_input_influence_preserves_family_and_input_order():
    inputs = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        inputs,
        np.eye(2),
        np.zeros((2, 3)),
    )

    summaries = system.modal_family_input_influence()

    np.testing.assert_array_equal(
        [summary.family.eigenvalues[0] for summary in summaries],
        system.eigenvalues(),
    )
    np.testing.assert_allclose(
        [summary.influence_magnitudes for summary in summaries], np.abs(inputs)
    )
    assert [summary.dominant_input_indices for summary in summaries] == [
        (2,),
        (2,),
    ]


def test_modal_family_input_influence_is_invariant_to_member_order(monkeypatch):
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        [[1.0, 3.0], [-2.0, 4.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )
    family = system.modal_families()[0]
    baseline = system.modal_family_input_influence()[0]
    reversed_family = ModalFamily(
        tuple(reversed(family.members)), is_oscillatory=True
    )
    monkeypatch.setattr(system, "modal_families", lambda: (reversed_family,))

    reordered = system.modal_family_input_influence()[0]

    np.testing.assert_allclose(
        reordered.influence_magnitudes, baseline.influence_magnitudes
    )
    assert reordered.dominant_input_indices == baseline.dominant_input_indices


def test_modal_family_input_influence_returns_all_dominant_ties():
    system = StateSpace(
        [[-1.0]],
        [[2.0, -2.0, 1.0]],
        [[1.0]],
        np.zeros((1, 3)),
    )

    (summary,) = system.modal_family_input_influence()

    np.testing.assert_allclose(summary.influence_magnitudes, [2.0, 2.0, 1.0])
    assert summary.dominant_input_indices == (0, 1)


def test_modal_family_input_influence_all_zero_has_no_dominant_input():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.zeros((2, 3)),
        np.eye(2),
        np.zeros((2, 3)),
    )

    summaries = system.modal_family_input_influence()

    for summary in summaries:
        np.testing.assert_array_equal(summary.influence_magnitudes, np.zeros(3))
        assert summary.dominant_input_indices == ()


def test_modal_family_input_influence_is_finite_real_and_nonnegative():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        np.eye(3),
        np.zeros((3, 2)),
    )

    summaries = system.modal_family_input_influence()

    assert len(summaries) == len(system.modal_families())
    for summary in summaries:
        assert summary.influence_magnitudes.shape == (system.n_inputs,)
        assert np.isrealobj(summary.influence_magnitudes)
        assert np.all(np.isfinite(summary.influence_magnitudes))
        assert np.all(summary.influence_magnitudes >= 0.0)


def test_modal_output_influence_for_diagonal_system_preserves_output_rows():
    outputs = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        outputs,
        [[100.0], [200.0]],
    )

    influence = system.modal_output_influence()

    assert influence.shape == (2, 3)
    assert np.iscomplexobj(influence)
    np.testing.assert_allclose(influence, outputs)


def test_modal_output_influence_matches_biorthogonal_right_vectors():
    system = StateSpace(
        [[1.0, 1.0], [-2.0, 4.0]],
        np.zeros((2, 1)),
        [[1.0, -2.0], [3.0, 4.0], [5.0, 6.0]],
        np.zeros((3, 1)),
    )

    modes = system.biorthogonal_modes()
    influence = system.modal_output_influence()

    np.testing.assert_array_equal(modes.eigenvalues, system.eigenvalues())
    np.testing.assert_allclose(
        [mode.eigenvalue for mode in system.modal_properties()], modes.eigenvalues
    )
    np.testing.assert_allclose(
        system.participation_factors(),
        modes.right_eigenvectors * np.conj(modes.left_eigenvectors),
    )
    np.testing.assert_allclose(
        system.modal_input_influence(), modes.left_eigenvectors.conj().T @ system.B
    )
    np.testing.assert_allclose(influence, system.C @ modes.right_eigenvectors)
    assert np.all(np.isfinite(influence.real))
    assert np.all(np.isfinite(influence.imag))


def test_modal_output_influence_preserves_complex_conjugate_modes():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        [[1.0, 2.0], [3.0, 4.0]],
        np.zeros((2, 1)),
    )

    modes = system.biorthogonal_modes()
    influence = system.modal_output_influence()

    assert influence.shape == (2, 2)
    assert np.iscomplexobj(influence)
    assert np.any(np.abs(influence.imag) > 0.0)
    np.testing.assert_allclose(influence, system.C @ modes.right_eigenvectors)
    np.testing.assert_allclose(influence[:, 0], np.conj(influence[:, 1]))


def test_modal_family_output_influence_real_family_matches_member_magnitude():
    outputs = np.array([[1.0, -2.0], [3.0, 4.0], [5.0, -6.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.zeros((2, 1)),
        outputs,
        np.zeros((3, 1)),
    )
    modal_influence = system.modal_output_influence()

    summaries = system.modal_family_output_influence()

    for column, summary in enumerate(summaries):
        np.testing.assert_allclose(
            summary.influence_magnitudes, np.abs(modal_influence[:, column])
        )


def test_modal_family_output_influence_averages_conjugate_member_magnitudes():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        [[1.0, -2.0], [3.0, 4.0], [-0.5, 2.0]],
        np.zeros((3, 1)),
    )
    modal_influence = system.modal_output_influence()

    (summary,) = system.modal_family_output_influence()

    np.testing.assert_allclose(
        summary.influence_magnitudes,
        np.mean(np.abs(modal_influence), axis=1),
    )
    assert summary.family.is_oscillatory is True


def test_modal_family_output_influence_preserves_family_and_output_order():
    outputs = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.zeros((2, 1)),
        outputs,
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_output_influence()

    np.testing.assert_array_equal(
        [summary.family.eigenvalues[0] for summary in summaries],
        system.eigenvalues(),
    )
    np.testing.assert_allclose(
        [summary.influence_magnitudes for summary in summaries],
        np.abs(outputs.T),
    )
    assert [summary.dominant_output_indices for summary in summaries] == [
        (2,),
        (2,),
    ]


def test_modal_family_output_influence_is_invariant_to_member_order(monkeypatch):
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        [[1.0, 3.0], [-2.0, 4.0]],
        np.zeros((2, 1)),
    )
    family = system.modal_families()[0]
    baseline = system.modal_family_output_influence()[0]
    reversed_family = ModalFamily(
        tuple(reversed(family.members)), is_oscillatory=True
    )
    monkeypatch.setattr(system, "modal_families", lambda: (reversed_family,))

    reordered = system.modal_family_output_influence()[0]

    np.testing.assert_allclose(
        reordered.influence_magnitudes, baseline.influence_magnitudes
    )
    assert reordered.dominant_output_indices == baseline.dominant_output_indices


def test_modal_family_output_influence_returns_all_dominant_ties():
    system = StateSpace(
        [[-1.0]],
        [[0.0]],
        [[2.0], [-2.0], [1.0]],
        np.zeros((3, 1)),
    )

    (summary,) = system.modal_family_output_influence()

    np.testing.assert_allclose(summary.influence_magnitudes, [2.0, 2.0, 1.0])
    assert summary.dominant_output_indices == (0, 1)


def test_modal_family_output_influence_all_zero_has_no_dominant_output():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.zeros((2, 1)),
        np.zeros((3, 2)),
        np.zeros((3, 1)),
    )

    summaries = system.modal_family_output_influence()

    for summary in summaries:
        np.testing.assert_array_equal(summary.influence_magnitudes, np.zeros(3))
        assert summary.dominant_output_indices == ()


def test_modal_family_output_influence_is_finite_real_and_nonnegative():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        np.zeros((3, 1)),
        [[1.0, -2.0, 0.5], [0.5, 3.0, -1.0]],
        np.zeros((2, 1)),
    )

    summaries = system.modal_family_output_influence()

    assert len(summaries) == len(system.modal_families())
    for summary in summaries:
        assert summary.influence_magnitudes.shape == (system.n_outputs,)
        assert np.isrealobj(summary.influence_magnitudes)
        assert np.all(np.isfinite(summary.influence_magnitudes))
        assert np.all(summary.influence_magnitudes >= 0.0)


def test_modal_family_characterization_for_single_real_family():
    system = StateSpace(
        [[-2.0]], [[3.0, -1.0]], [[2.0], [-4.0]], np.zeros((2, 2))
    )

    (characterization,) = system.modal_family_characterizations()

    assert characterization.family.multiplicity == 1
    assert characterization.dynamics.is_oscillatory is False
    assert characterization.dynamics.stability == "decaying"
    assert characterization.state_participation.dominant_state_indices == (0,)
    assert characterization.input_influence.dominant_input_indices == (0,)
    assert characterization.output_influence.dominant_output_indices == (1,)


def test_modal_family_characterization_for_single_oscillatory_family():
    system = StateSpace(
        [[-1.0, -2.0], [2.0, -1.0]],
        [[1.0], [3.0]],
        [[1.0, -2.0]],
        [[0.0]],
    )

    (characterization,) = system.modal_family_characterizations()

    assert characterization.family.multiplicity == 2
    assert characterization.dynamics.is_oscillatory is True
    assert characterization.dynamics.natural_frequency == pytest.approx(np.sqrt(5))
    assert characterization.dynamics.stability == "decaying"


def test_modal_family_characterizations_match_existing_component_results():
    system = StateSpace(
        [[-3.0, 0.0, 0.0], [0.0, -1.0, -2.0], [0.0, 2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        np.zeros((2, 2)),
    )
    expected_dynamics = system.modal_family_dynamic_summaries()
    expected_states = system.modal_family_state_participation()
    expected_inputs = system.modal_family_input_influence()
    expected_outputs = system.modal_family_output_influence()

    characterizations = system.modal_family_characterizations()

    assert len(characterizations) == 2
    for index, characterization in enumerate(characterizations):
        np.testing.assert_allclose(
            characterization.family.eigenvalues,
            expected_dynamics[index].family.eigenvalues,
        )
        assert characterization.dynamics[1:] == expected_dynamics[index][1:]
        np.testing.assert_allclose(
            characterization.state_participation.participation_magnitudes,
            expected_states[index].participation_magnitudes,
        )
        assert (
            characterization.state_participation.dominant_state_indices
            == expected_states[index].dominant_state_indices
        )
        np.testing.assert_allclose(
            characterization.input_influence.influence_magnitudes,
            expected_inputs[index].influence_magnitudes,
        )
        assert (
            characterization.input_influence.dominant_input_indices
            == expected_inputs[index].dominant_input_indices
        )
        np.testing.assert_allclose(
            characterization.output_influence.influence_magnitudes,
            expected_outputs[index].influence_magnitudes,
        )
        assert (
            characterization.output_influence.dominant_output_indices
            == expected_outputs[index].dominant_output_indices
        )


def test_modal_family_characterizations_share_one_canonical_family_object():
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )

    characterizations = system.modal_family_characterizations()

    for characterization in characterizations:
        assert characterization.dynamics.family is characterization.family
        assert characterization.state_participation.family is characterization.family
        assert characterization.input_influence.family is characterization.family
        assert characterization.output_influence.family is characterization.family


def test_modal_family_characterizations_preserve_canonical_order():
    system = StateSpace(
        np.diag([-3.0, 2.0, -1.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    characterizations = system.modal_family_characterizations()

    np.testing.assert_array_equal(
        [result.family.eigenvalues[0] for result in characterizations],
        system.eigenvalues(),
    )


def test_modal_family_characterizations_reject_mismatched_family(monkeypatch):
    system = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    )
    output_summaries = system.modal_family_output_influence()
    mismatched = (
        output_summaries[0]._replace(family=output_summaries[1].family),
        output_summaries[1],
    )
    monkeypatch.setattr(
        system, "modal_family_output_influence", lambda: mismatched
    )

    with pytest.raises(RuntimeError, match="components are inconsistent"):
        system.modal_family_characterizations()


def test_modal_coordinates_and_reconstructed_state_have_expected_shapes():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )

    coordinates = system.modal_coordinates([1.0, -2.0, 3.0])
    reconstructed = system.reconstruct_state(coordinates)

    assert coordinates.shape == (3,)
    assert reconstructed.shape == (3,)
    assert np.all(np.isfinite(coordinates))
    assert np.all(np.isfinite(reconstructed))


def test_modal_reconstruction_recovers_state_for_coupled_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        np.zeros((3, 2)),
        np.eye(3),
        np.zeros((3, 2)),
    )
    state = np.array([1.25, -0.75, 2.5])

    coordinates = system.modal_coordinates(state)
    reconstructed = system.reconstruct_state(coordinates)

    modes = system.biorthogonal_modes()
    np.testing.assert_allclose(
        modes.left_eigenvectors.conj().T @ modes.right_eigenvectors,
        np.eye(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(reconstructed, state, atol=1e-12)


def test_modal_coordinates_preserve_complex_values_for_oscillatory_modes():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    )
    state = np.array([1.0, 2.0])

    coordinates = system.modal_coordinates(state)

    assert np.iscomplexobj(coordinates)
    assert np.any(np.abs(coordinates.imag) > 0.0)
    np.testing.assert_allclose(coordinates[0], np.conj(coordinates[1]))
    np.testing.assert_allclose(system.reconstruct_state(coordinates), state)


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        ("modal_coordinates", [[1.0, 2.0]], "x must have shape"),
        ("reconstruct_state", [[1.0, 2.0]], "z must have shape"),
    ],
)
def test_modal_coordinate_transforms_reject_incompatible_shapes(
    method, value, message
):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match=message):
        getattr(system, method)(value)


def test_modal_representation_has_expected_shapes_and_components():
    system = StateSpace(
        np.diag([-1.0, -2.0, -3.0]),
        np.arange(6.0).reshape(3, 2),
        np.arange(12.0).reshape(4, 3),
        np.arange(8.0).reshape(4, 2),
    )

    representation = system.modal_representation()

    assert representation.Lambda.shape == (3, 3)
    assert representation.G_modal.shape == (3, 2)
    assert representation.H_modal.shape == (4, 3)
    assert representation.D.shape == (4, 2)
    for matrix in representation:
        assert np.all(np.isfinite(matrix))
    np.testing.assert_array_equal(
        representation.Lambda,
        np.diag(np.diag(representation.Lambda)),
    )
    np.testing.assert_array_equal(
        np.diag(representation.Lambda), system.eigenvalues()
    )
    np.testing.assert_allclose(
        representation.G_modal, system.modal_input_influence()
    )
    np.testing.assert_allclose(
        representation.H_modal, system.modal_output_influence()
    )
    np.testing.assert_array_equal(representation.D, system.D)


def test_modal_representation_reconstructs_coupled_physical_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )

    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    V = modes.right_eigenvectors
    W_H = modes.left_eigenvectors.conj().T

    assert np.iscomplexobj(representation.Lambda)
    assert np.iscomplexobj(representation.G_modal)
    assert np.iscomplexobj(representation.H_modal)
    np.testing.assert_allclose(
        system.A, V @ representation.Lambda @ W_H, atol=1e-12
    )
    np.testing.assert_allclose(system.B, V @ representation.G_modal, atol=1e-12)
    np.testing.assert_allclose(
        system.C, representation.H_modal @ W_H, atol=1e-12
    )
    np.testing.assert_array_equal(representation.D, system.D)


def test_modal_evaluation_matches_representation_with_complex_coordinates():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        [[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]],
    )
    representation = system.modal_representation()
    z = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    u = np.array([0.25, -1.5])

    derivative = representation.state_derivative(z, u)
    output = representation.output(z, u)

    assert derivative.shape == (2,)
    assert output.shape == (3,)
    assert np.iscomplexobj(derivative)
    assert np.iscomplexobj(output)
    assert np.all(np.isfinite(derivative))
    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(
        derivative, representation.Lambda @ z + representation.G_modal @ u
    )
    np.testing.assert_allclose(
        output, representation.H_modal @ z + representation.D @ u
    )


def test_modal_evaluation_matches_coupled_physical_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    z = np.array([1.0 + 0.5j, -0.25 + 1.5j, 2.0 - 0.75j])
    u = np.array([0.4, -1.2])
    x = modes.right_eigenvectors @ z

    physical_derivative = system.A @ x + system.B @ u
    modal_derivative = representation.state_derivative(z, u)
    physical_output = system.C @ x + system.D @ u
    modal_output = representation.output(z, u)

    np.testing.assert_allclose(
        physical_derivative,
        modes.right_eigenvectors @ modal_derivative,
        atol=1e-12,
    )
    np.testing.assert_allclose(physical_output, modal_output, atol=1e-12)


def test_modal_evaluation_with_zero_input():
    system = StateSpace(*valid_matrices())
    representation = system.modal_representation()
    z = np.array([1.0 + 0.25j, -2.0j])
    u = np.zeros(system.n_inputs)

    np.testing.assert_allclose(
        representation.state_derivative(z, u), representation.Lambda @ z
    )
    np.testing.assert_allclose(
        representation.output(z, u), representation.H_modal @ z
    )


@pytest.mark.parametrize(
    ("method", "z", "u", "message"),
    [
        ("state_derivative", [[1.0, 2.0]], [0.0], "z must be a 1D vector"),
        ("state_derivative", [1.0, 2.0], [[0.0]], "u must be a 1D vector"),
        ("output", [[1.0, 2.0]], [0.0], "z must be a 1D vector"),
        ("output", [1.0, 2.0], [[0.0]], "u must be a 1D vector"),
    ],
)
def test_modal_evaluation_rejects_incompatible_shapes(method, z, u, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        getattr(representation, method)(z, u)


def test_modal_euler_step_matches_forward_euler_with_complex_values():
    system = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )
    representation = system.modal_representation()
    z = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    u = np.array([0.25, -1.5])
    dt = 0.05

    result = representation.euler_step(z, u, dt)

    assert result.shape == (2,)
    assert np.iscomplexobj(result)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(
        result, z + dt * representation.state_derivative(z, u)
    )


def test_modal_rk4_step_matches_scalar_analytical_zero_input_solution():
    system = StateSpace([[-2.0]], [[0.0]], [[1.0]], [[0.0]])
    representation = system.modal_representation()
    z = np.array([1.0 + 0.5j])

    result = representation.rk4_step(z, [0.0], 0.1)

    assert result.shape == (1,)
    assert np.iscomplexobj(result)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, z * np.exp(-0.2), rtol=2e-5)


@pytest.mark.parametrize("method", ["euler_step", "rk4_step"])
def test_modal_step_matches_coupled_physical_step(method):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        np.eye(3),
        np.zeros((3, 2)),
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    state = np.array([1.25, -0.75, 2.5])
    z = modes.left_eigenvectors.conj().T @ state
    u = np.array([0.4, -1.2])
    dt = 0.025

    modal_next = getattr(representation, method)(z, u, dt)
    physical_next = getattr(system, method)(state, u, dt)

    np.testing.assert_allclose(
        modes.right_eigenvectors @ modal_next,
        physical_next,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("method", "z", "u", "message"),
    [
        ("euler_step", [[1.0, 2.0]], [0.0], "z must be a 1D vector"),
        ("euler_step", [1.0, 2.0], [[0.0]], "u must be a 1D vector"),
        ("rk4_step", [[1.0, 2.0]], [0.0], "z must be a 1D vector"),
        ("rk4_step", [1.0, 2.0], [[0.0]], "u must be a 1D vector"),
    ],
)
def test_modal_steps_reject_incompatible_shapes(method, z, u, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        getattr(representation, method)(z, u, 0.1)


@pytest.mark.parametrize("method", ["euler_step", "rk4_step"])
@pytest.mark.parametrize("dt", [0, -0.1, np.inf, [0.1]])
def test_modal_steps_require_positive_finite_scalar_dt(method, dt):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="dt must be a finite positive scalar"):
        getattr(representation, method)([1.0, 2.0], [0.0], dt)


def test_modal_exact_step_matches_real_scalar_solution():
    representation = StateSpace(
        [[-2.0]], [[0.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    z = np.array([1.5 + 0.25j])

    result = representation.exact_step(z, 0.3)

    assert result.shape == (1,)
    assert np.iscomplexobj(result)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, np.exp(-2.0 * 0.3) * z)


def test_modal_exact_step_matches_complex_eigenvalue_solution():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    z = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    dt = 0.2

    result = representation.exact_step(z, dt)

    expected = np.exp(np.diag(representation.Lambda) * dt) * z
    np.testing.assert_allclose(result, expected)
    assert np.any(np.abs(result.imag) > 0.0)


def test_modal_exact_step_propagates_stable_and_unstable_modes_independently():
    representation = StateSpace(
        np.diag([-2.0, 0.5, -0.25]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    ).modal_representation()
    z = np.array([1.0, 2.0, -3.0])
    dt = 0.4

    result = representation.exact_step(z, dt)

    np.testing.assert_allclose(
        result, np.exp(np.array([-2.0, 0.5, -0.25]) * dt) * z
    )
    assert abs(result[0]) < abs(z[0])
    assert abs(result[1]) > abs(z[1])
    assert abs(result[2]) < abs(z[2])


def test_modal_exact_step_preserves_zero_state():
    representation = StateSpace(*valid_matrices()).modal_representation()

    result = representation.exact_step(np.zeros(2, dtype=complex), 1.0)

    np.testing.assert_array_equal(result, np.zeros(2, dtype=complex))


def test_modal_exact_step_preserves_purely_imaginary_mode_magnitudes():
    representation = StateSpace(
        [[0.0, -2.0], [2.0, 0.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    z = np.array([1.0 + 0.5j, -2.0 + 0.25j])

    result = representation.exact_step(z, 0.7)

    np.testing.assert_allclose(np.abs(result), np.abs(z))


def test_modal_exact_step_is_consistent_for_coupled_physical_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    dt = 0.35

    exact_modal_state = representation.exact_step(initial_modal_state, dt)
    exact_physical_state = modes.right_eigenvectors @ exact_modal_state
    expected = (
        modes.right_eigenvectors
        @ np.diag(np.exp(modes.eigenvalues * dt))
        @ modes.left_eigenvectors.conj().T
        @ initial_state
    )

    np.testing.assert_allclose(exact_physical_state, expected, atol=1e-12)


def test_modal_rk4_step_is_more_accurate_than_euler_against_exact_step():
    representation = StateSpace(
        [[-2.0]], [[0.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    z = np.array([1.0])
    zero_input = np.array([0.0])
    dt = 0.5

    exact = representation.exact_step(z, dt)
    euler = representation.euler_step(z, zero_input, dt)
    rk4 = representation.rk4_step(z, zero_input, dt)

    np.testing.assert_allclose(exact, [np.exp(-1.0)])
    assert np.linalg.norm(rk4 - exact) < np.linalg.norm(euler - exact)


@pytest.mark.parametrize("z", [[1.0], [[1.0, 2.0]]])
def test_modal_exact_step_rejects_invalid_state_shape(z):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="z must be a 1D vector"):
        representation.exact_step(z, 0.1)


@pytest.mark.parametrize("dt", [0, -0.1, np.inf, [0.1]])
def test_modal_exact_step_requires_positive_finite_scalar_dt(dt):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="dt must be a finite positive scalar"):
        representation.exact_step([1.0, 2.0], dt)


def test_exact_forced_step_with_zero_input_matches_exact_step():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    state = np.array([1.0 + 2.0j, -0.5 + 0.75j])

    forced = representation.exact_forced_step(state, [0.0], 0.3)
    unforced = representation.exact_step(state, 0.3)

    np.testing.assert_array_equal(forced, unforced)


@pytest.mark.parametrize("eigenvalue", [-2.0, 0.75])
def test_exact_forced_step_matches_real_scalar_solution(eigenvalue):
    representation = StateSpace(
        [[eigenvalue]], [[1.5]], [[1.0]], [[0.0]]
    ).modal_representation()
    state = np.array([0.8 + 0.2j])
    control = np.array([2.0])
    dt = 0.4

    result = representation.exact_forced_step(state, control, dt)

    forcing = representation.G_modal @ control
    expected = (
        np.exp(eigenvalue * dt) * state
        + np.expm1(eigenvalue * dt) / eigenvalue * forcing
    )
    assert result.shape == (1,)
    assert np.iscomplexobj(result)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected)


def test_exact_forced_step_matches_complex_modal_solution():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0], [2.0]],
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    state = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    control = np.array([1.25])
    dt = 0.2

    result = representation.exact_forced_step(state, control, dt)

    eigenvalues = np.diag(representation.Lambda)
    forcing = representation.G_modal @ control
    expected = (
        np.exp(eigenvalues * dt) * state
        + np.expm1(eigenvalues * dt) / eigenvalues * forcing
    )
    np.testing.assert_allclose(result, expected)
    assert np.any(np.abs(result.imag) > 0.0)


def test_exact_forced_step_propagates_multiple_modes_and_inputs():
    system = StateSpace(
        np.diag([-2.0, 0.5, -0.25]),
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        np.eye(3),
        np.zeros((3, 2)),
    )
    representation = system.modal_representation()
    state = np.array([1.0 + 0.5j, -2.0j, 0.25])
    control = np.array([0.4, -1.2])
    dt = 0.35

    result = representation.exact_forced_step(state, control, dt)

    eigenvalues = np.diag(representation.Lambda)
    forcing = representation.G_modal @ control
    expected = (
        np.exp(eigenvalues * dt) * state
        + np.expm1(eigenvalues * dt) / eigenvalues * forcing
    )
    np.testing.assert_allclose(result, expected)
    assert np.all(np.abs(forcing) > 0.0)


def test_exact_forced_step_uses_zero_eigenvalue_limit():
    representation = StateSpace(
        [[0.0]], [[2.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    state = np.array([1.0 + 0.5j])
    control = np.array([3.0])
    dt = 0.25

    result = representation.exact_forced_step(state, control, dt)

    forcing = representation.G_modal @ control
    np.testing.assert_allclose(result, state + dt * forcing)


def test_exact_forced_step_is_continuous_near_zero_eigenvalue():
    representation = StateSpace(
        [[1e-14]], [[2.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    state = np.array([1.0])
    control = np.array([3.0])
    dt = 0.25

    result = representation.exact_forced_step(state, control, dt)
    zero_limit = state + dt * (representation.G_modal @ control)

    np.testing.assert_allclose(result, zero_limit, rtol=1e-14, atol=1e-14)


def test_exact_forced_step_from_zero_state_matches_forcing_solution():
    representation = StateSpace(
        [[-1.5]], [[2.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    control = np.array([1.25])
    dt = 0.4

    result = representation.exact_forced_step([0.0], control, dt)

    forcing = representation.G_modal @ control
    expected = np.expm1(-1.5 * dt) / -1.5 * forcing
    np.testing.assert_allclose(result, expected)


def test_exact_forced_step_is_consistent_for_coupled_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        np.eye(3),
        np.zeros((3, 2)),
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    control = np.array([0.4, -1.2])
    dt = 0.35

    exact_modal_state = representation.exact_forced_step(
        initial_modal_state, control, dt
    )
    eigenvalues = modes.eigenvalues
    forcing = representation.G_modal @ control
    expected_modal_state = (
        np.exp(eigenvalues * dt) * initial_modal_state
        + np.expm1(eigenvalues * dt) / eigenvalues * forcing
    )
    reconstructed_state = modes.right_eigenvectors @ exact_modal_state
    expected_physical_state = modes.right_eigenvectors @ expected_modal_state

    np.testing.assert_allclose(exact_modal_state, expected_modal_state, atol=1e-12)
    np.testing.assert_allclose(
        reconstructed_state, expected_physical_state, atol=1e-12
    )


def test_exact_forced_step_outperforms_numerical_steps():
    representation = StateSpace(
        [[-2.0]], [[1.5]], [[1.0]], [[0.0]]
    ).modal_representation()
    state = np.array([1.0])
    control = np.array([2.0])
    dt = 0.5

    exact = representation.exact_forced_step(state, control, dt)
    euler = representation.euler_step(state, control, dt)
    rk4 = representation.rk4_step(state, control, dt)
    analytical = np.exp(-2.0 * dt) * state + np.expm1(-2.0 * dt) / -2.0 * 3.0

    np.testing.assert_allclose(exact, analytical)
    assert np.linalg.norm(rk4 - exact) < np.linalg.norm(euler - exact)


@pytest.mark.parametrize(
    ("z", "u", "message"),
    [
        ([[1.0, 2.0]], [0.0], "z must be a 1D vector"),
        ([1.0, 2.0], [[0.0]], "u must be a 1D vector"),
    ],
)
def test_exact_forced_step_rejects_invalid_shapes(z, u, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.exact_forced_step(z, u, 0.1)


@pytest.mark.parametrize("dt", [0, -0.1, np.inf, [0.1]])
def test_exact_forced_step_requires_positive_finite_scalar_dt(dt):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="dt must be a finite positive scalar"):
        representation.exact_forced_step([1.0, 2.0], [0.0], dt)


def test_exact_zero_input_response_matches_scalar_decay_on_nonuniform_grid():
    representation = StateSpace(
        [[-2.0]], [[0.0]], [[3.0]], [[0.0]]
    ).modal_representation()
    initial_state = np.array([1.5 + 0.25j])
    time = np.array([2.0, 2.1, 2.4, 3.0])

    states, outputs = representation.exact_zero_input_response(
        initial_state, time
    )

    expected_states = (
        np.exp(-2.0 * (time - time[0]))[:, np.newaxis] * initial_state
    )
    assert states.shape == (4, 1)
    assert outputs.shape == (4, 1)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], initial_state)
    np.testing.assert_allclose(states, expected_states)
    np.testing.assert_allclose(outputs, 3.0 * expected_states)


def test_exact_zero_input_response_matches_complex_oscillatory_solution():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    initial_state = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    time = np.array([0.0, 0.1, 0.35, 0.8])

    states, _ = representation.exact_zero_input_response(initial_state, time)

    expected = np.exp(
        np.outer(time - time[0], np.diag(representation.Lambda))
    ) * initial_state
    np.testing.assert_allclose(states, expected)
    assert np.any(np.abs(states.imag) > 0.0)


def test_exact_zero_input_response_preserves_zero_state():
    representation = StateSpace(*valid_matrices()).modal_representation()

    states, outputs = representation.exact_zero_input_response(
        np.zeros(2), [0.0, 0.1, 0.4]
    )

    np.testing.assert_array_equal(states, np.zeros((3, 2), dtype=complex))
    np.testing.assert_array_equal(outputs, np.zeros((3, 1), dtype=complex))


def test_exact_zero_input_response_matches_repeated_exact_steps():
    representation = StateSpace(
        np.diag([-1.0, -2.0, 0.5]),
        np.zeros((3, 1)),
        np.eye(3),
        np.zeros((3, 1)),
    ).modal_representation()
    initial_state = np.array([1.0 + 0.5j, -2.0j, 0.25])
    time = np.array([0.0, 0.05, 0.2, 0.7])

    states, _ = representation.exact_zero_input_response(initial_state, time)

    expected = initial_state.copy()
    np.testing.assert_array_equal(states[0], expected)
    for index, dt in enumerate(np.diff(time), start=1):
        expected = representation.exact_step(expected, dt)
        np.testing.assert_array_equal(states[index], expected)


def test_exact_zero_input_response_is_consistent_for_coupled_system():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        np.zeros((3, 1)),
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        np.zeros((2, 1)),
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    time = np.array([1.0, 1.05, 1.2, 1.7])

    modal_states, modal_outputs = representation.exact_zero_input_response(
        initial_modal_state, time
    )

    for index, elapsed_time in enumerate(time - time[0]):
        modal_factors = np.exp(modes.eigenvalues * elapsed_time)
        expected_modal_state = modal_factors * initial_modal_state
        expected_physical_state = (
            modes.right_eigenvectors
            @ np.diag(modal_factors)
            @ modes.left_eigenvectors.conj().T
            @ initial_state
        )
        reconstructed_state = modes.right_eigenvectors @ modal_states[index]
        np.testing.assert_allclose(
            modal_states[index], expected_modal_state, atol=1e-12
        )
        np.testing.assert_allclose(
            reconstructed_state, expected_physical_state, atol=1e-12
        )
        np.testing.assert_allclose(
            modal_outputs[index], system.C @ expected_physical_state, atol=1e-12
        )


def test_exact_zero_input_response_outperforms_numerical_trajectories():
    representation = StateSpace(
        [[-2.0]], [[0.0]], [[1.0]], [[0.0]]
    ).modal_representation()
    initial_state = np.array([1.0])
    time = np.linspace(0.0, 1.0, 5)

    exact_states, _ = representation.exact_zero_input_response(
        initial_state, time
    )
    euler_states, _ = representation.zero_input_response(
        initial_state, time, method="euler"
    )
    rk4_states, _ = representation.zero_input_response(
        initial_state, time, method="rk4"
    )
    analytical = np.exp(-2.0 * time)[:, np.newaxis]

    np.testing.assert_allclose(exact_states, analytical)
    euler_error = np.linalg.norm(euler_states - exact_states)
    rk4_error = np.linalg.norm(rk4_states - exact_states)
    assert rk4_error < euler_error


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([], "time must be a non-empty 1D grid"),
        ([[0.0, 0.1]], "time must be a non-empty 1D grid"),
        ([0.0, np.nan], "time values must be finite"),
        ([0.0, 0.0], "time values must be strictly increasing"),
        ([0.1, 0.0], "time values must be strictly increasing"),
    ],
)
def test_exact_zero_input_response_rejects_invalid_time_grids(time, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.exact_zero_input_response([1.0, 2.0], time)


@pytest.mark.parametrize("z0", [[1.0], [[1.0, 2.0]]])
def test_exact_zero_input_response_rejects_invalid_state_shape(z0):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="z must be a 1D vector"):
        representation.exact_zero_input_response(z0, [0.0])


def test_exact_simulate_matches_scalar_constant_input_solution():
    representation = StateSpace(
        [[-2.0]], [[1.5]], [[2.0]], [[0.25]]
    ).modal_representation()
    initial_state = np.array([1.0 + 0.5j])
    control = np.array([2.0])
    time = np.array([1.0, 1.1, 1.4, 2.0])

    states, outputs = representation.exact_simulate(
        initial_state, control, time
    )

    elapsed = time - time[0]
    forcing = (representation.G_modal @ control)[0]
    expected_states = (
        np.exp(-2.0 * elapsed) * initial_state[0]
        + np.expm1(-2.0 * elapsed) / -2.0 * forcing
    )[:, np.newaxis]
    assert states.shape == (4, 1)
    assert outputs.shape == (4, 1)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], initial_state)
    np.testing.assert_allclose(states, expected_states)
    np.testing.assert_allclose(
        outputs,
        2.0 * expected_states + 0.25 * control,
    )


def test_exact_simulate_zero_input_matches_exact_zero_input_response():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 2)),
        np.eye(2),
        np.zeros((2, 2)),
    ).modal_representation()
    initial_state = np.array([1.0 + 2.0j, -0.5 + 0.75j])
    time = np.array([0.0, 0.1, 0.35, 0.8])

    exact = representation.exact_simulate(initial_state, np.zeros(2), time)
    unforced = representation.exact_zero_input_response(initial_state, time)

    for actual, expected in zip(exact, unforced, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_exact_simulate_coupled_system_uses_left_sampled_multi_input():
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    inputs = np.array(
        [[0.4, -1.2], [0.0, 0.5], [-0.75, 0.25], [2.0, -1.0]]
    )
    time = np.array([0.0, 0.025, 0.1, 0.2])

    modal_states, modal_outputs = representation.exact_simulate(
        initial_modal_state, inputs, time
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    assert reconstructed_states.shape == (4, 3)
    assert np.all(np.isfinite(reconstructed_states))
    for index, dt in enumerate(np.diff(time)):
        expected_next = representation.exact_forced_step(
            modal_states[index], inputs[index], dt
        )
        np.testing.assert_array_equal(modal_states[index + 1], expected_next)
    for index, state in enumerate(modal_states):
        np.testing.assert_allclose(
            modal_outputs[index],
            representation.H_modal @ state + representation.D @ inputs[index],
        )


def test_exact_simulate_outperforms_numerical_forced_trajectories():
    representation = StateSpace(
        [[-2.0]], [[1.5]], [[1.0]], [[0.0]]
    ).modal_representation()
    initial_state = np.array([1.0])
    control = np.array([2.0])
    time = np.linspace(0.0, 1.0, 5)

    exact_states, _ = representation.exact_simulate(
        initial_state, control, time
    )
    euler_states, _ = representation.simulate(
        initial_state, control, time, method="euler"
    )
    rk4_states, _ = representation.simulate(
        initial_state, control, time, method="rk4"
    )

    euler_error = np.linalg.norm(euler_states - exact_states)
    rk4_error = np.linalg.norm(rk4_states - exact_states)
    assert euler_error > 0.0
    assert rk4_error < euler_error


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([], "time must be a non-empty 1D grid"),
        ([[0.0, 0.1]], "time must be a non-empty 1D grid"),
        ([0.0, np.nan], "time values must be finite"),
        ([0.0, 0.0], "time values must be strictly increasing"),
        ([0.1, 0.0], "time values must be strictly increasing"),
    ],
)
def test_exact_simulate_rejects_invalid_time_grids(time, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.exact_simulate([1.0, 2.0], [3.0], time)


@pytest.mark.parametrize(
    ("z0", "u", "message"),
    [
        ([[1.0, 2.0]], [3.0], "z must be a 1D vector"),
        ([1.0, 2.0], [3.0, 4.0], "u must have shape"),
        ([1.0, 2.0], [[3.0], [4.0], [5.0]], "u must have shape"),
    ],
)
def test_exact_simulate_rejects_invalid_state_and_input_shapes(z0, u, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.exact_simulate(z0, u, [0.0, 0.1])


def test_modal_simulate_zero_input_preserves_initial_complex_state():
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        np.zeros((2, 1)),
        np.eye(2),
        np.zeros((2, 1)),
    ).modal_representation()
    z0 = np.array([1.0 + 0.5j, -0.25 - 2.0j])
    time = np.array([0.0, 0.1, 0.25])

    states, outputs = representation.simulate(z0, [0.0], time)

    assert states.shape == (3, 2)
    assert outputs.shape == (3, 2)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], z0)
    expected = z0.copy()
    for index, dt in enumerate(np.diff(time), start=1):
        expected = representation.euler_step(expected, [0.0], dt)
        np.testing.assert_allclose(states[index], expected)


def test_modal_simulate_constant_input_rk4_matches_repeated_steps():
    representation = StateSpace(*valid_matrices()).modal_representation()
    z0 = np.array([0.5 + 0.25j, -1.0j])
    u = np.array([2.0])
    time = np.array([0.0, 0.05, 0.2, 0.3])

    states, outputs = representation.simulate(z0, u, time, method="rk4")

    expected = z0.copy()
    np.testing.assert_array_equal(states[0], expected)
    for index, dt in enumerate(np.diff(time), start=1):
        expected = representation.rk4_step(expected, u, dt)
        np.testing.assert_allclose(states[index], expected)
    for index, state in enumerate(states):
        np.testing.assert_allclose(outputs[index], representation.output(state, u))


def test_modal_simulate_uses_left_sampled_time_varying_input():
    representation = StateSpace(
        [[0.0]], [[1.0]], [[1.0]], [[2.0]]
    ).modal_representation()
    inputs = np.array([[1.0], [2.0], [100.0]])

    states, outputs = representation.simulate(
        [0.0], inputs, [0.0, 0.5, 1.0], method="euler"
    )

    np.testing.assert_allclose(states, [[0.0], [0.5], [1.5]])
    np.testing.assert_allclose(outputs, [[2.0], [4.5], [201.5]])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_simulation_matches_coupled_physical_trajectory(method):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    inputs = np.array(
        [[0.4, -1.2], [0.0, 0.5], [-0.75, 0.25], [2.0, -1.0]]
    )
    time = np.array([0.0, 0.025, 0.1, 0.2])

    physical_states, physical_outputs = system.simulate(
        initial_state, inputs, time, method=method
    )
    modal_states, modal_outputs = representation.simulate(
        initial_modal_state, inputs, time, method=method
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    np.testing.assert_allclose(
        reconstructed_states, physical_states, atol=1e-11
    )
    np.testing.assert_allclose(modal_outputs, physical_outputs, atol=1e-11)


@pytest.mark.parametrize("method", ["bogus", "Euler", None])
def test_modal_simulate_rejects_unsupported_integration_method(method):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="method must be 'euler' or 'rk4'"):
        representation.simulate([1.0, 2.0], [3.0], [0.0, 0.1], method=method)


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([], "time must be a non-empty 1D grid"),
        ([[0.0, 0.1]], "time must be a non-empty 1D grid"),
        ([0.0, np.nan], "time values must be finite"),
        ([0.0, 0.0], "time values must be strictly increasing"),
        ([0.1, 0.0], "time values must be strictly increasing"),
    ],
)
def test_modal_simulate_rejects_invalid_time_grids(time, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.simulate([1.0, 2.0], [3.0], time)


@pytest.mark.parametrize(
    ("z0", "u", "message"),
    [
        ([[1.0, 2.0]], [3.0], "z must be a 1D vector"),
        ([1.0, 2.0], [3.0, 4.0], "u must have shape"),
        ([1.0, 2.0], [[3.0], [4.0], [5.0]], "u must have shape"),
    ],
)
def test_modal_simulate_rejects_invalid_state_and_input_shapes(z0, u, message):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match=message):
        representation.simulate(z0, u, [0.0, 0.1])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_zero_input_response_matches_direct_simulation(method):
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        [[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]],
    ).modal_representation()
    z0 = np.array([1.0 + 0.5j, -0.25 - 2.0j])
    time = np.array([0.0, 0.05, 0.2, 0.3])

    states, outputs = representation.zero_input_response(
        z0, time, method=method
    )
    direct_states, direct_outputs = representation.simulate(
        z0, np.zeros(2), time, method=method
    )

    assert states.shape == (4, 2)
    assert outputs.shape == (4, 3)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], z0)
    np.testing.assert_array_equal(states, direct_states)
    np.testing.assert_array_equal(outputs, direct_outputs)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_zero_input_response_matches_coupled_physical_response(method):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    initial_state = np.array([1.25, -0.75, 2.5])
    initial_modal_state = modes.left_eigenvectors.conj().T @ initial_state
    time = np.array([0.0, 0.025, 0.1, 0.2])

    physical_states, physical_outputs = system.zero_input_response(
        initial_state, time, method=method
    )
    modal_states, modal_outputs = representation.zero_input_response(
        initial_modal_state, time, method=method
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    np.testing.assert_allclose(
        reconstructed_states, physical_states, atol=1e-11
    )
    np.testing.assert_allclose(modal_outputs, physical_outputs, atol=1e-11)


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize(
    "inputs",
    [
        np.array([0.4, -1.2]),
        np.array(
            [[0.4, -1.2], [0.0, 0.5], [-0.75, 0.25], [2.0, -1.0]]
        ),
    ],
    ids=["constant", "time_varying"],
)
def test_modal_forced_response_matches_direct_zero_state_simulation(
    method, inputs
):
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        [[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]],
    ).modal_representation()
    time = np.array([0.0, 0.05, 0.2, 0.3])

    states, outputs = representation.forced_response(
        inputs, time, method=method
    )
    direct_states, direct_outputs = representation.simulate(
        np.zeros(2), inputs, time, method=method
    )

    assert states.shape == (4, 2)
    assert outputs.shape == (4, 3)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], np.zeros(2))
    np.testing.assert_array_equal(states, direct_states)
    np.testing.assert_array_equal(outputs, direct_outputs)


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize(
    "inputs",
    [
        np.array([0.4, -1.2]),
        np.array(
            [[0.4, -1.2], [0.0, 0.5], [-0.75, 0.25], [2.0, -1.0]]
        ),
    ],
    ids=["constant", "time_varying"],
)
def test_modal_forced_response_matches_coupled_physical_response(method, inputs):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    time = np.array([0.0, 0.025, 0.1, 0.2])

    physical_states, physical_outputs = system.forced_response(
        inputs, time, method=method
    )
    modal_states, modal_outputs = representation.forced_response(
        inputs, time, method=method
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    np.testing.assert_allclose(
        reconstructed_states, physical_states, atol=1e-11
    )
    np.testing.assert_allclose(modal_outputs, physical_outputs, atol=1e-11)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_step_response_matches_forced_response(method):
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        [[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]],
    ).modal_representation()
    amplitude = np.array([0.0, -2.5])
    time = np.array([0.0, 0.05, 0.2, 0.3])

    states, outputs = representation.step_response(
        amplitude, time, method=method
    )
    forced_states, forced_outputs = representation.forced_response(
        amplitude, time, method=method
    )

    assert states.shape == (4, 2)
    assert outputs.shape == (4, 3)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], np.zeros(2))
    np.testing.assert_array_equal(states, forced_states)
    np.testing.assert_array_equal(outputs, forced_outputs)


def test_modal_step_response_selects_one_input_channel():
    representation = StateSpace(
        np.diag([-1.0, -2.0]),
        np.eye(2),
        np.eye(2),
        np.zeros((2, 2)),
    ).modal_representation()

    states, outputs = representation.step_response(
        [0.0, 1.0], [0.0, 0.1, 0.2]
    )

    np.testing.assert_array_equal(states[:, 0], np.zeros(3))
    np.testing.assert_array_equal(outputs[:, 0], np.zeros(3))
    assert np.any(np.abs(states[1:, 1]) > 0.0)
    assert np.any(np.abs(outputs[1:, 1]) > 0.0)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_step_response_matches_coupled_physical_response(method):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    amplitude = np.array([0.0, -2.5])
    time = np.array([0.0, 0.025, 0.1, 0.2])

    physical_states, physical_outputs = system.step_response(
        amplitude, time, method=method
    )
    modal_states, modal_outputs = representation.step_response(
        amplitude, time, method=method
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    np.testing.assert_allclose(
        reconstructed_states, physical_states, atol=1e-11
    )
    np.testing.assert_allclose(modal_outputs, physical_outputs, atol=1e-11)


@pytest.mark.parametrize("amplitude", [1.0, [1.0], [0.0, np.inf]])
def test_modal_step_response_rejects_invalid_amplitude(amplitude):
    representation = StateSpace(
        np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))
    ).modal_representation()

    with pytest.raises(ValueError, match="amplitude"):
        representation.step_response(amplitude, [0.0, 0.1])


def test_modal_step_response_rejects_unsupported_method():
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="method must be 'euler' or 'rk4'"):
        representation.step_response([1.0], [0.0, 0.1], method="bogus")


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_impulse_response_matches_explicit_numerical_pulse(method):
    representation = StateSpace(
        [[-1.0, -3.0], [2.0, -1.0]],
        [[1.0, -2.0], [0.5, 3.0]],
        [[1.0, 2.0], [-0.5, 3.0], [2.0, -1.0]],
        [[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]],
    ).modal_representation()
    impulse = np.array([2.0, -1.5])
    time = np.array([0.0, 0.05, 0.2, 0.3])
    pulse = np.zeros((time.size, 2))
    pulse[0] = impulse / (time[1] - time[0])

    states, outputs = representation.impulse_response(
        impulse, time, method=method
    )
    forced_states, forced_outputs = representation.forced_response(
        pulse, time, method=method
    )

    assert states.shape == (4, 2)
    assert outputs.shape == (4, 3)
    assert np.iscomplexobj(states)
    assert np.iscomplexobj(outputs)
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(outputs))
    np.testing.assert_array_equal(states[0], np.zeros(2))
    np.testing.assert_array_equal(states, forced_states)
    np.testing.assert_array_equal(outputs, forced_outputs)
    np.testing.assert_allclose(pulse[0] * (time[1] - time[0]), impulse)
    np.testing.assert_array_equal(pulse[1:], np.zeros((3, 2)))


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_modal_impulse_response_matches_coupled_physical_response(method):
    system = StateSpace(
        [[-1.0, 2.0, 0.5], [-3.0, -1.0, 1.0], [0.25, -0.5, -4.0]],
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]],
        [[1.0, 0.0, 2.0], [-0.5, 3.0, 1.0]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    modes = system.biorthogonal_modes()
    representation = system.modal_representation()
    impulse = np.array([2.0, -1.5])
    time = np.array([0.0, 0.025, 0.1, 0.2])

    physical_states, physical_outputs = system.impulse_response(
        impulse, time, method=method
    )
    modal_states, modal_outputs = representation.impulse_response(
        impulse, time, method=method
    )
    reconstructed_states = modal_states @ modes.right_eigenvectors.T

    np.testing.assert_allclose(
        reconstructed_states, physical_states, atol=1e-11
    )
    np.testing.assert_allclose(modal_outputs, physical_outputs, atol=1e-11)


@pytest.mark.parametrize("impulse", [1.0, [1.0], [[1.0, 2.0]]])
def test_modal_impulse_response_rejects_invalid_impulse_shape(impulse):
    representation = StateSpace(
        np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))
    ).modal_representation()

    with pytest.raises(ValueError, match="impulse must have shape"):
        representation.impulse_response(impulse, [0.0, 0.1])


@pytest.mark.parametrize("time", [[], [0.0], [[0.0, 0.1]]])
def test_modal_impulse_response_requires_two_time_samples(time):
    representation = StateSpace(*valid_matrices()).modal_representation()

    with pytest.raises(ValueError, match="time must contain at least two samples"):
        representation.impulse_response([1.0], time)


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


def test_exact_forced_step_matches_scalar_constant_input_solution():
    system = StateSpace([[-2.0]], [[3.0]], [[1.0]], [[0.0]])
    state = np.array([1.25])
    control = np.array([0.5])
    dt = 0.4

    result = system.exact_forced_step(state, control, dt)

    expected = np.exp(-2.0 * dt) * state + 0.75 * (1.0 - np.exp(-2.0 * dt))
    np.testing.assert_allclose(result, expected)


def test_exact_forced_step_with_zero_input_matches_oscillator_solution():
    system = StateSpace(
        [[0.0, 1.0], [-1.0, 0.0]],
        [[0.0], [1.0]],
        np.eye(2),
        np.zeros((2, 1)),
    )
    state = np.array([1.5, -0.25])
    dt = 0.3

    result = system.exact_forced_step(state, [0.0], dt)

    expected = np.array(
        [
            np.cos(dt) * state[0] + np.sin(dt) * state[1],
            -np.sin(dt) * state[0] + np.cos(dt) * state[1],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_exact_forced_step_supports_singular_A_and_multiple_inputs():
    system = StateSpace(
        [[0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 2.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )
    state = np.array([1.0, -0.5])
    control = np.array([2.0, 3.0])
    dt = 0.25

    result = system.exact_forced_step(state, control, dt)

    acceleration = 2.0 * control[1]
    expected = np.array(
        [
            state[0] + dt * (state[1] + control[0]) + 0.5 * dt**2 * acceleration,
            state[1] + dt * acceleration,
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_exact_forced_step_is_more_accurate_than_euler():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    state = np.array([0.5])
    control = np.array([2.0])
    dt = 0.5
    analytical = 2.0 + (state - 2.0) * np.exp(-dt)

    exact = system.exact_forced_step(state, control, dt)
    euler = system.euler_step(state, control, dt)

    np.testing.assert_allclose(exact, analytical)
    assert np.linalg.norm(exact - analytical) < np.linalg.norm(euler - analytical)


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


def test_simulate_exact_uses_piecewise_constant_left_endpoint_inputs():
    system = StateSpace([[0.0]], [[1.0]], [[2.0]], [[3.0]])
    inputs = np.array([[1.0], [2.0], [100.0]])

    states, outputs = system.simulate(
        [0.0], inputs, [0.0, 0.5, 1.5], method="exact"
    )

    np.testing.assert_allclose(states, [[0.0], [0.5], [2.5]])
    np.testing.assert_allclose(outputs, [[3.0], [7.0], [305.0]])


def test_simulate_exact_matches_constant_input_analytical_trajectory():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    time = np.array([0.0, 0.1, 0.4, 1.0])

    states, outputs = system.simulate([0.5], [2.0], time, method="exact")

    expected = (2.0 - 1.5 * np.exp(-time))[:, np.newaxis]
    np.testing.assert_allclose(states, expected)
    np.testing.assert_allclose(outputs, expected)


@pytest.mark.parametrize("method", ["bogus", "Euler", None])
def test_simulate_rejects_unsupported_integration_method(method):
    system = StateSpace(*valid_matrices())

    with pytest.raises(
        ValueError, match="method must be 'euler', 'rk4', or 'exact'"
    ):
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


def test_forced_response_for_scalar_system_starts_from_zero():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])

    states, outputs = system.forced_response([2.0], [0.0, 0.1, 0.2])

    np.testing.assert_allclose(states, [[0.0], [0.2], [0.38]])
    np.testing.assert_allclose(outputs, states)
    assert states[0, 0] == 0.0
    assert states[-1, 0] != 0.0


def test_forced_response_accepts_time_varying_input_trajectory():
    system = StateSpace([[0.0]], [[1.0]], [[1.0]], [[0.0]])

    states, outputs = system.forced_response(
        [[1.0], [2.0], [3.0]], [0.0, 0.5, 1.0]
    )

    np.testing.assert_allclose(states, [[0.0], [0.5], [1.5]])
    np.testing.assert_allclose(outputs, states)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_forced_response_matches_direct_zero_state_simulation(method):
    system = StateSpace(*valid_matrices())
    time = [0.0, 0.1, 0.3]
    inputs = [[1.0], [2.0], [3.0]]

    response = system.forced_response(inputs, time, method=method)
    direct = system.simulate(np.zeros(2), inputs, time, method=method)

    for actual, expected in zip(response, direct, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_forced_response_for_multi_state_system():
    system = StateSpace(
        [[0.0, 1.0], [-1.0, -1.0]],
        [[0.0], [2.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0], [0.0]],
    )

    states, outputs = system.forced_response([0.5], [0.0, 0.1, 0.2], method="rk4")

    assert states.shape == (3, 2)
    assert outputs.shape == (3, 2)
    assert np.any(states[1:] != 0.0)
    np.testing.assert_allclose(outputs, states)


def test_forced_response_accepts_multiple_inputs():
    system = StateSpace(
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [-1.0, 3.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )

    states, _ = system.forced_response([2.0, -1.0], [0.0, 0.25, 0.5])

    np.testing.assert_allclose(states, [[0.0, 0.0], [0.0, -1.25], [0.0, -2.5]])


def test_forced_response_outputs_include_direct_feedthrough():
    system = StateSpace([[0.0]], [[1.0]], [[2.0]], [[3.0]])
    inputs = [[1.0], [2.0], [4.0]]

    states, outputs = system.forced_response(inputs, [0.0, 0.5, 1.0])

    np.testing.assert_allclose(states, [[0.0], [0.5], [1.5]])
    np.testing.assert_allclose(outputs, [[3.0], [7.0], [15.0]])


def test_step_response_for_scalar_first_order_system_trends_to_steady_state():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    time = np.linspace(0.0, 2.0, 21)

    states, outputs = system.step_response([1.0], time, method="rk4")

    assert states[0, 0] == 0.0
    assert np.all(np.diff(states[:, 0]) > 0.0)
    assert states[-1, 0] < 1.0
    np.testing.assert_allclose(outputs, states)


def test_step_response_accepts_non_unit_amplitude():
    system = StateSpace([[0.0]], [[2.0]], [[1.0]], [[0.0]])

    states, _ = system.step_response([3.0], [0.0, 0.25, 0.5])

    np.testing.assert_allclose(states, [[0.0], [1.5], [3.0]])


def test_step_response_accepts_multiple_input_amplitudes():
    system = StateSpace(
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [-1.0, 3.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )

    states, outputs = system.step_response([2.0, -1.0], [0.0, 0.25, 0.5])

    np.testing.assert_allclose(states, [[0.0, 0.0], [0.0, -1.25], [0.0, -2.5]])
    np.testing.assert_allclose(outputs, states)


def test_step_response_outputs_include_direct_feedthrough():
    system = StateSpace([[0.0]], [[1.0]], [[2.0]], [[3.0]])

    states, outputs = system.step_response([2.0], [0.0, 0.5, 1.0])

    np.testing.assert_allclose(states, [[0.0], [1.0], [2.0]])
    np.testing.assert_allclose(outputs, [[6.0], [8.0], [10.0]])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_step_response_matches_forced_response_with_constant_input(method):
    system = StateSpace(*valid_matrices())
    time = [0.0, 0.1, 0.3]
    amplitude = [2.0]

    response = system.step_response(amplitude, time, method=method)
    forced = system.forced_response(amplitude, time, method=method)

    for actual, expected in zip(response, forced, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("amplitude", [2.0, [[2.0]], [1.0, 2.0]])
def test_step_response_rejects_invalid_amplitude_shape(amplitude):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match=r"amplitude must have shape \(n_inputs,\)"):
        system.step_response(amplitude, [0.0, 0.1])


def test_impulse_response_for_scalar_system_uses_unit_area_pulse():
    system = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])

    states, outputs = system.impulse_response([1.0], [0.0, 0.1, 0.2, 0.3])

    np.testing.assert_allclose(states, [[0.0], [1.0], [0.9], [0.81]])
    np.testing.assert_allclose(outputs, states)


def test_impulse_response_accepts_non_unit_area():
    system = StateSpace([[0.0]], [[2.0]], [[1.0]], [[0.0]])

    states, _ = system.impulse_response([3.0], [0.0, 0.25, 0.5])

    np.testing.assert_allclose(states, [[0.0], [6.0], [6.0]])


def test_impulse_response_pulse_area_and_zero_later_inputs_with_feedthrough():
    system = StateSpace([[0.0]], [[0.0]], [[0.0]], [[2.0]])
    time = np.array([0.0, 0.25, 0.75])
    impulse = np.array([3.0])

    _, outputs = system.impulse_response(impulse, time)

    pulse = outputs[0] / 2.0
    np.testing.assert_allclose(pulse * (time[1] - time[0]), impulse)
    np.testing.assert_array_equal(outputs[1:], [[0.0], [0.0]])
    np.testing.assert_allclose(outputs[0], 2.0 * impulse / 0.25)


def test_impulse_response_accepts_multiple_input_areas():
    system = StateSpace(
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [-1.0, 3.0]],
        np.eye(2),
        np.zeros((2, 2)),
    )

    states, outputs = system.impulse_response([2.0, -1.0], [0.0, 0.25, 0.5])

    np.testing.assert_allclose(states, [[0.0, 0.0], [0.0, -5.0], [0.0, -5.0]])
    np.testing.assert_allclose(outputs, states)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_impulse_response_matches_explicit_forced_response(method):
    system = StateSpace(*valid_matrices())
    time = np.array([0.0, 0.1, 0.3])
    impulse = np.array([2.0])
    inputs = np.zeros((time.size, 1))
    inputs[0] = impulse / (time[1] - time[0])

    response = system.impulse_response(impulse, time, method=method)
    forced = system.forced_response(inputs, time, method=method)

    for actual, expected in zip(response, forced, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("impulse", [2.0, [[2.0]], [1.0, 2.0]])
def test_impulse_response_rejects_invalid_impulse_shape(impulse):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match=r"impulse must have shape \(n_inputs,\)"):
        system.impulse_response(impulse, [0.0, 0.1])


@pytest.mark.parametrize("time", [[], [0.0], 0.0])
def test_impulse_response_requires_at_least_two_time_samples(time):
    system = StateSpace(*valid_matrices())

    with pytest.raises(ValueError, match="time must contain at least two samples"):
        system.impulse_response([1.0], time)


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
