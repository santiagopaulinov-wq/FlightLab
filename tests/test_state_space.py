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
