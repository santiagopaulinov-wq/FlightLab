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
