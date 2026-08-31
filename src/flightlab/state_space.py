from typing import NamedTuple

import numpy as np

_CONJUGATE_RTOL = 1e-7
_CONJUGATE_ATOL = 1e-10
_FAMILY_PROPERTY_RTOL = 1e-7
_FAMILY_PROPERTY_ATOL = 1e-10
_STABILITY_ATOL = 1e-10


def _shared_family_property(properties, name):
    values = [getattr(member_properties, name) for member_properties in properties]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"modal family members have inconsistent {name}")
    numeric_values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(numeric_values)) or not np.allclose(
        numeric_values,
        numeric_values[0],
        rtol=_FAMILY_PROPERTY_RTOL,
        atol=_FAMILY_PROPERTY_ATOL,
    ):
        raise ValueError(f"modal family members have inconsistent {name}")
    return float(np.mean(numeric_values))


def _modal_families_are_consistent(first, second):
    if first.is_oscillatory != second.is_oscillatory:
        return False
    if first.multiplicity != second.multiplicity:
        return False

    for first_member, second_member in zip(
        first.members, second.members, strict=True
    ):
        if not np.isclose(
            first_member.eigenvalue,
            second_member.eigenvalue,
            rtol=_CONJUGATE_RTOL,
            atol=_CONJUGATE_ATOL,
        ):
            return False
        if first_member.dominant_state_indices != second_member.dominant_state_indices:
            return False
        if not np.allclose(
            first_member.participation_magnitudes,
            second_member.participation_magnitudes,
            rtol=1e-7,
            atol=1e-12,
        ):
            return False
        for first_value, second_value in zip(
            first_member.modal_properties,
            second_member.modal_properties,
            strict=True,
        ):
            if first_value is None or second_value is None:
                if first_value is not second_value:
                    return False
            elif not np.isclose(
                first_value,
                second_value,
                rtol=_FAMILY_PROPERTY_RTOL,
                atol=_FAMILY_PROPERTY_ATOL,
            ):
                return False

    return True


def _matrix_exponential(matrix):
    """Compute a matrix exponential with scaling and a degree-13 Pade approximant."""
    matrix = np.asarray(matrix)
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    norm = np.linalg.norm(matrix, 1)
    if norm == 0.0:
        return identity

    theta_13 = 5.371920351148152
    squarings = max(0, int(np.ceil(np.log2(norm / theta_13))))
    scaled = matrix / 2**squarings
    scaled_2 = scaled @ scaled
    scaled_4 = scaled_2 @ scaled_2
    scaled_6 = scaled_4 @ scaled_2
    coefficients = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )
    u = scaled @ (
        scaled_6
        @ (
            coefficients[13] * scaled_6
            + coefficients[11] * scaled_4
            + coefficients[9] * scaled_2
        )
        + coefficients[7] * scaled_6
        + coefficients[5] * scaled_4
        + coefficients[3] * scaled_2
        + coefficients[1] * identity
    )
    v = (
        scaled_6
        @ (
            coefficients[12] * scaled_6
            + coefficients[10] * scaled_4
            + coefficients[8] * scaled_2
        )
        + coefficients[6] * scaled_6
        + coefficients[4] * scaled_4
        + coefficients[2] * scaled_2
        + coefficients[0] * identity
    )
    result = np.linalg.solve(v - u, v + u)
    for _ in range(squarings):
        result = result @ result
    return result


class ModalProperties(NamedTuple):
    """Modal quantities for one continuous-time eigenvalue.

    Quantities that do not apply are represented by ``None``. In particular,
    real modes have no oscillatory quantities, and a zero eigenvalue has no
    derived modal quantities.
    """

    eigenvalue: complex
    natural_frequency: float | None
    damping_ratio: float | None
    damped_natural_frequency: float | None
    period: float | None
    time_constant: float | None


class ModalStateCharacterization(NamedTuple):
    """Physical-state participation summary for one eigenmode."""

    eigenvalue: complex
    modal_properties: ModalProperties
    participation_magnitudes: np.ndarray
    dominant_state_indices: tuple[int, ...]


class ModalFamily(NamedTuple):
    """One real mode or one complex-conjugate pair in canonical mode order."""

    members: tuple[ModalStateCharacterization, ...]
    is_oscillatory: bool

    @property
    def eigenvalues(self):
        """Return the family's eigenvalues in member order."""
        return tuple(member.eigenvalue for member in self.members)

    @property
    def multiplicity(self):
        """Return the number of individual modes in the family."""
        return len(self.members)


class ModalFamilyStateParticipation(NamedTuple):
    """Normalized physical-state participation summary for one modal family."""

    family: ModalFamily
    participation_magnitudes: np.ndarray
    dominant_state_indices: tuple[int, ...]


class ModalFamilyDynamicSummary(NamedTuple):
    """Generic shared dynamic quantities for one canonical modal family."""

    family: ModalFamily
    is_oscillatory: bool
    real_part: float
    natural_frequency: float | None
    damping_ratio: float | None
    damped_natural_frequency: float | None
    period: float | None
    time_constant: float | None
    stability: str


class ModalFamilyInputInfluence(NamedTuple):
    """Physical-input influence magnitudes for one canonical modal family."""

    family: ModalFamily
    influence_magnitudes: np.ndarray
    dominant_input_indices: tuple[int, ...]


class ModalFamilyOutputInfluence(NamedTuple):
    """Physical-output influence magnitudes for one canonical modal family."""

    family: ModalFamily
    influence_magnitudes: np.ndarray
    dominant_output_indices: tuple[int, ...]


class ModalFamilyCharacterization(NamedTuple):
    """Consolidated verified summaries for one canonical modal family."""

    family: ModalFamily
    dynamics: ModalFamilyDynamicSummary
    state_participation: ModalFamilyStateParticipation
    input_influence: ModalFamilyInputInfluence
    output_influence: ModalFamilyOutputInfluence


class BiorthogonalModes(NamedTuple):
    """Paired modal vectors with right columns scaled against left columns."""

    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray


class StructuralAnalysis(NamedTuple):
    """Immutable structural properties of a continuous-time realization."""

    controllable: bool
    observable: bool
    minimal: bool
    stabilizable: bool
    detectable: bool


class NonstablePBHDiagnostic(NamedTuple):
    """PBH failures for one nonstable continuous-time eigenvalue."""

    eigenvalue: complex
    controllability_failed: bool
    observability_failed: bool


class ModalStateSpace(NamedTuple):
    """System matrices expressed in biorthogonal modal coordinates."""

    Lambda: np.ndarray
    G_modal: np.ndarray
    H_modal: np.ndarray
    D: np.ndarray

    def state_derivative(self, z, u):
        """Evaluate ``Lambda @ z + G_modal @ u`` in modal coordinates."""
        z = np.asarray(z)
        u = np.asarray(u)
        n_states = self.Lambda.shape[0]
        n_inputs = self.G_modal.shape[1]

        if z.ndim != 1 or z.shape != (n_states,):
            raise ValueError("z must be a 1D vector with shape (n_states,)")
        if u.ndim != 1 or u.shape != (n_inputs,):
            raise ValueError("u must be a 1D vector with shape (n_inputs,)")

        return self.Lambda @ z + self.G_modal @ u

    def output(self, z, u):
        """Evaluate ``H_modal @ z + D @ u`` in modal coordinates."""
        z = np.asarray(z)
        u = np.asarray(u)
        n_states = self.Lambda.shape[0]
        n_inputs = self.G_modal.shape[1]

        if z.ndim != 1 or z.shape != (n_states,):
            raise ValueError("z must be a 1D vector with shape (n_states,)")
        if u.ndim != 1 or u.shape != (n_inputs,):
            raise ValueError("u must be a 1D vector with shape (n_inputs,)")

        return self.H_modal @ z + self.D @ u

    def euler_step(self, z, u, dt):
        """Advance one modal-coordinate step using forward Euler."""
        z = np.asarray(z)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")

        return z + dt * self.state_derivative(z, u)

    def rk4_step(self, z, u, dt):
        """Advance one modal-coordinate step using classical Runge-Kutta."""
        z = np.asarray(z)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")

        k1 = self.state_derivative(z, u)
        k2 = self.state_derivative(z + dt * k1 / 2.0, u)
        k3 = self.state_derivative(z + dt * k2 / 2.0, u)
        k4 = self.state_derivative(z + dt * k3, u)

        return z + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def exact_step(self, z, dt):
        """Propagate unforced modal coordinates exactly over one interval."""
        z = np.asarray(z)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        if z.ndim != 1 or z.shape != (self.Lambda.shape[0],):
            raise ValueError("z must be a 1D vector with shape (n_states,)")

        return np.exp(np.diag(self.Lambda) * dt) * z

    def exact_forced_step(self, z, u, dt):
        """Propagate modal coordinates exactly for one constant-input interval."""
        homogeneous_state = self.exact_step(z, dt)
        u = np.asarray(u)
        n_inputs = self.G_modal.shape[1]
        if u.ndim != 1 or u.shape != (n_inputs,):
            raise ValueError("u must be a 1D vector with shape (n_inputs,)")

        scaled_eigenvalues = np.diag(self.Lambda) * dt
        forcing_factors = np.ones_like(scaled_eigenvalues, dtype=complex)
        np.divide(
            np.expm1(scaled_eigenvalues),
            scaled_eigenvalues,
            out=forcing_factors,
            where=scaled_eigenvalues != 0.0,
        )
        forcing = self.G_modal @ u
        return homogeneous_state + dt * forcing_factors * forcing

    def exact_zero_input_response(self, z0, time):
        """Propagate an unforced modal trajectory exactly over a time grid."""
        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty 1D grid")
        if not np.all(np.isfinite(time)):
            raise ValueError("time values must be finite")

        time_steps = np.diff(time)
        if np.any(time_steps <= 0):
            raise ValueError("time values must be strictly increasing")

        state = np.asarray(z0)
        n_states = self.Lambda.shape[0]
        if state.ndim != 1 or state.shape != (n_states,):
            raise ValueError("z must be a 1D vector with shape (n_states,)")

        zero_input = np.zeros(self.G_modal.shape[1])
        state_trajectory = np.empty((time.size, n_states), dtype=complex)
        state_trajectory[0] = state

        for index, dt in enumerate(time_steps, start=1):
            state = self.exact_step(state, dt)
            state_trajectory[index] = state

        output_trajectory = np.empty(
            (time.size, self.H_modal.shape[0]), dtype=complex
        )
        for index, state in enumerate(state_trajectory):
            output_trajectory[index] = self.output(state, zero_input)

        return state_trajectory, output_trajectory

    def exact_simulate(self, z0, u, time):
        """Simulate modal dynamics exactly with zero-order-hold inputs."""
        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty 1D grid")
        if not np.all(np.isfinite(time)):
            raise ValueError("time values must be finite")

        time_steps = np.diff(time)
        if np.any(time_steps <= 0):
            raise ValueError("time values must be strictly increasing")

        state = np.asarray(z0)
        u = np.asarray(u)
        n_states = self.Lambda.shape[0]
        n_inputs = self.G_modal.shape[1]
        n_outputs = self.H_modal.shape[0]

        if state.ndim != 1 or state.shape != (n_states,):
            raise ValueError("z must be a 1D vector with shape (n_states,)")
        if u.ndim == 1 and u.shape == (n_inputs,):
            input_trajectory = np.broadcast_to(u, (time.size, n_inputs))
        elif u.ndim == 2 and u.shape == (time.size, n_inputs):
            input_trajectory = u
        else:
            raise ValueError(
                "u must have shape (n_inputs,) or (n_time_samples, n_inputs)"
            )

        state_trajectory = np.empty((time.size, n_states), dtype=complex)
        state_trajectory[0] = state

        for index, dt in enumerate(time_steps, start=1):
            state = self.exact_forced_step(
                state, input_trajectory[index - 1], dt
            )
            state_trajectory[index] = state

        output_trajectory = np.empty((time.size, n_outputs), dtype=complex)
        for index, state in enumerate(state_trajectory):
            output_trajectory[index] = self.output(
                state, input_trajectory[index]
            )

        return state_trajectory, output_trajectory

    def simulate(self, z0, u, time, method="euler"):
        """Simulate modal dynamics using Euler or RK4 with left-endpoint inputs.

        ``u`` may be constant with shape ``(m,)`` or time-varying with shape
        ``(N, m)``. Modal-state and output trajectories have shapes ``(N, n)``
        and ``(N, p)``, respectively.
        """
        if not isinstance(method, str) or method not in ("euler", "rk4"):
            raise ValueError("method must be 'euler' or 'rk4'")

        step = self.euler_step if method == "euler" else self.rk4_step
        time = np.asarray(time, dtype=float)

        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty 1D grid")
        if not np.all(np.isfinite(time)):
            raise ValueError("time values must be finite")

        time_steps = np.diff(time)
        if np.any(time_steps <= 0):
            raise ValueError("time values must be strictly increasing")

        state = np.asarray(z0)
        u = np.asarray(u)
        n_states = self.Lambda.shape[0]
        n_inputs = self.G_modal.shape[1]
        n_outputs = self.H_modal.shape[0]

        if u.ndim == 1 and u.shape == (n_inputs,):
            input_trajectory = np.broadcast_to(u, (time.size, n_inputs))
        elif u.ndim == 2 and u.shape == (time.size, n_inputs):
            input_trajectory = u
        else:
            raise ValueError(
                "u must have shape (n_inputs,) or (n_time_samples, n_inputs)"
            )

        self.state_derivative(state, input_trajectory[0])

        state_trajectory = np.empty((time.size, n_states), dtype=complex)
        state_trajectory[0] = state

        for index, dt in enumerate(time_steps, start=1):
            state = step(state, input_trajectory[index - 1], dt)
            state_trajectory[index] = state

        output_trajectory = np.empty((time.size, n_outputs), dtype=complex)
        for index, state in enumerate(state_trajectory):
            output_trajectory[index] = self.output(
                state, input_trajectory[index]
            )

        return state_trajectory, output_trajectory

    def zero_input_response(self, z0, time, method="euler"):
        """Simulate the modal response from an initial state with zero input."""
        return self.simulate(
            z0, np.zeros(self.G_modal.shape[1]), time, method=method
        )

    def forced_response(self, u, time, method="euler"):
        """Simulate the modal response to an input from the zero state."""
        return self.simulate(
            np.zeros(self.Lambda.shape[0]), u, time, method=method
        )

    def step_response(self, amplitude, time, method="euler"):
        """Simulate the modal response to a constant input from the zero state."""
        amplitude = np.asarray(amplitude, dtype=float)
        if amplitude.ndim != 1 or amplitude.shape != (self.G_modal.shape[1],):
            raise ValueError("amplitude must have shape (n_inputs,)")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("amplitude values must be finite")
        return self.forced_response(amplitude, time, method=method)

    def impulse_response(self, impulse, time, method="euler"):
        """Simulate a finite-width modal approximation to an impulse.

        The impulse-area vector is applied as a left-sampled rectangular pulse
        over the first interval. All later input samples are zero.
        """
        impulse = np.asarray(impulse, dtype=float)
        n_inputs = self.G_modal.shape[1]
        if impulse.ndim != 1 or impulse.shape != (n_inputs,):
            raise ValueError("impulse must have shape (n_inputs,)")

        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must contain at least two samples")

        input_trajectory = np.zeros((time.size, n_inputs), dtype=float)
        input_trajectory[0] = impulse / (time[1] - time[0])
        return self.forced_response(input_trajectory, time, method=method)


class StateSpace:
    def __init__(self, A, B, C, D):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must have shape (n, n)")

        self.n_states = self.A.shape[0]

        if self.B.ndim != 2 or self.B.shape[0] != self.n_states:
            raise ValueError("B must have shape (n, m)")

        self.n_inputs = self.B.shape[1]

        if self.C.ndim != 2 or self.C.shape[1] != self.n_states:
            raise ValueError("C must have shape (p, n)")

        self.n_outputs = self.C.shape[0]

        if self.D.ndim != 2 or self.D.shape != (self.n_outputs, self.n_inputs):
            raise ValueError("D must have shape (p, m)")

        if not all(
            np.all(np.isfinite(matrix))
            for matrix in (self.A, self.B, self.C, self.D)
        ):
            raise ValueError("A, B, C, and D must contain only finite values")

    def eigenvalues(self):
        """Return the eigenvalues of the continuous-time system matrix."""
        return np.linalg.eigvals(self.A)

    def controllability_matrix(self):
        """Return ``[B, A B, ..., A^(n-1) B]`` for the continuous-time system."""
        blocks = []
        block = self.B
        for _ in range(self.n_states):
            blocks.append(block)
            block = self.A @ block
        if not blocks:
            return np.empty((0, 0), dtype=float)
        return np.hstack(blocks)

    def controllability_rank(self):
        """Return the numerical rank of the controllability matrix."""
        matrix = self.controllability_matrix()
        return 0 if matrix.size == 0 else int(np.linalg.matrix_rank(matrix))

    def is_fully_controllable(self):
        """Return whether the controllability matrix has full state rank."""
        return self.controllability_rank() == self.n_states

    def controllability_gramian(self):
        """Return the infinite-horizon continuous-time controllability Gramian.

        For an asymptotically stable system, the returned real symmetric matrix
        ``Wc`` is the unique solution of
        ``A @ Wc + Wc @ A.T + B @ B.T = 0``. Nonstable and neutral systems do
        not have the required finite infinite-horizon Gramian and raise a
        ``ValueError``.
        """
        if not self.is_asymptotically_stable():
            raise ValueError(
                "controllability Gramian requires an asymptotically stable system"
            )

        identity = np.eye(self.n_states)
        lyapunov_operator = np.kron(identity, self.A) + np.kron(self.A, identity)
        forcing = self.B @ self.B.T
        gramian = np.linalg.solve(
            lyapunov_operator, -forcing.reshape(-1, order="F")
        ).reshape((self.n_states, self.n_states), order="F")
        return np.asarray((gramian + gramian.T) / 2.0, dtype=float)

    def is_stabilizable(self):
        """Return whether every nonstable mode satisfies the PBH rank test.

        Rank uses NumPy's default SVD-based tolerance, matching the existing
        controllability and observability rank methods.
        """
        identity = np.eye(self.n_states)
        for eigenvalue in self.eigenvalues():
            if eigenvalue.real >= 0.0:
                pbh_matrix = np.hstack((eigenvalue * identity - self.A, self.B))
                if np.linalg.matrix_rank(pbh_matrix) < self.n_states:
                    return False
        return True

    def observability_matrix(self):
        """Return ``[C; C A; ...; C A^(n-1)]`` for the continuous-time system."""
        blocks = []
        block = self.C
        for _ in range(self.n_states):
            blocks.append(block)
            block = block @ self.A
        if not blocks:
            return np.empty((0, 0), dtype=float)
        return np.vstack(blocks)

    def observability_rank(self):
        """Return the numerical rank of the observability matrix."""
        matrix = self.observability_matrix()
        return 0 if matrix.size == 0 else int(np.linalg.matrix_rank(matrix))

    def is_fully_observable(self):
        """Return whether the observability matrix has full state rank."""
        return self.observability_rank() == self.n_states

    def observability_gramian(self):
        """Return the infinite-horizon continuous-time observability Gramian.

        For an asymptotically stable system, the returned real symmetric matrix
        ``Wo`` is the unique solution of
        ``A.T @ Wo + Wo @ A + C.T @ C = 0``. Nonstable and neutral systems do
        not have the required finite infinite-horizon Gramian and raise a
        ``ValueError``.
        """
        if not self.is_asymptotically_stable():
            raise ValueError(
                "observability Gramian requires an asymptotically stable system"
            )

        identity = np.eye(self.n_states)
        transposed_A = self.A.T
        lyapunov_operator = np.kron(identity, transposed_A) + np.kron(
            transposed_A, identity
        )
        forcing = self.C.T @ self.C
        gramian = np.linalg.solve(
            lyapunov_operator, -forcing.reshape(-1, order="F")
        ).reshape((self.n_states, self.n_states), order="F")
        return np.asarray((gramian + gramian.T) / 2.0, dtype=float)

    def hankel_singular_values(self):
        """Return stable continuous-time Hankel singular values in descending order.

        The values are ``sqrt(eigvals(Wc @ Wo))`` for the existing
        controllability and observability Gramians. A symmetric
        positive-semidefinite equivalent is used to avoid insignificant complex
        roundoff. Tiny negative eigenvalues are clipped to zero; materially
        negative or nonfinite results raise ``ValueError``. The Gramian APIs
        enforce asymptotic stability.
        """

        def clipped_psd_eigenvalues(values):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "Hankel singular value computation produced a nonfinite result"
                )
            scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
            tolerance = (
                100.0
                * np.finfo(float).eps
                * max(1, self.n_states)
                * scale
            )
            if np.any(values < -tolerance):
                raise ValueError(
                    "Hankel singular value computation produced a materially "
                    "negative result"
                )
            return np.clip(values, 0.0, None)

        controllability_gramian = self.controllability_gramian()
        observability_gramian = self.observability_gramian()

        controllability_eigenvalues, controllability_eigenvectors = np.linalg.eigh(
            controllability_gramian
        )
        controllability_eigenvalues = clipped_psd_eigenvalues(
            controllability_eigenvalues
        )
        clipped_psd_eigenvalues(np.linalg.eigvalsh(observability_gramian))

        gramian_square_root = (
            controllability_eigenvectors * np.sqrt(controllability_eigenvalues)
        ) @ controllability_eigenvectors.T
        squared_values_matrix = (
            gramian_square_root @ observability_gramian @ gramian_square_root
        )
        squared_values_matrix = (
            squared_values_matrix + squared_values_matrix.T
        ) / 2.0
        squared_values = clipped_psd_eigenvalues(
            np.linalg.eigvalsh(squared_values_matrix)
        )
        return np.sqrt(squared_values)[::-1]

    def is_detectable(self):
        """Return whether every nonstable mode satisfies the PBH rank test.

        Rank uses NumPy's default SVD-based tolerance, matching the existing
        controllability, observability, and stabilizability checks.
        """
        identity = np.eye(self.n_states)
        for eigenvalue in self.eigenvalues():
            if eigenvalue.real >= 0.0:
                pbh_matrix = np.vstack((eigenvalue * identity - self.A, self.C))
                if np.linalg.matrix_rank(pbh_matrix) < self.n_states:
                    return False
        return True

    def is_minimal_realization(self):
        """Return whether the realization is controllable and observable."""
        return self.is_fully_controllable() and self.is_fully_observable()

    def structural_analysis(self):
        """Return one immutable summary of the existing structural checks."""
        return StructuralAnalysis(
            controllable=self.is_fully_controllable(),
            observable=self.is_fully_observable(),
            minimal=self.is_minimal_realization(),
            stabilizable=self.is_stabilizable(),
            detectable=self.is_detectable(),
        )

    def nonstable_pbh_diagnostics(self):
        """Return PBH failures for nonstable modes in eigenvalue order.

        Nonstable modes that pass both PBH conditions are omitted. Rank uses
        NumPy's default SVD-based tolerance, matching the structural checks.
        """
        diagnostics = []
        identity = np.eye(self.n_states)
        for eigenvalue in self.eigenvalues():
            if eigenvalue.real < 0.0:
                continue
            controllability_matrix = np.hstack(
                (eigenvalue * identity - self.A, self.B)
            )
            observability_matrix = np.vstack(
                (eigenvalue * identity - self.A, self.C)
            )
            controllability_failed = bool(
                np.linalg.matrix_rank(controllability_matrix) < self.n_states
            )
            observability_failed = bool(
                np.linalg.matrix_rank(observability_matrix) < self.n_states
            )
            if controllability_failed or observability_failed:
                diagnostics.append(
                    NonstablePBHDiagnostic(
                        eigenvalue=eigenvalue,
                        controllability_failed=controllability_failed,
                        observability_failed=observability_failed,
                    )
                )
        return tuple(diagnostics)

    def right_eigenvectors(self):
        """Return normalized right eigenvectors as columns.

        Column ``i`` corresponds to eigenvalue ``i`` from :meth:`eigenvalues`
        and satisfies ``A @ v[:, i] = eigenvalues()[i] * v[:, i]``.
        NumPy's normalization and complex phase are preserved.
        """
        return np.linalg.eig(self.A).eigenvectors

    def left_eigenvectors(self):
        """Return normalized left eigenvectors as paired columns.

        Column ``i`` corresponds to eigenvalue ``i`` from :meth:`eigenvalues`
        and satisfies ``w[:, i].conj().T @ A = eigenvalues()[i] *
        w[:, i].conj().T``. NumPy's normalization and complex phase are
        preserved.
        """
        eigenvalues = self.eigenvalues()
        adjoint_result = np.linalg.eig(self.A.conj().T)
        available = np.ones(self.n_states, dtype=bool)
        matched_vectors = np.empty_like(adjoint_result.eigenvectors)

        for index, eigenvalue in enumerate(eigenvalues):
            distances = np.abs(adjoint_result.eigenvalues - np.conj(eigenvalue))
            distances[~available] = np.inf
            match = int(np.argmin(distances))
            if not np.isclose(
                adjoint_result.eigenvalues[match],
                np.conj(eigenvalue),
                rtol=1e-7,
                atol=1e-10,
            ):
                raise RuntimeError("could not match left eigenvector to eigenvalue")
            matched_vectors[:, index] = adjoint_result.eigenvectors[:, match]
            available[match] = False

        return matched_vectors

    def biorthogonal_modes(self):
        """Return paired modes scaled so each ``w_i^H @ v_i`` equals one.

        Left eigenvectors retain their NumPy normalization. Each right
        eigenvector is divided by its paired inner product with the left
        eigenvector. No additional sign or phase canonicalization is applied.
        """
        eigenvalues = self.eigenvalues()
        right_eigenvectors = self.right_eigenvectors().copy()
        left_eigenvectors = self.left_eigenvectors()

        for index in range(self.n_states):
            paired_product = (
                left_eigenvectors[:, index].conj().T
                @ right_eigenvectors[:, index]
            )
            if np.isclose(paired_product, 0.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "paired left/right eigenvector inner product is too close to zero"
                )
            right_eigenvectors[:, index] /= paired_product

        return BiorthogonalModes(
            eigenvalues=eigenvalues,
            right_eigenvectors=right_eigenvectors,
            left_eigenvectors=left_eigenvectors,
        )

    def participation_factors(self):
        """Return complex state participation factors by state row and mode column.

        For biorthogonally scaled modal vectors, element ``[k, i]`` is
        ``v[k, i] * conjugate(w[k, i])``. No magnitude conversion or further
        column normalization is applied.
        """
        modes = self.biorthogonal_modes()
        return modes.right_eigenvectors * np.conj(modes.left_eigenvectors)

    def modal_state_characterization(self):
        """Summarize normalized physical-state participation for each mode.

        Magnitudes are the absolute values of the existing participation
        factors, normalized within each modal column to have unit sum.
        Dominant indices include every state numerically tied for the maximum.
        """
        participation = self.participation_factors()
        modal_properties = self.modal_properties()
        characterizations = []

        for index, properties in enumerate(modal_properties):
            magnitudes = np.abs(participation[:, index])
            magnitude_sum = np.sum(magnitudes)
            if not np.isfinite(magnitude_sum) or magnitude_sum <= 0.0:
                raise ValueError("modal participation magnitudes cannot be normalized")
            normalized_magnitudes = magnitudes / magnitude_sum
            maximum = np.max(normalized_magnitudes)
            dominant_indices = tuple(
                int(state_index)
                for state_index in np.flatnonzero(
                    np.isclose(
                        normalized_magnitudes,
                        maximum,
                        rtol=1e-7,
                        atol=1e-12,
                    )
                )
            )
            characterizations.append(
                ModalStateCharacterization(
                    eigenvalue=properties.eigenvalue,
                    modal_properties=properties,
                    participation_magnitudes=normalized_magnitudes,
                    dominant_state_indices=dominant_indices,
                )
            )

        return tuple(characterizations)

    def modal_families(self):
        """Group canonical modes into real singletons and conjugate pairs.

        Eigenvalues are considered conjugate with ``rtol=1e-7`` and
        ``atol=1e-10``. An imaginary part within the absolute tolerance of zero
        is treated as real. Family order follows the first member's existing
        eigenvalue index, and member order also remains unchanged.
        """
        modes = self.modal_state_characterization()
        assigned = np.zeros(len(modes), dtype=bool)
        families = []

        for index, mode in enumerate(modes):
            if assigned[index]:
                continue

            eigenvalue = mode.eigenvalue
            if np.isclose(
                eigenvalue.imag, 0.0, rtol=0.0, atol=_CONJUGATE_ATOL
            ):
                families.append(ModalFamily((mode,), is_oscillatory=False))
                assigned[index] = True
                continue

            candidates = [
                candidate_index
                for candidate_index in range(index + 1, len(modes))
                if not assigned[candidate_index]
                and np.isclose(
                    modes[candidate_index].eigenvalue,
                    np.conj(eigenvalue),
                    rtol=_CONJUGATE_RTOL,
                    atol=_CONJUGATE_ATOL,
                )
            ]
            if not candidates:
                raise RuntimeError(
                    f"could not match conjugate mode for eigenvalue {eigenvalue!r}"
                )
            match = min(
                candidates,
                key=lambda candidate_index: (
                    abs(modes[candidate_index].eigenvalue - np.conj(eigenvalue)),
                    candidate_index,
                ),
            )
            families.append(
                ModalFamily((mode, modes[match]), is_oscillatory=True)
            )
            assigned[index] = True
            assigned[match] = True

        return tuple(families)

    def modal_family_state_participation(self):
        """Summarize normalized state participation for each modal family.

        Each value is the arithmetic mean of that state's existing unit-sum
        participation magnitude across the family's members. Thus a real
        singleton is unchanged, while a conjugate pair contributes both modes
        symmetrically. Dominant indices include every state tied for the maximum
        using ``rtol=1e-7`` and ``atol=1e-12``.
        """
        summaries = []

        for family in self.modal_families():
            member_magnitudes = np.stack(
                [member.participation_magnitudes for member in family.members]
            )
            participation_magnitudes = np.asarray(
                np.mean(member_magnitudes, axis=0), dtype=float
            )
            if (
                participation_magnitudes.shape != (self.n_states,)
                or not np.all(np.isfinite(participation_magnitudes))
                or np.any(participation_magnitudes < 0.0)
            ):
                raise ValueError(
                    "family participation magnitudes must be finite, "
                    "nonnegative, and have shape (n_states,)"
                )
            maximum = np.max(participation_magnitudes)
            dominant_indices = tuple(
                int(state_index)
                for state_index in np.flatnonzero(
                    np.isclose(
                        participation_magnitudes,
                        maximum,
                        rtol=1e-7,
                        atol=1e-12,
                    )
                )
            )
            summaries.append(
                ModalFamilyStateParticipation(
                    family=family,
                    participation_magnitudes=participation_magnitudes,
                    dominant_state_indices=dominant_indices,
                )
            )

        return tuple(summaries)

    def modal_family_dynamic_summaries(self):
        """Return shared dynamic quantities for each canonical modal family.

        Conjugate-member values must agree with ``rtol=1e-7`` and
        ``atol=1e-10`` before their arithmetic mean is reported. Stability is
        ``"decaying"`` below a real part of ``-1e-10``, ``"growing"`` above
        ``1e-10``, and ``"neutral"`` within that inclusive band.
        """
        summaries = []

        for family in self.modal_families():
            properties = [member.modal_properties for member in family.members]
            real_parts = np.asarray(
                [member.eigenvalue.real for member in family.members], dtype=float
            )
            if not np.all(np.isfinite(real_parts)) or not np.allclose(
                real_parts,
                real_parts[0],
                rtol=_FAMILY_PROPERTY_RTOL,
                atol=_FAMILY_PROPERTY_ATOL,
            ):
                raise ValueError(
                    "modal family members have inconsistent eigenvalue real parts"
                )
            real_part = float(np.mean(real_parts))
            if real_part < -_STABILITY_ATOL:
                stability = "decaying"
            elif real_part > _STABILITY_ATOL:
                stability = "growing"
            else:
                stability = "neutral"

            summaries.append(
                ModalFamilyDynamicSummary(
                    family=family,
                    is_oscillatory=family.is_oscillatory,
                    real_part=real_part,
                    natural_frequency=_shared_family_property(
                        properties, "natural_frequency"
                    ),
                    damping_ratio=_shared_family_property(
                        properties, "damping_ratio"
                    ),
                    damped_natural_frequency=_shared_family_property(
                        properties,
                        "damped_natural_frequency"
                    ),
                    period=_shared_family_property(properties, "period"),
                    time_constant=_shared_family_property(
                        properties, "time_constant"
                    ),
                    stability=stability,
                )
            )

        return tuple(summaries)

    def modal_input_influence(self):
        """Return ``W^H @ B`` with mode rows and physical-input columns.

        ``W`` is the paired left-eigenvector matrix from
        :meth:`biorthogonal_modes`. Complex sign and phase are preserved; no
        row or column normalization is applied.
        """
        modes = self.biorthogonal_modes()
        return np.asarray(modes.left_eigenvectors.conj().T @ self.B, dtype=complex)

    def modal_family_input_influence(self):
        """Summarize physical-input influence for each canonical modal family.

        For input ``j``, the family value is the arithmetic mean of
        ``abs(G_modal[i, j])`` over its member-mode rows. No normalization is
        applied. Dominant inputs include all maxima tied with ``rtol=1e-7`` and
        ``atol=1e-12``; an all-zero family has no dominant input.
        """
        families = self.modal_families()
        modal_influence = self.modal_input_influence()
        eigenvalues = self.eigenvalues()
        available = np.ones(self.n_states, dtype=bool)
        summaries = []

        for family in families:
            member_rows = []
            for member in family.members:
                distances = np.abs(eigenvalues - member.eigenvalue)
                distances[~available] = np.inf
                row = int(np.argmin(distances))
                if not np.isclose(
                    eigenvalues[row],
                    member.eigenvalue,
                    rtol=_CONJUGATE_RTOL,
                    atol=_CONJUGATE_ATOL,
                ):
                    raise RuntimeError(
                        "could not match modal family member to input-influence row"
                    )
                member_rows.append(row)
                available[row] = False

            influence_magnitudes = np.asarray(
                np.mean(np.abs(modal_influence[member_rows]), axis=0), dtype=float
            )
            if (
                influence_magnitudes.shape != (self.n_inputs,)
                or not np.all(np.isfinite(influence_magnitudes))
                or np.any(influence_magnitudes < 0.0)
            ):
                raise ValueError(
                    "family input influences must be finite, nonnegative, "
                    "and have shape (n_inputs,)"
                )
            if np.allclose(
                influence_magnitudes, 0.0, rtol=0.0, atol=1e-12
            ):
                dominant_indices = ()
            else:
                maximum = np.max(influence_magnitudes)
                dominant_indices = tuple(
                    int(input_index)
                    for input_index in np.flatnonzero(
                        np.isclose(
                            influence_magnitudes,
                            maximum,
                            rtol=1e-7,
                            atol=1e-12,
                        )
                    )
                )
            summaries.append(
                ModalFamilyInputInfluence(
                    family=family,
                    influence_magnitudes=influence_magnitudes,
                    dominant_input_indices=dominant_indices,
                )
            )

        return tuple(summaries)

    def modal_output_influence(self):
        """Return ``C @ V`` with physical-output rows and mode columns.

        ``V`` is the scaled right-eigenvector matrix from
        :meth:`biorthogonal_modes`. Complex sign and phase are preserved; no
        row or column normalization is applied. Direct feedthrough ``D`` is
        not part of this modal-state output matrix.
        """
        modes = self.biorthogonal_modes()
        return np.asarray(self.C @ modes.right_eigenvectors, dtype=complex)

    def modal_family_output_influence(self):
        """Summarize physical-output influence for each canonical modal family.

        For output ``k``, the family value is the arithmetic mean of
        ``abs(H_modal[k, i])`` over its member-mode columns. No normalization is
        applied. Dominant outputs include all maxima tied with ``rtol=1e-7``
        and ``atol=1e-12``; an all-zero family has no dominant output.
        """
        families = self.modal_families()
        modal_influence = self.modal_output_influence()
        eigenvalues = self.eigenvalues()
        available = np.ones(self.n_states, dtype=bool)
        summaries = []

        for family in families:
            member_columns = []
            for member in family.members:
                distances = np.abs(eigenvalues - member.eigenvalue)
                distances[~available] = np.inf
                column = int(np.argmin(distances))
                if not np.isclose(
                    eigenvalues[column],
                    member.eigenvalue,
                    rtol=_CONJUGATE_RTOL,
                    atol=_CONJUGATE_ATOL,
                ):
                    raise RuntimeError(
                        "could not match modal family member to output-influence "
                        "column"
                    )
                member_columns.append(column)
                available[column] = False

            influence_magnitudes = np.asarray(
                np.mean(np.abs(modal_influence[:, member_columns]), axis=1),
                dtype=float,
            )
            if (
                influence_magnitudes.shape != (self.n_outputs,)
                or not np.all(np.isfinite(influence_magnitudes))
                or np.any(influence_magnitudes < 0.0)
            ):
                raise ValueError(
                    "family output influences must be finite, nonnegative, "
                    "and have shape (n_outputs,)"
                )
            if np.allclose(
                influence_magnitudes, 0.0, rtol=0.0, atol=1e-12
            ):
                dominant_indices = ()
            else:
                maximum = np.max(influence_magnitudes)
                dominant_indices = tuple(
                    int(output_index)
                    for output_index in np.flatnonzero(
                        np.isclose(
                            influence_magnitudes,
                            maximum,
                            rtol=1e-7,
                            atol=1e-12,
                        )
                    )
                )
            summaries.append(
                ModalFamilyOutputInfluence(
                    family=family,
                    influence_magnitudes=influence_magnitudes,
                    dominant_output_indices=dominant_indices,
                )
            )

        return tuple(summaries)

    def modal_family_characterizations(self):
        """Link all existing verified summaries by canonical modal family."""
        dynamics = self.modal_family_dynamic_summaries()
        state_participation = self.modal_family_state_participation()
        input_influence = self.modal_family_input_influence()
        output_influence = self.modal_family_output_influence()
        components = (
            state_participation,
            input_influence,
            output_influence,
        )

        if any(len(component) != len(dynamics) for component in components):
            raise RuntimeError(
                "modal family characterization component counts are inconsistent"
            )

        characterizations = []
        for index, dynamic_summary in enumerate(dynamics):
            family = dynamic_summary.family
            component_summaries = (
                state_participation[index],
                input_influence[index],
                output_influence[index],
            )
            if any(
                not _modal_families_are_consistent(family, summary.family)
                for summary in component_summaries
            ):
                raise RuntimeError(
                    "modal family characterization components are inconsistent"
                )

            canonical_state = component_summaries[0]._replace(family=family)
            canonical_input = component_summaries[1]._replace(family=family)
            canonical_output = component_summaries[2]._replace(family=family)
            characterizations.append(
                ModalFamilyCharacterization(
                    family=family,
                    dynamics=dynamic_summary,
                    state_participation=canonical_state,
                    input_influence=canonical_input,
                    output_influence=canonical_output,
                )
            )

        return tuple(characterizations)

    def modal_coordinates(self, x):
        """Transform a physical state vector to modal coordinates ``W^H @ x``."""
        x = np.asarray(x)
        if x.ndim != 1 or x.shape != (self.n_states,):
            raise ValueError("x must have shape (n_states,)")

        modes = self.biorthogonal_modes()
        return np.asarray(modes.left_eigenvectors.conj().T @ x, dtype=complex)

    def reconstruct_state(self, z):
        """Reconstruct a physical state vector from modal coordinates ``V @ z``."""
        z = np.asarray(z)
        if z.ndim != 1 or z.shape != (self.n_states,):
            raise ValueError("z must have shape (n_states,)")

        modes = self.biorthogonal_modes()
        return np.asarray(modes.right_eigenvectors @ z, dtype=complex)

    def modal_representation(self):
        """Return ``(Lambda, G_modal, H_modal, D)`` in the modal basis.

        The resulting dynamics are ``z_dot = Lambda @ z + G_modal @ u`` and
        ``y = H_modal @ z + D @ u``.
        """
        modes = self.biorthogonal_modes()
        return ModalStateSpace(
            Lambda=np.diag(modes.eigenvalues),
            G_modal=self.modal_input_influence(),
            H_modal=self.modal_output_influence(),
            D=self.D,
        )

    def modal_properties(self):
        """Return modal quantities in the same order as :meth:`eigenvalues`."""
        properties = []
        for eigenvalue in self.eigenvalues():
            eigenvalue = complex(eigenvalue)
            real_part = eigenvalue.real
            imaginary_part = eigenvalue.imag

            if imaginary_part != 0.0:
                natural_frequency = abs(eigenvalue)
                damping_ratio = -real_part / natural_frequency
                damped_natural_frequency = abs(imaginary_part)
                period = 2.0 * np.pi / damped_natural_frequency
            else:
                natural_frequency = None
                damping_ratio = None
                damped_natural_frequency = None
                period = None

            time_constant = -1.0 / real_part if real_part != 0.0 else None
            properties.append(
                ModalProperties(
                    eigenvalue=eigenvalue,
                    natural_frequency=natural_frequency,
                    damping_ratio=damping_ratio,
                    damped_natural_frequency=damped_natural_frequency,
                    period=period,
                    time_constant=time_constant,
                )
            )

        return tuple(properties)

    def is_asymptotically_stable(self):
        """Return whether every eigenvalue has a strictly negative real part."""
        return bool(np.all(self.eigenvalues().real < 0.0))

    def state_derivative(self, x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if x.ndim != 1 or x.shape != (self.n_states,):
            raise ValueError("x must be a 1D vector with shape (n_states,)")
        if u.ndim != 1 or u.shape != (self.n_inputs,):
            raise ValueError("u must be a 1D vector with shape (n_inputs,)")

        return self.A @ x + self.B @ u

    def output(self, x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if x.ndim != 1 or x.shape != (self.n_states,):
            raise ValueError("x must be a 1D vector with shape (n_states,)")
        if u.ndim != 1 or u.shape != (self.n_inputs,):
            raise ValueError("u must be a 1D vector with shape (n_inputs,)")

        return self.C @ x + self.D @ u

    def euler_step(self, x, u, dt):
        x = np.asarray(x, dtype=float)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")

        return x + dt * self.state_derivative(x, u)

    def rk4_step(self, x, u, dt):
        """Advance one step using classical fourth-order Runge-Kutta."""
        x = np.asarray(x, dtype=float)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")

        k1 = self.state_derivative(x, u)
        k2 = self.state_derivative(x + dt * k1 / 2.0, u)
        k3 = self.state_derivative(x + dt * k2 / 2.0, u)
        k4 = self.state_derivative(x + dt * k3, u)

        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def exact_forced_step(self, x, u, dt):
        """Propagate exactly over one interval with constant input."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        dt = np.asarray(dt, dtype=float)

        if dt.ndim != 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        self.state_derivative(x, u)

        augmented = np.zeros(
            (self.n_states + self.n_inputs, self.n_states + self.n_inputs)
        )
        augmented[: self.n_states, : self.n_states] = self.A
        augmented[: self.n_states, self.n_states :] = self.B
        transition = _matrix_exponential(augmented * dt)
        return (
            transition[: self.n_states, : self.n_states] @ x
            + transition[: self.n_states, self.n_states :] @ u
        )

    def zero_input_response(self, x0, time, method="euler"):
        """Simulate the response from an initial state with zero input."""
        return self.simulate(x0, np.zeros(self.n_inputs), time, method=method)

    def forced_response(self, u, time, method="euler"):
        """Simulate the response to an input from the zero initial state."""
        return self.simulate(np.zeros(self.n_states), u, time, method=method)

    def step_response(self, amplitude, time, method="euler"):
        """Simulate the response to a constant input from the zero state."""
        amplitude = np.asarray(amplitude, dtype=float)
        if amplitude.ndim != 1 or amplitude.shape != (self.n_inputs,):
            raise ValueError("amplitude must have shape (n_inputs,)")
        return self.forced_response(amplitude, time, method=method)

    def impulse_response(self, impulse, time, method="euler"):
        """Simulate a finite-width numerical approximation to an impulse.

        This is not an exact Dirac delta. The requested impulse-area vector is
        applied as a left-sampled rectangular pulse over the first interval:
        ``u[0] = impulse / (time[1] - time[0])``. All later input samples are
        zero, so the first pulse's numerical area equals ``impulse``.
        """
        impulse = np.asarray(impulse, dtype=float)
        if impulse.ndim != 1 or impulse.shape != (self.n_inputs,):
            raise ValueError("impulse must have shape (n_inputs,)")

        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time must contain at least two samples")

        input_trajectory = np.zeros((time.size, self.n_inputs), dtype=float)
        input_trajectory[0] = impulse / (time[1] - time[0])
        return self.forced_response(input_trajectory, time, method=method)

    def simulate(self, x0, u, time, method="euler"):
        """Simulate using Euler or RK4 with left-endpoint inputs.

        Each output sample uses the input at its corresponding time sample.
        ``u`` may be constant with shape ``(m,)`` or time-varying with shape
        ``(N, m)``. Returns state and output trajectories with shapes ``(N, n)``
        and ``(N, p)``, respectively, where ``N`` is the number of time samples,
        ``n`` states, ``m`` inputs, and ``p`` outputs. ``method`` must be
        ``"euler"``, ``"rk4"``, or ``"exact"`` and defaults to ``"euler"``.
        The exact method treats each left-endpoint input as constant over its
        interval.
        """
        if not isinstance(method, str) or method not in ("euler", "rk4", "exact"):
            raise ValueError("method must be 'euler', 'rk4', or 'exact'")

        steps = {
            "euler": self.euler_step,
            "rk4": self.rk4_step,
            "exact": self.exact_forced_step,
        }
        step = steps[method]
        time = np.asarray(time, dtype=float)

        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty 1D grid")
        if not np.all(np.isfinite(time)):
            raise ValueError("time values must be finite")

        time_steps = np.diff(time)
        if np.any(time_steps <= 0):
            raise ValueError("time values must be strictly increasing")

        state = np.asarray(x0, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.ndim == 1 and u.shape == (self.n_inputs,):
            input_trajectory = np.broadcast_to(u, (time.size, self.n_inputs))
        elif u.ndim == 2 and u.shape == (time.size, self.n_inputs):
            input_trajectory = u
        else:
            raise ValueError(
                "u must have shape (n_inputs,) or (n_time_samples, n_inputs)"
            )

        self.state_derivative(state, input_trajectory[0])

        state_trajectory = np.empty((time.size, self.n_states), dtype=float)
        state_trajectory[0] = state

        for index, dt in enumerate(time_steps, start=1):
            state = step(state, input_trajectory[index - 1], dt)
            state_trajectory[index] = state

        output_trajectory = np.empty((time.size, self.n_outputs), dtype=float)
        for index, state in enumerate(state_trajectory):
            output_trajectory[index] = self.output(state, input_trajectory[index])

        return state_trajectory, output_trajectory
