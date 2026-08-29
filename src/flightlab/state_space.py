from typing import NamedTuple

import numpy as np


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


class BiorthogonalModes(NamedTuple):
    """Paired modal vectors with right columns scaled against left columns."""

    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray


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

    def eigenvalues(self):
        """Return the eigenvalues of the continuous-time system matrix."""
        return np.linalg.eigvals(self.A)

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

    def modal_input_influence(self):
        """Return ``W^H @ B`` with mode rows and physical-input columns.

        ``W`` is the paired left-eigenvector matrix from
        :meth:`biorthogonal_modes`. Complex sign and phase are preserved; no
        row or column normalization is applied.
        """
        modes = self.biorthogonal_modes()
        return np.asarray(modes.left_eigenvectors.conj().T @ self.B, dtype=complex)

    def modal_output_influence(self):
        """Return ``C @ V`` with physical-output rows and mode columns.

        ``V`` is the scaled right-eigenvector matrix from
        :meth:`biorthogonal_modes`. Complex sign and phase are preserved; no
        row or column normalization is applied. Direct feedthrough ``D`` is
        not part of this modal-state output matrix.
        """
        modes = self.biorthogonal_modes()
        return np.asarray(self.C @ modes.right_eigenvectors, dtype=complex)

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
        ``"euler"`` or ``"rk4"`` and defaults to ``"euler"``.
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
