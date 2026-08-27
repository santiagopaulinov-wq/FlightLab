import numpy as np


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
