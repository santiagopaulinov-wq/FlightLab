from typing import NamedTuple

import numpy as np

_CONJUGATE_RTOL = 1e-7
_CONJUGATE_ATOL = 1e-10
_FAMILY_PROPERTY_RTOL = 1e-7
_FAMILY_PROPERTY_ATOL = 1e-10
_STABILITY_ATOL = 1e-10
_PREFILTER_DC_GAIN_ATOL = 100.0 * np.finfo(float).eps


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


class LuenbergerObserverInterconnection(NamedTuple):
    """Full-order plant and observer interconnection for a supplied gain.

    ``system`` has augmented state order ``[x; x_hat]``, retains the plant
    input ``u`` as its external input, and has output order ``[y; x_hat]``.
    ``observer_gain`` is the validated real matrix ``L`` with shape
    ``(n_states, n_outputs)``. This immutable result records the gain alongside
    the realization so its convention remains explicit.
    """

    system: "StateSpace"
    observer_gain: np.ndarray


class ObserverBasedOutputFeedbackInterconnection(NamedTuple):
    """Dynamic output-feedback interconnection for supplied ``K`` and ``L``.

    ``system`` has augmented state order ``[x; x_hat]``, uses ``v`` as its
    external input, and exposes the plant output ``y``. The validated gain
    copies record the conventions ``u = v - K x_hat`` and
    ``x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)``.
    """

    system: "StateSpace"
    state_feedback_gain: np.ndarray
    observer_gain: np.ndarray


class BalancedRealization(NamedTuple):
    """Full-order balanced system and its original-coordinate transformation.

    ``transformation`` is the real nonsingular matrix ``T`` in ``x = T @ z``,
    where ``x`` is the original state and ``z`` is the balanced state. Thus
    ``z = np.linalg.solve(T, x)`` maps an original state into balanced
    coordinates. No states are truncated.
    """

    system: "StateSpace"
    transformation: np.ndarray


class BalancedTruncation(NamedTuple):
    """Reduced balanced system and explicit original-state coordinate maps.

    ``projection @ x`` retains the first ``retained_order`` coordinates of the
    balanced state ``z = solve(balanced_transformation, x)``. Conversely,
    ``reconstruction @ x_reduced`` maps a reduced state back to the original
    state space with every discarded balanced coordinate set to zero. The
    reconstruction is therefore approximate in general.
    ``a_priori_error_bound`` is twice the sum of
    ``discarded_hankel_singular_values`` and bounds the induced input-output
    H-infinity norm error; it is not a state-reconstruction bound or equality.
    """

    system: "StateSpace"
    retained_order: int
    projection: np.ndarray
    reconstruction: np.ndarray
    balanced_transformation: np.ndarray
    retained_hankel_singular_values: np.ndarray
    discarded_hankel_singular_values: np.ndarray
    a_priori_error_bound: float


class FrequencyResponseSingularDirections(NamedTuple):
    """Reduced singular triplets for transfer matrices at explicit frequencies.

    ``left_singular_directions`` is ``U`` and
    ``right_singular_directions`` is ``V``, so each transfer matrix reconstructs
    as ``U @ diag(singular_values) @ V.conj().T``. Rows of ``U`` follow output
    channel order; rows of ``V`` follow input channel order. Individual vectors
    have arbitrary unit-magnitude complex phase, and bases within repeated-
    singular-value subspaces may rotate.
    """

    singular_values: np.ndarray
    left_singular_directions: np.ndarray
    right_singular_directions: np.ndarray


class BalancedTruncationErrorSingularDirections(NamedTuple):
    """Reduced singular triplets for sampled balanced-truncation errors.

    The sampled error reconstructs as
    ``Ue @ diag(error_singular_values) @ Ve.conj().T``. Rows of ``Ue`` follow
    original output-channel order and rows of ``Ve`` follow original
    input-channel order. Paired directions have arbitrary unit-complex phase,
    and bases within repeated-singular-value subspaces may rotate.
    """

    error_singular_values: np.ndarray
    left_error_singular_directions: np.ndarray
    right_error_singular_directions: np.ndarray


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

    def full_state_feedback(self, gain):
        """Return the static full-state feedback interconnection ``u = v - K x``.

        ``K`` must be a finite real two-dimensional array with shape
        ``(n_inputs, n_states)``. Here ``v`` is the new external closed-loop
        input and has the same dimension and channel order as the plant input
        ``u``. Substitution into ``x_dot = A x + B u`` and
        ``y = C x + D u`` gives ``A_cl = A - B K``, ``B_cl = B``,
        ``C_cl = C - D K``, and ``D_cl = D``.

        The returned :class:`StateSpace` is new and the plant is not mutated.
        For a plant with no input channels, the only valid gain has shape
        ``(0, n_states)``; the resulting zero-dimensional ``v`` leaves all
        plant matrices unchanged. This method interconnects a supplied gain
        only and performs no controller synthesis, tuning, or reference design.
        """
        raw_gain = np.asarray(gain)
        if np.iscomplexobj(raw_gain):
            raise TypeError("K must contain only real values")
        try:
            gain = np.asarray(gain, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("K must be a real numeric 2D array") from error
        if gain.ndim != 2:
            raise ValueError(
                "K must be a 2D array with shape (n_inputs, n_states)"
            )
        if gain.shape != (self.n_inputs, self.n_states):
            raise ValueError("K must have shape (n_inputs, n_states)")
        if not np.all(np.isfinite(gain)):
            raise ValueError("K must contain only finite values")

        return StateSpace(
            self.A - self.B @ gain,
            self.B.copy(),
            self.C - self.D @ gain,
            self.D.copy(),
        )

    def siso_reference_prefilter(self, state_feedback_gain):
        """Return nominal constant-reference scaling for stable SISO feedback.

        The conventions are ``u = v - K x`` and ``v = N r``, hence
        ``u = N r - K x``. For the realization returned by
        :meth:`full_state_feedback`, with ``A_cl = A - B K``, ``B_cl = B``,
        ``C_cl = C - D K``, and ``D_cl = D``, its scalar DC gain is
        ``G_cl(0) = -C_cl @ solve(A_cl, B_cl) + D_cl``. This method returns the
        finite real scalar ``N = 1 / G_cl(0)``, so the nominal equilibrium
        relation is ``y_ss = G_cl(0) N r = r``.

        The plant must have exactly one input and one output. ``K`` uses the
        same finite-real, two-dimensional validation and shape
        ``(1, n_states)`` as :meth:`full_state_feedback`, and ``A_cl`` must be
        asymptotically stable. DC response is evaluated through
        :meth:`frequency_response`, which uses a linear solve and includes
        nonzero ``D`` exactly. A nonfinite or materially complex DC result, or
        a DC gain whose magnitude is no larger than
        ``100 * machine epsilon``, is rejected as numerically unusable.

        This is nominal constant-reference scaling only. It provides no
        integral action, disturbance rejection, robustness guarantee, or
        reference dynamics. The plant is not mutated.
        """
        if self.n_inputs != 1 or self.n_outputs != 1:
            raise ValueError(
                "SISO reference prefilter requires exactly one input and one output"
            )

        closed_loop = self.full_state_feedback(state_feedback_gain)
        if not closed_loop.is_asymptotically_stable():
            raise ValueError(
                "SISO reference prefilter requires an asymptotically stable "
                "state-feedback closed loop"
            )

        dc_gain = closed_loop.frequency_response(0.0)[0, 0]
        if not np.isfinite(dc_gain):
            raise ValueError("state-feedback closed-loop DC gain must be finite")
        if abs(dc_gain.imag) > _CONJUGATE_ATOL + _CONJUGATE_RTOL * max(
            1.0, abs(dc_gain.real)
        ):
            raise ValueError("state-feedback closed-loop DC gain must be real")
        real_dc_gain = float(dc_gain.real)
        if abs(real_dc_gain) <= _PREFILTER_DC_GAIN_ATOL:
            raise ValueError(
                "state-feedback closed-loop DC gain is zero or numerically unusable"
            )

        prefilter_gain = 1.0 / real_dc_gain
        if not np.isfinite(prefilter_gain):
            raise ValueError("SISO reference prefilter gain must be finite")
        return float(prefilter_gain)

    def luenberger_observer(self, observer_gain):
        """Interconnect a supplied full-order continuous-time observer gain.

        The observer convention is
        ``x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)`` with estimation
        error ``e = x - x_hat``. Because the measured plant output is
        ``y = C x + D u``, the two feedthrough terms cancel in the innovation.
        Thus ``x_hat_dot = L C x + (A - L C) x_hat + B u`` and
        ``e_dot = (A - L C) e``.

        The returned immutable :class:`LuenbergerObserverInterconnection`
        contains an augmented :class:`StateSpace` with state order
        ``[x; x_hat]``, external input ``u``, and output order ``[y; x_hat]``::

            A_aug = [[A,   0],       B_aug = [[B],
                     [L C, A-L C]]            [B]]
            C_aug = [[C, 0],         D_aug = [[D],
                     [0, I]]                  [0]]

        ``L`` must be a finite real two-dimensional array with shape
        ``(n_states, n_outputs)``. Observability is not required. With no
        output channels, the valid shape is ``(n_states, 0)`` and the observer
        has no measurement correction. The plant is not mutated, and this
        method performs no observer-gain synthesis or output-feedback design.
        """
        raw_gain = np.asarray(observer_gain)
        if np.iscomplexobj(raw_gain):
            raise TypeError("L must contain only real values")
        try:
            observer_gain = np.asarray(observer_gain, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("L must be a real numeric 2D array") from error
        if observer_gain.ndim != 2:
            raise ValueError(
                "L must be a 2D array with shape (n_states, n_outputs)"
            )
        if observer_gain.shape != (self.n_states, self.n_outputs):
            raise ValueError("L must have shape (n_states, n_outputs)")
        if not np.all(np.isfinite(observer_gain)):
            raise ValueError("L must contain only finite values")

        state_count = self.n_states
        output_count = self.n_outputs
        correction = observer_gain @ self.C
        augmented_system = StateSpace(
            np.block(
                [
                    [self.A, np.zeros((state_count, state_count))],
                    [correction, self.A - correction],
                ]
            ),
            np.vstack((self.B, self.B)),
            np.block(
                [
                    [self.C, np.zeros((output_count, state_count))],
                    [
                        np.zeros((state_count, state_count)),
                        np.eye(state_count),
                    ],
                ]
            ),
            np.vstack((self.D, np.zeros((state_count, self.n_inputs)))),
        )
        return LuenbergerObserverInterconnection(
            augmented_system, observer_gain.copy()
        )

    def observer_based_output_feedback(self, state_feedback_gain, observer_gain):
        """Interconnect supplied state-feedback and observer gains.

        The control and full-order observer conventions are
        ``u = v - K x_hat`` and
        ``x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)``, where ``v`` is
        the new external command and the plant output is ``y = C x + D u``.
        Substitution cancels the two ``D u`` terms inside the innovation. With
        augmented state order ``[x; x_hat]``, external input ``v``, and output
        ``y``, the returned realization is::

            A_aug = [[A,   -B K],       B_aug = [[B],
                     [L C, A-B K-L C]]           [B]]
            C_aug = [C, -D K]           D_aug = D

        Equivalently, for ``e = x - x_hat``, the coordinates ``[x; e]`` obey
        ``x_dot = (A - B K) x + B K e + B v`` and
        ``e_dot = (A - L C) e``. The corresponding block-triangular dynamics
        establish the separation principle: the augmented eigenvalue multiset
        is the union of those of ``A - B K`` and ``A - L C``.

        ``K`` and ``L`` must be finite real two-dimensional arrays with shapes
        ``(n_inputs, n_states)`` and ``(n_states, n_outputs)`` respectively.
        Empty input or output channel dimensions are valid with their matching
        empty gain shapes. No controllability, observability, or stability is
        required. The plant is not mutated, and no gain synthesis, reference
        design, integral action, filtering, or saturation is performed.
        """
        raw_state_feedback_gain = np.asarray(state_feedback_gain)
        if np.iscomplexobj(raw_state_feedback_gain):
            raise TypeError("K must contain only real values")
        try:
            state_feedback_gain = np.asarray(state_feedback_gain, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("K must be a real numeric 2D array") from error
        if state_feedback_gain.ndim != 2:
            raise ValueError(
                "K must be a 2D array with shape (n_inputs, n_states)"
            )
        if state_feedback_gain.shape != (self.n_inputs, self.n_states):
            raise ValueError("K must have shape (n_inputs, n_states)")
        if not np.all(np.isfinite(state_feedback_gain)):
            raise ValueError("K must contain only finite values")

        raw_observer_gain = np.asarray(observer_gain)
        if np.iscomplexobj(raw_observer_gain):
            raise TypeError("L must contain only real values")
        try:
            observer_gain = np.asarray(observer_gain, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("L must be a real numeric 2D array") from error
        if observer_gain.ndim != 2:
            raise ValueError(
                "L must be a 2D array with shape (n_states, n_outputs)"
            )
        if observer_gain.shape != (self.n_states, self.n_outputs):
            raise ValueError("L must have shape (n_states, n_outputs)")
        if not np.all(np.isfinite(observer_gain)):
            raise ValueError("L must contain only finite values")

        feedback = self.B @ state_feedback_gain
        correction = observer_gain @ self.C
        augmented_system = StateSpace(
            np.block(
                [
                    [self.A, -feedback],
                    [correction, self.A - feedback - correction],
                ]
            ),
            np.vstack((self.B, self.B)),
            np.hstack((self.C, -self.D @ state_feedback_gain)),
            self.D.copy(),
        )
        return ObserverBasedOutputFeedbackInterconnection(
            augmented_system,
            state_feedback_gain.copy(),
            observer_gain.copy(),
        )

    def place_siso_observer_poles(self, desired_poles):
        """Return a single-output observer gain using Ackermann duality.

        Under :meth:`luenberger_observer`, the estimation error
        ``e = x - x_hat`` satisfies ``e_dot = (A - L C) e``. Since
        ``(A - L C).T = A.T - C.T L.T``, observer pole placement is the dual
        of state-feedback placement for the pair ``(A.T, C.T)``. This method
        applies :meth:`place_siso_poles` to that dual system and transposes its
        gain, preserving the existing NumPy-only Ackermann solve, Horner
        evaluation, and desired-pole validation semantics.

        The plant must have exactly one output, at least one state, and full
        observability under :meth:`is_fully_observable`. The finite real result
        has shape ``(n_states, 1)`` and is directly accepted by
        :meth:`luenberger_observer`. Desired poles need not be stable;
        estimation-error convergence requires the caller to choose poles with
        strictly negative real parts. The plant is not mutated. Multi-output
        placement, Kalman filtering, and output-feedback synthesis are not
        performed.
        """
        if self.n_outputs != 1:
            raise ValueError(
                "SISO observer pole placement requires exactly one output channel"
            )
        if self.n_states == 0:
            raise ValueError("SISO observer pole placement requires at least one state")
        if not self.is_fully_observable():
            raise ValueError(
                "SISO observer pole placement requires an observable system"
            )

        dual_system = StateSpace(
            self.A.T,
            self.C.T,
            np.empty((0, self.n_states)),
            np.empty((0, 1)),
        )
        observer_gain = dual_system.place_siso_poles(desired_poles).T
        if not np.all(np.isfinite(observer_gain)):
            raise ValueError("SISO observer pole placement produced a nonfinite gain")
        return observer_gain

    def place_siso_poles(self, desired_poles):
        """Return a SISO Ackermann gain compatible with ``u = v - K x``.

        The plant must have exactly one input, at least one state, and full
        controllability under :meth:`is_fully_controllable`. ``desired_poles``
        must be a finite numeric one-dimensional sequence of length
        ``n_states``. Complex poles must form a complete conjugate-closed
        multiset using ``rtol=1e-7`` and ``atol=1e-10`` so a real gain exists.

        For ``Ctrb = [B, A B, ..., A^(n-1) B]`` and desired characteristic
        polynomial ``phi``, Ackermann's formula is
        ``K = e_n.T @ inv(Ctrb) @ phi(A)``. This implementation obtains the
        selector row with a linear solve and evaluates ``phi(A)`` by Horner's
        method; it forms no explicit inverse. The finite real result has shape
        ``(1, n_states)`` and directly places the eigenvalues of ``A - B K``
        when passed to :meth:`full_state_feedback`.

        Desired poles need not be stable. Closed-loop stability is entirely the
        caller's responsibility. The plant is not mutated, and no MIMO pole
        placement, tuning, or optimal-control design is performed.
        """
        if self.n_inputs != 1:
            raise ValueError("SISO pole placement requires exactly one input channel")
        if self.n_states == 0:
            raise ValueError("SISO pole placement requires at least one state")
        if not self.is_fully_controllable():
            raise ValueError("SISO pole placement requires a controllable system")

        try:
            poles = np.asarray(desired_poles, dtype=complex)
        except (TypeError, ValueError) as error:
            raise TypeError("desired poles must be a numeric 1D sequence") from error
        if poles.ndim != 1:
            raise ValueError("desired poles must be a 1D sequence")
        if poles.size != self.n_states:
            raise ValueError("desired poles must contain exactly n_states values")
        if not np.all(np.isfinite(poles)):
            raise ValueError("desired poles must contain only finite values")

        matched = np.zeros(poles.size, dtype=bool)
        for index, pole in enumerate(poles):
            if matched[index]:
                continue
            if abs(pole.imag) <= _CONJUGATE_ATOL:
                matched[index] = True
                continue
            candidates = [
                candidate_index
                for candidate_index in range(index + 1, poles.size)
                if not matched[candidate_index]
                and np.isclose(
                    poles[candidate_index],
                    np.conj(pole),
                    rtol=_CONJUGATE_RTOL,
                    atol=_CONJUGATE_ATOL,
                )
            ]
            if not candidates:
                raise ValueError(
                    "complex desired poles must include matching conjugates"
                )
            matched[index] = True
            matched[candidates[0]] = True

        coefficients = np.poly(poles)
        coefficient_scale = np.maximum(1.0, np.abs(coefficients.real))
        if np.any(
            np.abs(coefficients.imag)
            > _CONJUGATE_ATOL + _CONJUGATE_RTOL * coefficient_scale
        ):
            raise ValueError("desired poles do not define a real polynomial")
        coefficients = coefficients.real

        identity = np.eye(self.n_states)
        polynomial_matrix = identity.copy()
        for coefficient in coefficients[1:]:
            polynomial_matrix = polynomial_matrix @ self.A + coefficient * identity

        selector = np.zeros(self.n_states)
        selector[-1] = 1.0
        try:
            ackermann_row = np.linalg.solve(
                self.controllability_matrix().T, selector
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "SISO pole placement controllability matrix is numerically singular"
            ) from error
        gain = np.asarray((ackermann_row @ polynomial_matrix)[np.newaxis, :])
        if not np.all(np.isfinite(gain)):
            raise ValueError("SISO pole placement produced a nonfinite gain")
        return gain

    def frequency_response(self, angular_frequencies):
        """Evaluate ``G(j omega)`` at explicit angular frequencies in rad/s.

        The transfer matrix is
        ``C @ solve(1j * omega * I - A, B) + D``. A finite real scalar returns
        one complex array with shape ``(n_outputs, n_inputs)``. A nonempty
        finite real one-dimensional array returns shape
        ``(n_frequencies, n_outputs, n_inputs)`` in the supplied frequency
        order. Zero input and output dimensions are preserved.

        Stability is not required. If ``1j * omega * I - A`` is singular at a
        requested frequency, a ``ValueError`` is raised because the frequency
        response is undefined at that pole. Linear solves are used; no matrix
        inverse or automatic frequency grid is formed.
        """
        frequencies = np.asarray(angular_frequencies)
        if frequencies.ndim not in (0, 1):
            raise ValueError(
                "angular frequencies must be a real scalar or 1D array"
            )
        if frequencies.ndim == 1 and frequencies.size == 0:
            raise ValueError("angular frequency array must contain at least one value")
        if np.iscomplexobj(frequencies):
            raise TypeError("angular frequencies must be real numeric values")
        try:
            frequencies = np.asarray(frequencies, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("angular frequencies must be real numeric values") from error
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("angular frequencies must contain only finite values")

        scalar_input = frequencies.ndim == 0
        frequency_values = np.atleast_1d(frequencies)
        response = np.empty(
            (frequency_values.size, self.n_outputs, self.n_inputs), dtype=complex
        )
        identity = np.eye(self.n_states, dtype=complex)
        for index, frequency in enumerate(frequency_values):
            resolvent = 1j * frequency * identity - self.A
            right_hand_side = (
                self.B
                if self.n_inputs > 0
                else np.zeros((self.n_states, 1), dtype=complex)
            )
            try:
                solved = np.linalg.solve(resolvent, right_hand_side)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "frequency response is undefined at angular frequency "
                    f"{frequency} rad/s because 1j * omega * I - A is singular"
                ) from error
            dynamic_response = (
                self.C @ solved
                if self.n_inputs > 0
                else np.empty((self.n_outputs, 0), dtype=complex)
            )
            response[index] = dynamic_response + self.D

        return response[0] if scalar_input else response

    def frequency_response_singular_values(self, angular_frequencies):
        """Return singular values of ``G(j omega)`` at frequencies in rad/s.

        This is a thin layer over :meth:`frequency_response` and therefore
        accepts the same finite real scalar or one-dimensional frequency input
        and preserves its ordering, validation, and pole errors. Singular
        values are real, nonnegative, preserve multiplicity, and are returned
        in descending order for each transfer matrix.

        A scalar frequency returns shape ``(min(n_outputs, n_inputs),)``. A
        frequency vector returns shape
        ``(n_frequencies, min(n_outputs, n_inputs))``. If either channel
        dimension is zero, the corresponding final dimension is zero. This
        method selects no grid and performs no maximization or H-infinity norm
        estimation.
        """
        response = self.frequency_response(angular_frequencies)
        singular_value_count = min(self.n_outputs, self.n_inputs)
        if singular_value_count == 0:
            return np.empty(response.shape[:-2] + (0,), dtype=float)
        return np.asarray(
            np.linalg.svd(response, compute_uv=False), dtype=float
        )

    def frequency_response_singular_directions(self, angular_frequencies):
        """Return reduced singular triplets of ``G(j omega)`` in rad/s.

        The method delegates frequency handling to :meth:`frequency_response`
        and applies ``np.linalg.svd(..., full_matrices=False)`` directly. For
        ``p`` outputs, ``m`` inputs, and ``k = min(p, m)``, scalar input returns
        singular values with shape ``(k,)``, left directions ``U`` with shape
        ``(p, k)``, and right directions ``V`` with shape ``(m, k)``. A vector
        of ``f`` frequencies returns shapes ``(f, k)``, ``(f, p, k)``, and
        ``(f, m, k)``. Empty channels preserve these shapes with ``k = 0``.

        ``G = U @ diag(singular_values) @ V.conj().T``. Rows of ``U`` correspond
        to output-channel order and rows of ``V`` to input-channel order.
        Singular vectors are not unique: each paired direction may carry an
        arbitrary unit-magnitude complex phase, and directions in a repeated-
        singular-value subspace may rotate. No phase normalization is imposed.
        This API selects no grid and performs no maximization or norm estimate.
        """
        response = self.frequency_response(angular_frequencies)
        singular_value_count = min(self.n_outputs, self.n_inputs)
        if singular_value_count == 0:
            leading_shape = response.shape[:-2]
            return FrequencyResponseSingularDirections(
                np.empty(leading_shape + (0,), dtype=float),
                np.empty(leading_shape + (self.n_outputs, 0), dtype=complex),
                np.empty(leading_shape + (self.n_inputs, 0), dtype=complex),
            )

        left, singular_values, right_conjugate_transpose = np.linalg.svd(
            response, full_matrices=False
        )
        right = np.swapaxes(right_conjugate_transpose.conj(), -2, -1)
        return FrequencyResponseSingularDirections(
            np.asarray(singular_values, dtype=float), left, right
        )

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

    def balanced_realization(self):
        """Return the full-order balanced realization and transformation.

        The returned :class:`BalancedRealization` uses ``x = T @ z``. Its
        system matrices are ``solve(T, A @ T)``, ``solve(T, B)``, ``C @ T``,
        and an unchanged ``D``. For an asymptotically stable minimal system,
        both balanced infinite-horizon Gramians equal the diagonal matrix of
        descending Hankel singular values up to floating-point roundoff.

        Cholesky factors of the existing Gramians and an SVD of their cross
        product are used. A ``ValueError`` is raised for a nonstable or
        nonminimal realization, or when the Gramians, Hankel values, or
        resulting transformation are nonfinite, not numerically positive
        definite, or numerically singular under an epsilon-scaled threshold.
        This method changes coordinates only and performs no model reduction.
        """
        if not self.is_asymptotically_stable():
            raise ValueError(
                "balanced realization requires an asymptotically stable system"
            )
        if not self.is_minimal_realization():
            raise ValueError("balanced realization requires a minimal realization")

        controllability_gramian = self.controllability_gramian()
        observability_gramian = self.observability_gramian()
        try:
            controllability_factor = np.linalg.cholesky(controllability_gramian)
            observability_factor = np.linalg.cholesky(observability_gramian)
            left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
                observability_factor.T @ controllability_factor
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "balanced realization requires numerically positive-definite "
                "Gramian factors"
            ) from error

        if not all(
            np.all(np.isfinite(values))
            for values in (
                controllability_factor,
                observability_factor,
                left_vectors,
                singular_values,
                right_vectors_transposed,
            )
        ):
            raise ValueError(
                "balanced realization Gramian factorization produced invalid values"
            )

        scale = float(np.max(singular_values, initial=0.0))
        singular_tolerance = (
            np.finfo(float).eps * max(1, self.n_states) * scale
        )
        if np.any(singular_values <= singular_tolerance):
            raise ValueError(
                "balanced realization Gramian factors are numerically singular"
            )

        inverse_sqrt_values = 1.0 / np.sqrt(singular_values)
        transformation = (
            controllability_factor
            @ right_vectors_transposed.T
            @ np.diag(inverse_sqrt_values)
        )
        inverse_transformation = (
            np.diag(inverse_sqrt_values)
            @ left_vectors.T
            @ observability_factor.T
        )
        inverse_residual = inverse_transformation @ transformation - np.eye(
            self.n_states
        )
        residual_tolerance = 100.0 * np.finfo(float).eps * max(1, self.n_states)
        if (
            not np.all(np.isfinite(transformation))
            or np.linalg.matrix_rank(transformation) < self.n_states
            or not np.allclose(
                inverse_residual, 0.0, rtol=0.0, atol=residual_tolerance
            )
        ):
            raise ValueError(
                "balanced realization state transformation is numerically singular"
            )

        try:
            balanced_system = StateSpace(
                np.linalg.solve(transformation, self.A @ transformation),
                np.linalg.solve(transformation, self.B),
                self.C @ transformation,
                self.D.copy(),
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "balanced realization state transformation is numerically singular"
            ) from error
        return BalancedRealization(balanced_system, transformation)

    def balanced_truncation(self, retained_order):
        """Return an explicitly ordered balanced truncation of this system.

        ``retained_order`` must be an integer ``r`` satisfying ``1 <= r < n``;
        order ``n`` is intentionally rejected because
        :meth:`balanced_realization` is the full-order API. Starting with its
        convention ``x = T @ z``, this method retains ``z[:r]`` and returns
        the leading blocks ``A_bal[:r, :r]``, ``B_bal[:r, :]``,
        ``C_bal[:, :r]``, and the unchanged ``D``.

        The returned projection ``P`` maps ``x`` to ``P @ x = z[:r]``. The
        reconstruction ``R`` maps a reduced state to ``R @ z[:r]`` in the
        original state space, setting discarded balanced coordinates to zero.
        Consequently this is an approximate model reduction, not an equivalent
        coordinate transformation. Stability, minimality, and numerical
        factorization errors are inherited from :meth:`balanced_realization`.
        ``a_priori_error_bound`` is the classical diagnostic
        ``2 * sum(discarded_hankel_singular_values)`` satisfying
        ``||G - G_r||_inf <= a_priori_error_bound`` for the input-output
        induced H-infinity norm. It is an upper bound, not an equality, an
        estimate of sampled response error, or a state-reconstruction bound.
        No order is selected automatically.
        """
        if isinstance(retained_order, (bool, np.bool_)) or not isinstance(
            retained_order, (int, np.integer)
        ):
            raise TypeError("retained order must be an integer")
        retained_order = int(retained_order)
        if not 1 <= retained_order < self.n_states:
            raise ValueError("retained order must satisfy 1 <= r < n_states")

        balanced_result = self.balanced_realization()
        balanced_system = balanced_result.system
        transformation = balanced_result.transformation
        projection = np.linalg.solve(
            transformation, np.eye(self.n_states)
        )[:retained_order, :]
        reconstruction = transformation[:, :retained_order].copy()
        reduced_system = StateSpace(
            balanced_system.A[:retained_order, :retained_order].copy(),
            balanced_system.B[:retained_order, :].copy(),
            balanced_system.C[:, :retained_order].copy(),
            balanced_system.D.copy(),
        )
        hankel_singular_values = self.hankel_singular_values()
        retained_hankel_singular_values = hankel_singular_values[
            :retained_order
        ].copy()
        discarded_hankel_singular_values = hankel_singular_values[
            retained_order:
        ].copy()
        a_priori_error_bound = float(
            2.0 * np.sum(discarded_hankel_singular_values, dtype=float)
        )
        if not np.isfinite(a_priori_error_bound) or a_priori_error_bound < 0.0:
            raise ValueError(
                "balanced truncation error bound must be finite and nonnegative"
            )
        return BalancedTruncation(
            reduced_system,
            retained_order,
            projection,
            reconstruction,
            transformation,
            retained_hankel_singular_values,
            discarded_hankel_singular_values,
            a_priori_error_bound,
        )

    def balanced_truncation_frequency_response_error(
        self, retained_order, angular_frequencies
    ):
        """Return explicit-frequency transfer errors for balanced truncation.

        For the caller-supplied order ``r``, this returns
        ``G(j omega) - G_r(j omega)`` using the original system and the reduced
        system returned by :meth:`balanced_truncation`. Angular frequencies are
        in rad/s and use :meth:`frequency_response` validation, ordering, pole
        behavior, and shapes: scalar input returns ``(n_outputs, n_inputs)``;
        vector input returns
        ``(n_frequencies, n_outputs, n_inputs)``. Empty channel axes are
        preserved whenever the underlying truncation context is valid.

        These are local input-output transfer-matrix samples only. They are not
        state-reconstruction errors, are not equal in general to
        ``BalancedTruncation.a_priori_error_bound``, and do not estimate the
        global H-infinity error. No grid, interpolation, maximization, or order
        selection is performed. Errors from the two underlying APIs propagate
        unchanged.
        """
        original_response = self.frequency_response(angular_frequencies)
        truncation = self.balanced_truncation(retained_order)
        reduced_response = truncation.system.frequency_response(angular_frequencies)
        return original_response - reduced_response

    def balanced_truncation_frequency_response_error_singular_values(
        self, retained_order, angular_frequencies
    ):
        """Return singular values of sampled balanced-truncation errors.

        This is a thin layer over
        :meth:`balanced_truncation_frequency_response_error`. For
        ``k = min(n_outputs, n_inputs)``, scalar frequency input returns real
        nonnegative descending singular values with shape ``(k,)``; vector
        input returns ``(n_frequencies, k)`` in caller order. Multiplicity and
        valid zero-channel shapes are preserved.

        The largest value at one frequency is the worst-case local input-output
        gain of the reduction error at that frequency. These sampled gains are
        not an H-infinity norm estimate and do not generally equal the global
        ``BalancedTruncation.a_priori_error_bound``. No grid, interpolation,
        maximization, or automatic order selection is performed. All errors
        from the underlying sampled-error API propagate unchanged.
        """
        error_response = self.balanced_truncation_frequency_response_error(
            retained_order, angular_frequencies
        )
        singular_value_count = min(self.n_outputs, self.n_inputs)
        if singular_value_count == 0:
            return np.empty(error_response.shape[:-2] + (0,), dtype=float)
        return np.asarray(
            np.linalg.svd(error_response, compute_uv=False), dtype=float
        )

    def balanced_truncation_frequency_response_error_singular_directions(
        self, retained_order, angular_frequencies
    ):
        """Return reduced singular triplets of sampled truncation errors.

        This method delegates to
        :meth:`balanced_truncation_frequency_response_error` and applies
        ``np.linalg.svd(..., full_matrices=False)``. For ``p`` outputs, ``m``
        inputs, and ``k = min(p, m)``, scalar input returns error singular
        values with shape ``(k,)``, left directions ``Ue`` with ``(p, k)``, and
        right directions ``Ve`` with ``(m, k)``. A vector of ``f`` frequencies
        returns ``(f, k)``, ``(f, p, k)``, and ``(f, m, k)``. Empty channels
        preserve these shapes with ``k = 0``.

        ``E = Ue @ diag(error_singular_values) @ Ve.conj().T``. Rows of ``Ue``
        follow original output-channel order; rows of ``Ve`` follow original
        input-channel order. Paired directions may differ by arbitrary unit-
        magnitude complex phase, and bases in repeated-singular-value subspaces
        may rotate. No phase normalization is imposed. All underlying order,
        stability, minimality, frequency, and pole errors propagate unchanged;
        no grid, interpolation, maximization, or norm estimate is added.
        """
        error_response = self.balanced_truncation_frequency_response_error(
            retained_order, angular_frequencies
        )
        singular_value_count = min(self.n_outputs, self.n_inputs)
        if singular_value_count == 0:
            leading_shape = error_response.shape[:-2]
            return BalancedTruncationErrorSingularDirections(
                np.empty(leading_shape + (0,), dtype=float),
                np.empty(leading_shape + (self.n_outputs, 0), dtype=complex),
                np.empty(leading_shape + (self.n_inputs, 0), dtype=complex),
            )

        left, singular_values, right_conjugate_transpose = np.linalg.svd(
            error_response, full_matrices=False
        )
        right = np.swapaxes(right_conjugate_transpose.conj(), -2, -1)
        return BalancedTruncationErrorSingularDirections(
            np.asarray(singular_values, dtype=float), left, right
        )

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
