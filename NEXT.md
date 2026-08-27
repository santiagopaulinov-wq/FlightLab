# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler and classical RK4 state stepping, Euler or RK4 simulation with constant or time-varying inputs, zero-input, zero-state forced-response, step-response, and finite-grid numerical impulse-response simulation, eigenvalue calculation, continuous-time modal properties, right-eigenvector modal shapes, and asymptotic-stability checks.
- Modal properties preserve eigenvalue ordering and provide natural frequency, damping ratio, damped natural frequency, period, and signed time constant where applicable; non-applicable quantities are represented by `None`.
- Right modal shapes use NumPy-normalized eigenvector columns, with column `i` corresponding to eigenvalue and modal-property result `i`; complex phase is preserved.
- Numerical impulse response uses a left-sampled rectangular pulse over the first interval with amplitude `impulse / (time[1] - time[0])` and zero input samples afterward.
- Generic trajectory analysis extracts component-wise minima and maxima with their first occurrence times from state or output trajectories.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

118 tests.

# Current architectural boundary

Linear dimensional state-space foundations, explicit Euler and RK4 simulation, zero-input, zero-state forced-response, step-response, finite-grid numerical impulse-response simulation, eigenvalue-based scalar modal properties, right modal shapes, asymptotic-stability analysis, and basic trajectory-extrema analysis are complete. Do not redesign these foundations next session unless a verified inconsistency is found.

# Next recommended technical step

Add general left-eigenvector modal shapes for `StateSpace`, preserving eigenvalue pairing and postponing participation-factor and aircraft-mode classification.
