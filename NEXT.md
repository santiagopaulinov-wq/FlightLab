# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler and classical RK4 state stepping, Euler or RK4 simulation with constant or time-varying inputs, zero-input and zero-state forced-response simulation, eigenvalue calculation, and continuous-time asymptotic-stability checks.
- Generic trajectory analysis extracts component-wise minima and maxima with their first occurrence times from state or output trajectories.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

89 tests.

# Current architectural boundary

Linear dimensional state-space foundations, explicit Euler and RK4 simulation, zero-input and zero-state forced-response simulation, eigenvalue-based asymptotic-stability analysis, and basic trajectory-extrema analysis are complete. Do not redesign these foundations next session unless a verified inconsistency is found.

# Next recommended technical step

Add a general step-response convenience for `StateSpace`, reusing the verified zero-state forced-response path and existing simulation input conventions.
