# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler stepping, constant or time-varying input simulation, eigenvalue calculation, and continuous-time asymptotic-stability checks.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

46 tests.

# Current architectural boundary

Linear dimensional state-space foundations, forward-Euler simulation, and eigenvalue-based asymptotic-stability analysis are complete. Do not redesign these foundations next session unless a verified inconsistency is found.

# Next recommended technical step

Add a small time-response analysis capability for extracting basic extrema and their occurrence times from simulated state or output trajectories.
