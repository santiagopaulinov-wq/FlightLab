# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler stepping, and constant or time-varying input simulation.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

43 tests.

# Current architectural boundary

Linear dimensional state-space foundations and forward-Euler simulation are complete. Do not redesign these foundations next session unless a verified inconsistency is found.

# Next recommended technical step

Define the next flight-dynamics capability before implementation.
