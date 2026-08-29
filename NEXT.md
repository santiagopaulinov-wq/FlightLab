# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler and classical RK4 state stepping, Euler or RK4 simulation with constant or time-varying inputs, zero-input, zero-state forced-response, step-response, and finite-grid numerical impulse-response simulation, complete numerical and exact zero-order-hold modal-coordinate infrastructure, state participation factors, normalized modal state characterization by physical-state index, modal input and output influence, continuous-time modal properties, and asymptotic-stability checks.
- Modal properties preserve eigenvalue ordering and provide natural frequency, damping ratio, damped natural frequency, period, and signed time constant where applicable; non-applicable quantities are represented by `None`.
- Right modal shapes use NumPy-normalized eigenvector columns, with column `i` corresponding to eigenvalue and modal-property result `i`; complex phase is preserved.
- Left modal shapes are NumPy-normalized columns explicitly matched to the corresponding eigenvalue order and satisfy `w_i^H A = lambda_i w_i^H`; complex phase is preserved.
- Biorthogonal modes preserve paired left columns and scale only right columns by `w_i^H v_i`, producing `w_i^H v_i = 1`; numerically unsafe near-zero paired products raise an error.
- State participation factors use `p[k, i] = v[k, i] * conjugate(w[k, i])` from biorthogonal modes, preserving state rows, modal columns, and complex values without magnitude normalization.
- Modal input influence uses `G_modal = W^H B` from biorthogonal modes, preserving mode rows, physical-input columns, and complex values without normalization.
- Modal output influence uses `H_modal = C V` from biorthogonal modes, preserving physical-output rows, mode columns, and complex values without normalization; direct feedthrough `D` remains separate.
- Modal state decomposition uses `z = W^H x`, and reconstruction uses `x = V z`, preserving complex modal coordinates and recovering compatible physical state vectors within numerical tolerance.
- The modal state-space bundle provides `(Lambda, G_modal, H_modal, D)` for `z_dot = Lambda z + G_modal u` and `y = H_modal z + D u`, using the verified biorthogonal basis.
- Modal derivative and output evaluation apply the bundled matrices directly, preserve complex modal coordinates, and do not recompute modal quantities.
- Modal Euler and RK4 stepping follow the physical-coordinate integration conventions while preserving complex modal states.
- Modal time-grid simulation supports constant and left-sampled time-varying inputs with Euler or RK4, returning complex modal-state and output trajectories.
- Modal zero-input response delegates to modal simulation with a correctly sized zero-input vector.
- Modal forced response delegates to modal simulation with a correctly sized zero initial modal state and supports constant or time-varying inputs.
- Modal step response accepts the established full input-amplitude vector and delegates constant-input evaluation to modal forced response.
- Modal impulse response uses the same first-interval rectangular pulse convention as the physical system and delegates integration to modal forced response.
- Exact unforced modal stepping applies `exp(lambda_i * dt)` independently to each modal coordinate without a matrix-exponential dependency.
- Exact unforced modal time-grid propagation iterates with the verified exact step and evaluates outputs with zero input.
- Exact constant-input modal stepping combines the verified homogeneous step with a stable `expm1` forcing factor and an explicit zero-eigenvalue limit.
- Exact modal simulation applies the verified constant-input step over each interval using the established left-sampled zero-order-hold convention.
- Modal state characterization derives real nonnegative magnitudes from existing participation factors, normalizes each mode to unit sum, and reports all numerically tied dominant state indices without aircraft-specific labels.
- Numerical impulse response uses a left-sampled rectangular pulse over the first interval with amplitude `impulse / (time[1] - time[0])` and zero input samples afterward.
- Generic trajectory analysis extracts component-wise minima and maxima with their first occurrence times from state or output trajectories.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

270 tests.

# Current architectural boundary

The modal numerical infrastructure is sufficiently complete for the current project stage. FlightLab has moved into a generic modal flight-dynamics interpretation layer, beginning with normalized physical-state participation and dominant state indices. Aircraft-specific mode names and heuristics remain outside the current boundary. Do not redesign the verified numerical foundations unless a verified inconsistency is found.

# Next recommended technical step

Group conjugate eigenvalues into generic modal families while preserving individual-mode results, without assigning aircraft-specific mode names.
