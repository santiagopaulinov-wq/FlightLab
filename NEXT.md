# Current completed capabilities

- `StateSpace` supports output evaluation, forward-Euler and classical RK4 state stepping, Euler or RK4 simulation with constant or time-varying inputs, zero-input, zero-state forced-response, step-response, and finite-grid numerical impulse-response simulation, eigenvalue calculation, continuous-time modal properties, right- and left-eigenvector modal shapes, explicit biorthogonal modal scaling, modal state decomposition and reconstruction, a complete modal state-space representation with derivative, output, Euler/RK4 stepping, time-grid simulation, zero-input response, and forced response, state participation factors, modal input and output influence, and asymptotic-stability checks.
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
- Numerical impulse response uses a left-sampled rectangular pulse over the first interval with amplitude `impulse / (time[1] - time[0])` and zero input samples afterward.
- Generic trajectory analysis extracts component-wise minima and maxima with their first occurrence times from state or output trajectories.
- `LongitudinalModel` uses states `(u, w, q, theta)` with elevator input.
- `LateralDirectionalModel` uses states `(v, p, r, phi)` with aileron and rudder inputs.
- Both models use SI units and right-handed body axes (x forward, y right, z down).
- Both model formulations have been audited for physical, dimensional, and sign consistency.
- Synthetic longitudinal and lateral-directional dynamic-response tests exist.

# Current test count

192 tests.

# Current architectural boundary

Linear dimensional state-space foundations, explicit Euler and RK4 simulation, zero-input, zero-state forced-response, step-response, finite-grid numerical impulse-response simulation, eigenvalue-based scalar modal properties, paired right and left modal shapes with explicit biorthogonal scaling, modal state decomposition and reconstruction, a complete modal state-space representation with derivative, output, Euler/RK4 stepping, time-grid simulation, zero-input response, and forced response, state participation and modal input/output-influence analysis, asymptotic-stability analysis, and basic trajectory-extrema analysis are complete. Do not redesign these foundations next session unless a verified inconsistency is found.

# Next recommended technical step

Add a modal step-response helper using the verified forced response, without adding impulse response or aircraft-mode classification.
