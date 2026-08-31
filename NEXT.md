# FlightLab session checkpoint

## Current project stage

FlightLab now supports exact zero-order-hold forced propagation directly in
physical state coordinates, in addition to its existing modal numerical
infrastructure and generic modal flight-dynamics interpretation layer. The
interpretation capability characterizes how physical state indices participate
in each individual eigenmode and groups real modes and complex-conjugate pairs
into generic modal families with family-level state-participation summaries.
Longitudinal and lateral-directional physical-mode identification v1 are both
complete at their verified conservative baselines. The active development stage
is now general continuous-time controllability and observability analysis.
Every `StateSpace` can construct the standard controllability and observability
matrices, report their numerical ranks, and test full-state controllability,
observability, continuous-time stabilizability, and continuous-time
detectability, with all five results available through one immutable structural
summary and nonstable PBH failures available as immutable mode diagnostics.
Each modal family also has a generic dynamic summary derived from
its canonical member
properties and generic physical-input and physical-output influence summaries.
These verified summaries are available together through one consolidated,
immutable generic characterization per canonical family.
Stable minimal realizations can also be transformed into full-order balanced
coordinates, with no state truncation, through a real transformation using the
convention `x = T z`, or explicitly reduced by retaining a caller-selected
number of leading balanced coordinates. Each valid truncation reports the
classical a priori induced H-infinity error bound from its discarded Hankel
singular values and can be compared with the original transfer matrix at
explicit caller-supplied frequencies through both error matrices and their
directional gains and input/output singular directions. The aircraft-model
layer can interpret dominant generic state indices using each model's
established physical state labels without changing generic modal results, and
likewise interprets dominant input and output indices using the models'
established channel ordering.
Every `StateSpace` can also evaluate its complex continuous-time transfer
matrix, descending singular values, and corresponding input/output singular
directions at explicit real angular frequencies without requiring stability.
The controller-design foundation now includes a generic, non-mutating static
full-state feedback interconnection and NumPy-only Ackermann pole placement for
controllable SISO systems, plus a generic full-order Luenberger observer
interconnection and NumPy-only observer pole placement for fully observable
single-output systems. Caller-supplied controller and observer gains can now be
combined in a generic observer-based dynamic output-feedback realization, and
stable SISO full-state-feedback loops support nominal steady-state reference
prefilter calculation.
Both aircraft models can filter these immutable interpreted families by their
existing oscillatory status, mathematical stability, and exact dominant
state/input/output labels, with configurable global or per-category ANY, ALL,
or EXACT set matching for both inclusions and exclusions.

## Checkpoint commit

- Pre-checkpoint commit:
  `27571a8728d43f669e019d81748c3ac49c6b315c`
- Pre-checkpoint message: `feat: add observer-based output feedback`
- The current checkpoint adds nominal steady-state reference-prefilter
  calculation for stable SISO full-state-feedback loops.

## Current verification baseline

- Test count: 743 tests.
- `uv run pytest -q` passes.
- `.venv/bin/ruff check` passes.
- `git diff --check` passes.

## Completed continuous-time structural-analysis layer

- Infinite-horizon controllability Gramian for asymptotically stable systems,
  defined by `A Wc + Wc A.T + B B.T = 0` and returned as a real symmetric
  matrix from a NumPy-only Kronecker-sum solve.
- Infinite-horizon observability Gramian for asymptotically stable systems,
  defined by `A.T Wo + Wo A + C.T C = 0` with the same NumPy-only solve and
  exact returned symmetry.
- Hankel singular values `sqrt(eigvals(Wc Wo))` for asymptotically stable
  systems, returned as a descending real nonnegative vector with multiplicity
  preserved through a validated symmetric positive-semidefinite formulation.
- Stable unreachable and unobservable directions are verified to produce the
  expected zero-value multiplicity, including distinct deficient directions;
  stable minimal realizations are verified to have only positive values.
- `StateSpace.balanced_realization()` returns an immutable
  `BalancedRealization` containing the full-order balanced `StateSpace` and the
  real nonsingular transformation `T` for `x = T z`; original states map to
  balanced states with `solve(T, x)`.
- Balanced matrices follow `A_bal = T^-1 A T`, `B_bal = T^-1 B`,
  `C_bal = C T`, and `D_bal = D`. The balanced controllability and observability
  Gramians are equal and diagonal, with their shared descending diagonal equal
  to the existing Hankel singular values up to floating-point roundoff.
- Balancing uses NumPy-only Cholesky Gramian factors and an SVD of their cross
  product. It requires asymptotic stability and the existing minimal-
  realization check, preserves full state dimension, and performs no
  truncation. Nonfinite, non-positive-definite, and epsilon-threshold singular
  factorization or transformation results raise clear `ValueError`s.
- `StateSpace.balanced_truncation(r)` requires an explicit integer order with
  `1 <= r < n`; order `n` is rejected in favor of `balanced_realization()`, and
  no automatic order selection is performed.
- The immutable `BalancedTruncation` result contains the reduced `StateSpace`,
  retained order, original-to-reduced projection, reduced-to-original
  reconstruction, full balanced transformation, and retained descending
  Hankel singular values. For `x = T z`, projection is the first `r` rows of
  `T^-1` and reconstruction is the first `r` columns of `T`.
- Reduced matrices are exactly the leading balanced blocks and preserve `D`.
  Reconstruction sets discarded coordinates to zero, so state reconstruction
  and reduced input-output behavior are explicitly approximate. A stable
  diagonal system with a separated discarded Hankel value verifies a bounded,
  deterministic step-response discrepancy against the diagnostic bound without
  treating the sampled discrepancy as the induced-norm error.
- Every `BalancedTruncation` records its discarded Hankel singular values and
  finite real nonnegative `a_priori_error_bound`, computed exactly as
  `2 * sum(discarded_hankel_singular_values)` from the original stable minimal
  realization.
- The bound satisfies the classical continuous-time statement
  `||G - G_r||_inf <= bound` for the induced input-output H-infinity norm. It is
  diagnostic only: not an equality claim, sampled-response estimate,
  state-reconstruction bound, or automatic order-selection rule. Because valid
  truncations require `r < n` and stable minimal Hankel values are positive,
  their discarded sum is positive; the mathematically empty full-order tail is
  verified separately without permitting `balanced_truncation(n)`.
- `StateSpace.frequency_response(angular_frequencies)` evaluates
  `C solve(j omega I - A, B) + D` at a finite real scalar or nonempty finite
  real one-dimensional frequency array in rad/s, without an explicit inverse.
- Scalar frequency input returns complex shape `(n_outputs, n_inputs)`; vector
  input returns `(n_frequencies, n_outputs, n_inputs)` in caller order. MIMO,
  zero-input, zero-output, and jointly empty channel shapes are preserved, and
  direct feedthrough is included exactly.
- Frequency response does not require stability. A singular resolvent at an
  exact imaginary-axis pole raises a clear `ValueError`, including when there
  are no input channels. Nonfinite, complex, empty-vector, and non-scalar/non-1D
  inputs are rejected. No grid selection or H-infinity estimation is performed.
- `StateSpace.frequency_response_singular_values(angular_frequencies)` reuses
  `frequency_response()` and applies NumPy SVD independently to each returned
  transfer matrix. It therefore inherits rad/s units, caller ordering,
  feedthrough, validation, stability independence, and exact-pole errors.
- Singular values are real, nonnegative, descending, and multiplicity-
  preserving. Scalar input returns `(min(n_outputs, n_inputs),)` and vector
  input returns `(n_frequencies, min(n_outputs, n_inputs))`; empty channels
  preserve a zero final dimension. No grid selection, maximization, or
  H-infinity estimation is performed.
- `StateSpace.frequency_response_singular_directions(angular_frequencies)`
  returns immutable reduced-SVD triplets with `G = U diag(sigma) V^H`, reusing
  `frequency_response()` for rad/s units, ordering, validation, and pole errors.
- With `p` outputs, `m` inputs, and `k = min(p, m)`, scalar shapes are `(k,)`,
  `(p, k)`, and `(m, k)` for singular values, left directions `U`, and right
  directions `V`; vector input prepends the frequency dimension. Empty channel
  cases retain these shapes with `k = 0`.
- Rows of `U` follow output-channel order and rows of `V` follow input-channel
  order. No phase normalization is imposed: paired directions have arbitrary
  unit-magnitude complex phase, and repeated-value subspaces may rotate. Tests
  therefore verify reconstruction and orthonormality rather than raw phases.
- `StateSpace.balanced_truncation_frequency_response_error(r, frequencies)`
  returns `G(j omega) - G_r(j omega)` using the original system and the exact
  reduced realization produced by `balanced_truncation(r)`.
- Scalar and vector results preserve the existing complex frequency-response
  shapes and caller ordering, including empty channel axes where an underlying
  valid truncation is available. Direct feedthrough cancels because both
  systems retain the same `D`.
- Samples are local input-output transfer errors only: not state-reconstruction
  errors, equalities to the a priori bound, or estimates of the global
  H-infinity error. The API selects no grid and performs no interpolation,
  maximization, norm estimation, or automatic order selection. Underlying
  frequency, pole, retained-order, stability, and minimality errors propagate
  unchanged.
- `StateSpace.balanced_truncation_frequency_response_error_singular_values()`
  applies NumPy SVD directly to those sampled error matrices and returns real
  nonnegative descending directional gains with multiplicity preserved.
- For `k = min(n_outputs, n_inputs)`, scalar shape is `(k,)`, vector shape is
  `(n_frequencies, k)`, and empty channels preserve a zero final dimension.
  All underlying validation and pole behavior remains unchanged.
- The largest value at one frequency is the worst-case local input-output gain
  of the reduction error there. It is not a global H-infinity estimate and does
  not generally equal the a priori bound; representative samples are verified
  not to exceed that global bound without maximizing over frequency.
- `StateSpace.balanced_truncation_frequency_response_error_singular_directions()`
  returns immutable reduced-SVD error triplets satisfying
  `E = Ue diag(sigma_e) Ve^H` at the explicit caller frequencies.
- With `p` original outputs, `m` original inputs, and `k = min(p, m)`, scalar
  shapes are `(k,)`, `(p, k)`, and `(m, k)`; vector input prepends the frequency
  dimension, and empty channels retain `k = 0` shapes. Rows of `Ue` preserve
  original output order and rows of `Ve` preserve original input order.
- No phase normalization is imposed: paired directions have arbitrary unit-
  complex phase and repeated-value subspace bases may rotate. Reconstruction,
  orthonormality, phase-invariant channel ordering, and inherited errors are
  verified without adding a grid, maximization, or norm estimation.

## Controller-design foundation

- `StateSpace.full_state_feedback(K)` returns a new generic closed-loop
  realization under `u = v - K x`, where `v` is the external command in the
  plant's existing input dimension and channel order.
- The complete interconnection is `A_cl = A - B K`, `B_cl = B`,
  `C_cl = C - D K`, and `D_cl = D`, so nonzero plant feedthrough is handled in
  both the output equation and direct command path.
- Gains must be finite real two-dimensional arrays with shape
  `(n_inputs, n_states)`. A zero-input plant accepts only `(0, n_states)`, for
  which the zero-dimensional command and feedback leave every matrix unchanged.
- The plant is not mutated. This capability performs interconnection only: no
  pole placement, LQR, tuning, reference tracking, observers, saturation, or
  aircraft-specific controller behavior is included.
- `StateSpace.place_siso_poles(desired_poles)` returns a finite real gain with
  shape `(1, n_states)` that is directly accepted by `full_state_feedback()`
  under the same `u = v - K x` convention.
- Placement uses `Ctrb = [B, A B, ..., A^(n-1) B]` and Ackermann's formula
  `K = e_n^T Ctrb^-1 phi(A)`. The implementation forms no inverse: it obtains
  the selector row by solving the transposed controllability system and
  evaluates `phi(A)` by Horner's method.
- The plant must have exactly one input, at least one state, and full
  controllability according to the existing structural analysis. Desired
  poles must be a finite one-dimensional sequence of length `n_states`.
- Complex desired poles must be conjugate-closed within `rtol=1e-7` and
  `atol=1e-10`, so the polynomial and gain are real. Unsupported MIMO and
  uncontrollable plants, invalid pole arrays, and numerically singular solves
  raise clear errors without mutating the plant.
- Requested poles are not required to be stable. The caller controls the
  closed-loop pole set and is responsible for its stability; no MIMO
  placement, optimal design, gain tuning, or aircraft-specific behavior is
  included.
- `StateSpace.luenberger_observer(L)` returns an immutable
  `LuenbergerObserverInterconnection` containing the augmented `StateSpace`
  and validated observer gain. It synthesizes no gain and does not require
  observability.
- The observer convention is
  `x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)`, with estimation error
  `e = x - x_hat` and exact error dynamics `e_dot = (A - L C) e`.
- The augmented state order is `[x; x_hat]`, the external input is the known
  plant input `u`, and output order is `[y; x_hat]`. Its matrices are
  `A_aug = [[A, 0], [L C, A-L C]]`, `B_aug = [[B], [B]]`,
  `C_aug = [[C, 0], [0, I]]`, and `D_aug = [[D], [0]]`.
- Plant feedthrough cancels from the observer state equation because measured
  `y = C x + D u` and the innovation subtracts the same `D u`; it remains in
  the exposed plant-output rows. Gains must be finite real matrices with shape
  `(n_states, n_outputs)`.
- A zero-output plant accepts `(n_states, 0)` and has no measurement
  correction. The plant is not mutated, and no observer synthesis, Kalman
  filtering, output-feedback control, or aircraft-specific behavior is added.
- `StateSpace.place_siso_observer_poles(desired_poles)` returns a finite real
  observer gain with shape `(n_states, 1)` that is directly compatible with
  `luenberger_observer()` and the established `e_dot = (A - L C)e` convention.
- Placement uses `(A - L C).T = A.T - C.T L.T` and delegates to the existing
  NumPy-only `place_siso_poles()` implementation on the dual pair `(A.T,
  C.T)`. Ackermann evaluation, linear solves, Horner evaluation, and pole-input
  semantics therefore remain centralized rather than duplicated.
- The plant must have exactly one output, at least one state, and full
  observability under the existing structural analysis. Desired poles must be
  finite, one-dimensional, contain exactly `n_states` values, and be
  conjugate-closed within `rtol=1e-7` and `atol=1e-10` when complex.
- Requested observer poles need not be stable. Estimation convergence is the
  caller's responsibility and requires strictly negative real parts. The
  plant is not mutated, and no multi-output placement, Kalman filtering,
  output-feedback synthesis, or aircraft-specific tuning is added.
- `StateSpace.observer_based_output_feedback(K, L)` returns an immutable
  `ObserverBasedOutputFeedbackInterconnection` containing the augmented system
  and validated gain copies. It uses `u = v - K x_hat`, retains `v` as the
  external input, exposes plant output `y`, and orders states as `[x; x_hat]`.
- The exact matrices are `A_aug = [[A, -B K], [L C, A-B K-L C]]`,
  `B_aug = [[B], [B]]`, `C_aug = [C, -D K]`, and `D_aug = D`.
- Feedthrough is retained in the exposed output. No algebraic loop is present:
  `u` depends on `v` and `x_hat`, while measured `D u` cancels the observer's
  subtracted `D u` inside the innovation.
- In equivalent `[x; e]` coordinates with `e = x - x_hat`, dynamics are
  `x_dot = (A-B K)x + B K e + B v` and `e_dot = (A-L C)e`. Therefore the
  augmented eigenvalue multiset is the union of those of `A-B K` and `A-L C`.
- Finite real gains must have shapes `(n_inputs, n_states)` and
  `(n_states, n_outputs)`. Matching empty dimensions are valid. No structural
  or stability restriction is imposed, the plant is not mutated, and no gain
  synthesis, prefilter, integral action, Kalman filter, or saturation is added.
- `StateSpace.siso_reference_prefilter(K)` returns a finite real scalar `N` for
  `u = N r - K x`, using the verified full-state-feedback realization and its
  zero-frequency response.
- For `A_cl = A-B K`, `B_cl = B`, `C_cl = C-D K`, and `D_cl = D`, the method
  evaluates `G_cl(0) = -C_cl solve(A_cl, B_cl) + D_cl` without an inverse and
  returns `N = 1 / G_cl(0)`. Nonzero `D` is therefore included exactly.
- The plant must have one input and one output, `K` retains existing validation,
  and `A_cl` must be asymptotically stable. Nonfinite, materially complex, and
  zero or machine-epsilon-scale DC gains are rejected clearly.
- At nominal equilibrium, `y_ss = G_cl(0) N r = r`. This is constant-reference
  scaling only, not integral action, disturbance rejection, robust tracking,
  or reference-model dynamics. The plant is not mutated.
- Nonstable and neutral systems raise clear errors because no finite
  infinite-horizon Gramians are returned; stable zero-input and zero-output
  systems return correctly shaped zero Gramians.
- Controllability matrix, numerical rank, and full controllability.
- Observability matrix, numerical rank, and full observability.
- Minimal-realization check from the existing controllability and observability
  results.
- PBH stabilizability for every eigenvalue with nonnegative real part.
- PBH detectability for every eigenvalue with nonnegative real part.
- Immutable `StructuralAnalysis`, containing exactly the existing controllable,
  observable, minimal, stabilizable, and detectable booleans.
- Immutable `NonstablePBHDiagnostic` entries returned in an immutable tuple for
  nonstable eigenvalues that fail PBH controllability, observability, or both.
- Focused diagnostic tests verify that unstable complex-conjugate modes remain
  separate and in `eigenvalues()` order, and that repeated failing nonstable
  eigenvalues retain one entry per occurrence while stable and passing modes
  remain omitted.
- A defective nonstable Jordan block is verified to retain one diagnostic per
  algebraic eigenvalue occurrence in `eigenvalues()` order despite having only
  one independent eigenvector.
- One explicit multistate system verifies controllability-only,
  observability-only, and combined PBH failure flags in eigenvalue order while
  a fourth nonstable mode that passes both conditions is omitted.
- Four representative systems verify that aggregate diagnostic failure flags
  are exactly consistent with `is_stabilizable()` and `is_detectable()`,
  including the fully passing case with no diagnostics.
- The same four structural outcomes are verified for an eigenvalue exactly
  equal to zero, including its treatment as nonstable and omission when both
  PBH conditions pass.
- A purely imaginary conjugate pair is verified across the same four outcomes,
  with separate ordered diagnostics for both members when either PBH condition
  fails and no diagnostics when both pass.
- Valid zero-input, zero-output, and jointly empty-channel systems are verified
  to report exact PBH failure flags for ordered nonstable modes while omitting
  stable modes.
- An asymptotically stable system with no input or output channels is verified
  to remain stabilizable and detectable with no diagnostics despite being
  neither fully controllable nor fully observable.
- Focused PBH diagnostic coverage is complete for stable, unstable, neutral,
  repeated, defective, complex-conjugate, mixed-outcome, and empty-channel
  cases.

## Important existing capabilities

- General continuous-time `StateSpace` construction, output evaluation,
  asymptotic-stability checks, Euler/RK4 stepping, time-grid simulation, and
  zero-input, forced, step, and finite-grid impulse responses.
- Exact physical-state forced stepping and simulation for piecewise-constant
  left-endpoint inputs, using an augmented-matrix exponential that supports
  singular `A`, multiple inputs, zero input, constant input, and time-varying
  zero-order-hold input trajectories without adding dependencies.
- Exact simulation preserves the existing output convention `y = C x + D u`,
  with each output evaluated using the input at that same time sample.
- Longitudinal states `("u", "w", "q", "theta")` with input order
  `("delta_e",)`, and lateral-directional states `("v", "p", "r", "phi")`
  with input order `("delta_a", "delta_r")`, using SI units and right-handed
  body axes `(x forward, y right, z down)`. Each model's output order equals its
  state order.
- Eigenvalues and continuous-time modal properties in stable eigenvalue order.
- Paired right/left eigenvectors and verified biorthogonal modes satisfying
  `W^H V ~= I`.
- State participation factors `p[k, i] = v[k, i] * conjugate(w[k, i])`.
- Modal coordinates `z = W^H x`, reconstruction `x = V z`, and modal matrices
  `(Lambda, G_modal, H_modal, D)`.
- Modal derivative/output evaluation, Euler/RK4 stepping and simulation, and
  zero-input, forced, step, and finite-grid impulse response helpers.
- Exact unforced and constant-input modal stepping, exact unforced trajectories,
  and exact zero-order-hold modal simulation without SciPy.
- `ModalStateCharacterization` and
  `StateSpace.modal_state_characterization()`, providing one result per
  eigenmode with its eigenvalue, existing `ModalProperties`, unit-sum
  nonnegative state-participation magnitudes, and all dominant state indices.
- Characterization preserves eigenvalue ordering, is invariant to reciprocal
  biorthogonal vector scaling, and treats conjugate modes consistently without
  collapsing them.
- `ModalFamily` and `StateSpace.modal_families()`, grouping each real mode into
  a singleton non-oscillatory family and each numerical complex-conjugate pair
  into one oscillatory family while retaining the canonical individual
  `ModalStateCharacterization` members.
- Modal-family conjugacy uses `rtol=1e-7` and `atol=1e-10`; imaginary parts
  within the absolute tolerance of zero are treated as real. Family order is
  set by each family's first canonical mode index, and member order is
  preserved. Unmatched genuinely complex modes raise a clear error.
- `ModalFamilyStateParticipation` and
  `StateSpace.modal_family_state_participation()`, providing one real,
  nonnegative, unit-sum physical-state participation vector per family in
  original state order, with direct access to its canonical `ModalFamily`.
- Family participation is the arithmetic mean of the members' existing
  unit-sum participation magnitudes. A real singleton is therefore unchanged,
  while a conjugate pair contributes symmetrically without recomputing modal
  quantities. Every dominant state tied under `rtol=1e-7`, `atol=1e-12` is
  retained in ascending state-index order.
- `ModalFamilyDynamicSummary` and
  `StateSpace.modal_family_dynamic_summaries()`, exposing each canonical
  family's oscillatory status, representative real part, existing frequency,
  damping, period, and signed time-constant quantities, plus a minimal
  mathematical stability value.
- Shared conjugate-member dynamic properties are validated with `rtol=1e-7`
  and `atol=1e-10`, then represented by their arithmetic mean. Mixed `None`
  and defined values, nonfinite values, or inconsistent paired values raise a
  clear error rather than selecting one member.
- Family stability is `decaying` for real part below `-1e-10`, `growing` above
  `1e-10`, and `neutral` within the inclusive tolerance band. Real families
  retain the existing `None` convention for undefined oscillatory quantities.
- `ModalFamilyInputInfluence` and
  `StateSpace.modal_family_input_influence()`, providing one real, finite,
  nonnegative influence magnitude per physical input in original input order,
  with direct access to the canonical family.
- Family input influence is
  `mean(abs(G_modal[i, j]) for i in family)` with no additional normalization.
  Dominant inputs include every maximum tied under `rtol=1e-7`, `atol=1e-12`
  in ascending input-index order. If all influences are zero within
  `atol=1e-12`, the dominant-input tuple is empty.
- `ModalFamilyOutputInfluence` and
  `StateSpace.modal_family_output_influence()`, providing one real, finite,
  nonnegative influence magnitude per physical output in original output order,
  with direct access to the canonical family.
- Family output influence is
  `mean(abs(H_modal[k, i]) for i in family)` with no additional normalization.
  Dominant outputs include every maximum tied under `rtol=1e-7`, `atol=1e-12`
  in ascending output-index order. If all influences are zero within
  `atol=1e-12`, the dominant-output tuple is empty.
- `ModalFamilyCharacterization` and
  `StateSpace.modal_family_characterizations()`, composing the existing family
  dynamic, state-participation, input-influence, and output-influence summaries
  without recalculating their numerical values.
- Consolidation preserves the dynamic-summary family order as canonical,
  validates equal component counts and full family membership consistency, and
  rebinds every immutable component to the exact same canonical `ModalFamily`
  object. Missing, duplicated, reordered, or mismatched family data raises a
  clear error.
- `AircraftModalFamilyCharacterization` and consistent
  `LongitudinalModel.modal_family_characterizations()` and
  `LateralDirectionalModel.modal_family_characterizations()` APIs, retaining
  each underlying generic characterization by identity while attaching the
  model's complete ordered state labels and dominant-state labels.
- Longitudinal state labels are `("u", "w", "q", "theta")` for indices
  `(0, 1, 2, 3)`; lateral-directional labels are `("v", "p", "r", "phi")`.
  Participation values remain unchanged and are also available paired with
  labels in model state order.
- Dominant-state ties preserve every label in the existing ascending index
  order. Participation-vector dimension and every dominant index are validated;
  impossible model/characterization mismatches raise a clear error.
- `AircraftModalFamilyCharacterization` also exposes complete ordered input and
  output labels, dominant input and output labels, and label/value pairs while
  retaining all existing state-label fields unchanged.
- Aircraft input labels come directly from each model's `INPUT_ORDER`, and
  output labels come directly from `OUTPUT_ORDER`: longitudinal uses
  `("delta_e",)` and `("u", "w", "q", "theta")`; lateral-directional uses
  `("delta_a", "delta_r")` and `("v", "p", "r", "phi")`.
- Dominant input/output ties retain all labels in ascending physical index
  order. Empty generic dominant-index tuples remain empty label tuples. Input
  and output influence dimensions and dominant indices are validated; no data
  is padded, truncated, normalized, or reordered.
- Consistent `LongitudinalModel.filter_modal_family_characterizations()` and
  `LateralDirectionalModel.filter_modal_family_characterizations()` APIs apply
  pure filtering to existing aircraft characterizations. `oscillatory` accepts
  `True`, `False`, or `None`; `stability` accepts `decaying`, `growing`,
  `neutral`, or `None`.
- When both filters are supplied they use logical AND. Surviving objects retain
  canonical order and object identity; no matches returns an empty tuple.
  Invalid categorical filter values raise a clear `ValueError`. No numerical
  stability or oscillation calculations are performed by the filter.
- The same filter APIs accept `dominant_state_labels`,
  `dominant_input_labels`, and `dominant_output_labels`. Each accepts one exact
  string, a nonempty iterable of exact strings, or `None`. Strings remain
  atomic, duplicate requested labels are deduplicated through order-independent
  set semantics, and labels are validated against the corresponding model order
  constants without aliases, case conversion, or fuzzy matching.
- Label matching uses ANY within each provided category and logical AND across
  state, input, output, oscillatory, and stability categories. Families with an
  empty dominant-label tuple simply fail a corresponding label filter. Results
  preserve canonical order and object identity; no matches remains an empty
  tuple, and empty requested label collections raise `ValueError`.
- Dominant-label filtering accepts `dominant_label_match` as `ANY`, `ALL`, or
  `EXACT`, with `ANY` remaining the default. The selected set semantics apply
  independently within every supplied state, input, and output label category:
  `ALL` requires every requested label to be dominant, while `EXACT` requires
  equality with the complete dominant-label set. Duplicate requested labels
  remain deduplicated, and invalid match modes raise `ValueError`.
- `dominant_state_label_match`, `dominant_input_label_match`, and
  `dominant_output_label_match` optionally override the global match mode for
  their corresponding supplied label filters. Each accepts `ANY`, `ALL`,
  `EXACT`, or `None`; `None` falls back to `dominant_label_match`. Categories
  continue to combine with logical AND, and every surviving object retains its
  canonical order and identity.
- `exclude_dominant_state_labels`, `exclude_dominant_input_labels`, and
  `exclude_dominant_output_labels` accept one exact string, a nonempty iterable
  of exact strings, or `None`. Requested labels use the same validation and
  set-deduplication rules as inclusion filters. A family is excluded when any
  requested exact label occurs in the corresponding existing dominant-label
  tuple; exclusions combine with inclusion and dynamic filters using logical
  AND without changing inclusion match modes.
- Exclusion filtering accepts `exclude_dominant_label_match` as `ANY`, `ALL`,
  or `EXACT`, with `ANY` preserving the original exclusion behavior. Optional
  `exclude_dominant_state_label_match`,
  `exclude_dominant_input_label_match`, and
  `exclude_dominant_output_label_match` values override that global mode for
  their category and fall back to it when `None`. Exclusion set predicates use
  the same exact set semantics as inclusion before negating the matched family.
- `LongitudinalModeIdentification` and
  `LongitudinalModel.physical_mode_identifications()` preserve each interpreted
  characterization by identity and attach `short_period`, `phugoid`, or `None`,
  plus immutable decision evidence.
- Only oscillatory families with finite positive natural frequency and period
  are eligible. The unique fastest and slowest eligible families must have a
  natural-frequency ratio of at least 3. The fast family additionally requires
  at least 60% combined `w`/`q` participation with all dominant labels confined
  to that set; the slow family analogously requires at least 60% combined
  `u`/`theta` participation. Missing, tied, insufficiently separated, or mixed
  evidence remains unclassified.
- The tentative fastest/slowest pair must also have finite, strictly positive,
  subcritical damping ratios, with phugoid damping below short-period damping
  and both below 1. Numerically equal, reversed, missing, nonfinite, neutral,
  growing, or
  non-oscillatory damping evidence leaves both physical names unassigned. This
  relative guard adds no aircraft-specific damping threshold.
- `LongitudinalModeEvidence` reports each family's oscillatory status, raw
  natural frequency, period, and damping ratio, frequency/period eligibility,
  unique fastest/slowest role, shared separation result, expected-state grouped
  participation, dominant-state consistency, damping validity, and pairwise
  damping-order consistency. Nullable fields distinguish facts that were not
  applicable or evaluated; no replacement evidence is invented.
- Evidence is computed in the existing classification pass and stored on the
  backward-compatible `LongitudinalModeIdentification` result. It is immutable,
  preserves the underlying characterization by identity, and does not change
  any classification threshold or outcome.
- Deterministic synthetic `LongitudinalModel` coefficients verify the complete
  production pipeline from state-space construction and eigendecomposition
  through modal properties, family state participation, dominant aircraft
  labels, physical identification, and decision evidence without monkeypatching
  characterization data.
- The clear synthetic system produces a fast `w`/`q` short-period family near
  `4.77 rad/s` and a slow `u`/`theta` phugoid family near `0.832 rad/s`, with
  consistent damping and participation evidence. A fixed derivative variant
  preserves clear frequency/state structure but produces contradictory damping
  evidence, and both families remain unclassified.
- Longitudinal classification boundary tests verify that the natural-frequency
  separation guard is inclusive at exactly 3×: ratios `2.999`, `3.0`, and
  `3.001` respectively reject, classify, and classify both otherwise-clear
  candidates.
- Longitudinal damping boundary coverage now verifies both candidate families
  require strictly subcritical positive damping: exact `0.0` and `1.0` values
  are rejected, while the nearest representable values just inside those
  bounds are accepted when all other evidence is valid.
- Damping-order boundary coverage verifies equal and reversed ratios are
  rejected, while a phugoid ratio one representable float below the
  short-period ratio is accepted. This closed an inconsistency where an
  additional numerical-closeness check overrode the documented strict
  `zeta_phugoid < zeta_short_period` rule.
- The grouped expected-state participation guard is likewise inclusive at
  exactly 60% for both physical names: values `0.599`, `0.600`, and `0.601`
  respectively reject, classify, and classify the affected candidate while
  leaving the other valid candidate unchanged. Evidence reports the tested
  values and all unchanged damping and dominant-state guards remain satisfied.
- Longitudinal physical-mode identification v1 is intentionally complete at
  the current 423-test baseline; its APIs, rules, evidence, and verified
  behavior are frozen while generic structural analysis develops.
- `LateralDirectionalModeIdentification` and
  `LateralDirectionalModel.physical_mode_identifications()` preserve each
  interpreted characterization by identity and attach `dutch_roll`,
  `roll_subsidence`, `spiral`, or `None` in canonical family order, plus
  immutable decision evidence.
- Dutch roll requires exactly one eligible oscillatory family with decaying
  stability, finite positive natural frequency and period, subcritical positive
  damping, at least 60% combined `v`/`r` participation, and all dominant labels
  confined to `v`/`r`.
- Roll subsidence and spiral require at least two finite nonneutral real
  families with unique fastest and slowest absolute real-part rates separated
  by at least 3×. Roll subsidence is the fast decaying family with at least 60%
  `p` participation and only `p` dominant. Spiral is the slow family with at
  least 60% combined `v`/`r`/`phi` participation and no dominant `p` label;
  either decaying or growing nonneutral spiral behavior is eligible.
- Multiple eligible oscillatory families, tied real rates, insufficient rate
  separation, invalid dynamics, or inconsistent state evidence remain
  unclassified rather than guessed.
- `LateralDirectionalModeEvidence` reports oscillatory status, stability,
  oscillatory and real-mode eligibility, raw frequency, period, damping ratio,
  damping validity, absolute real rate, fastest/slowest role, extreme
  uniqueness, 3× separation, expected-state grouped participation,
  dominant-state consistency, and candidate ambiguity. Nullable fields mark
  evidence that is inapplicable or was not evaluated.
- Lateral evidence is computed in the existing classification pass, is attached
  through an optional result field for practical constructor compatibility, and
  changes no Dutch-roll, roll-subsidence, spiral, or `None` outcome.
- Deterministic synthetic `LateralDirectionalModel` coefficients verify the
  complete production pipeline from state-space construction and
  eigendecomposition through modal properties, family participation, dominant
  aircraft labels, physical identification, and immutable evidence without
  monkeypatching characterization data.
- The clear synthetic lateral system produces a fast real roll-subsidence mode
  near `-7.37`, a Dutch-roll pair near `-0.996 ± 1.872j`, and a slow real spiral
  mode near `-0.302`, with strong corresponding `p`, `v`/`r`, and
  `v`/`r`/`phi` evidence. Changing only the synthetic `l_p` derivative produces
  two eligible oscillatory families, both explicitly ambiguous and unclassified.
- Lateral-directional physical-mode identification v1 is intentionally complete
  at the 406-test baseline. Its APIs, criteria, evidence, and end-to-end
  behavior are frozen alongside longitudinal v1.
- `StateSpace.controllability_matrix()` returns the standard continuous-time
  matrix `[B, A B, ..., A^(n-1) B]` with shape `(n, n*m)`.
  `controllability_rank()` reports its NumPy numerical matrix rank, and
  `is_fully_controllable()` tests whether that rank equals the state dimension.
- `StateSpace.observability_matrix()` returns the standard vertically stacked
  matrix `[C; C A; ...; C A^(n-1)]` with shape `(n*p, n)`.
  `observability_rank()` reports its NumPy numerical matrix rank, and
  `is_fully_observable()` tests whether that rank equals the state dimension.
- Full-rank and rank-deficient synthetic systems are verified. Existing
  longitudinal and lateral-directional models use the same generic APIs without
  changing their equations or modal results; their full-state output matrices
  remain fully observable.
- `StateSpace.is_minimal_realization()` returns `True` only when the existing
  `is_fully_controllable()` and `is_fully_observable()` checks both return
  `True`. It introduces no new rank calculation, tolerance, or reduction
  behavior. All four controllability/observability truth-table combinations and
  both aircraft model conversions are verified.
- `StateSpace.is_stabilizable()` applies the continuous-time PBH condition to
  every eigenvalue with nonnegative real part. Each `[lambda I - A, B]` matrix
  must have full state rank under NumPy's default SVD-based `matrix_rank`
  tolerance; strictly stable uncontrollable modes are permitted.
- Fully controllable unstable, wholly stable uncontrollable, unstable and
  neutral uncontrollable, and mixed multi-state realizations are verified.
  `StateSpace` now consistently rejects nonfinite entries in any system matrix
  after preserving its existing shape validation.
- `StateSpace.is_detectable()` applies the dual continuous-time PBH condition
  to every eigenvalue with nonnegative real part. Each vertically stacked
  `[lambda I - A; C]` matrix must have full state rank using the same NumPy
  default SVD tolerance; strictly stable unobservable modes are permitted.
- Fully observable unstable, wholly stable unobservable, unstable and neutral
  unobservable, and mixed multi-state realizations are verified. Both aircraft
  model conversions remain compatible and detectable through their existing
  full-state outputs.
- `StructuralAnalysis` is a public immutable five-boolean result containing
  exactly controllable, observable, minimal, stabilizable, and detectable.
  `StateSpace.structural_analysis()` delegates each field to its corresponding
  existing public check without introducing rank data or new mathematics.
- Fully minimal, stabilizable-but-uncontrollable,
  detectable-but-unobservable, and neither-stabilizable-nor-detectable systems
  are verified field by field. Both aircraft model conversions expose the same
  generic summary and remain consistent with their individual checks.
- `NonstablePBHDiagnostic` identifies one nonstable eigenvalue and whether it
  fails PBH controllability, PBH observability, or both.
  `StateSpace.nonstable_pbh_diagnostics()` returns only failing modes as an
  immutable tuple in existing eigenvalue order, using the same NumPy default
  SVD rank tolerance as stabilizability and detectability.
- Unstable one-sided and two-sided failures, neutral failures, omission of
  stable failures, omission of passing nonstable modes, deterministic order,
  immutability, and both aircraft model conversions are verified. Existing
  structural booleans and `StructuralAnalysis` remain unchanged.
- Dominant-label filtering is intentionally paused and complete enough for the
  current stage; all verified inclusion and exclusion APIs remain available.
- Generic trajectory-extrema analysis and synthetic longitudinal and
  lateral-directional response tests.

## Architectural constraints

- Preserve the existing eigenvalue ordering and individual
  `ModalStateCharacterization` results.
- Preserve deterministic modal-family ordering, canonical family members, and
  the documented conjugacy tolerances.
- Derive family participation exclusively from canonical family members and
  preserve its arithmetic-mean aggregation and dominant-state tie convention.
- Derive family dynamic summaries exclusively from member `ModalProperties`;
  preserve their consistency and stability tolerance conventions.
- Derive family input influence exclusively from canonical family membership
  and the existing modal input-influence matrix; preserve its scale, ordering,
  aggregation, tie, and zero-influence conventions.
- Derive family output influence exclusively from canonical family membership
  and the existing modal output-influence matrix; preserve its scale, ordering,
  aggregation, tie, and zero-influence conventions.
- Compose consolidated characterizations only from the existing verified
  family-level APIs. Preserve component values, family order, and the invariant
  that every component references the same canonical family object.
- Keep aircraft state-label interpretation outside generic `StateSpace`, retain
  underlying generic characterizations unchanged, and preserve each model's
  established state ordering and validation behavior.
- Reuse model `INPUT_ORDER` and `OUTPUT_ORDER` directly for aircraft channel
  interpretation, preserving generic influence values, ties, empty dominance,
  and validation behavior.
- Filter aircraft characterizations only through existing categorical dynamic
  fields, preserving object identity, canonical order, AND semantics, and empty
  results without adding thresholds.
- Filter dominant aircraft labels only through exact membership in the existing
  wrapper label tuples, preserving globally or independently configured
  ANY/ALL/EXACT inclusion and exclusion semantics, validation, identity,
  ordering, AND-across behavior, and empty results.
- Keep physical-mode identification in the aircraft-model layer and derive it
  only from existing immutable modal-family dynamics and state participation.
- Prefer `None` over a physical-mode guess whenever frequency or state evidence
  or damping evidence is missing, tied, insufficiently separated, nonfinite,
  or contradictory.
- Reuse `eigenvalues()`, `modal_properties()`, participation factors, and the
  existing characterization infrastructure.
- Do not duplicate eigendecomposition or eigenvector-pairing logic.
- Keep generic interpretation at the `StateSpace` layer based on physical-state
  indices; `StateSpace` does not know aircraft state names.
- Preserve all verified flight-dynamics equations, units, axes, and sign
  conventions.
- Preserve complex modal mathematics and use explicit numerical tolerances for
  conjugate matching.
- Add no dependencies and avoid unrelated refactors.
- Preserve the physical-state augmented-matrix exact propagation and its
  left-endpoint zero-order-hold input convention.
- Do not resume expanding modal numerical response helpers unless a later
  interpretation capability requires it.

## Must not be added or changed next

- Do not add other aircraft-specific mode names yet.
- Do not change longitudinal or lateral-directional physical-mode identification
  v1 without a demonstrated inconsistency.
- Do not expand dominant-label filtering during the physical-identification
  stage unless identification requires it.
- Do not add real aircraft data or controllers.
- Do not change the verified numerical modal infrastructure, flight-dynamics
  equations, or sign conventions without a demonstrated inconsistency.
- Do not add dependencies or perform unrelated refactors.

## Exact next smallest task

### SISO output-error integral augmentation

Add a generic continuous-time SISO design augmentation with one integral state
driven by constant-reference output error, establishing conventions before any
integral-gain synthesis.

## Suggested implementation direction

- Define an integral state using an explicit sign convention such as
  `xi_dot = r - y`, retain both plant input and scalar reference as documented
  augmentation inputs, and include nonzero `D` exactly.
- Return an immutable augmented design model with auditable state, input, and
  output ordering. Do not synthesize state or integral gains yet.
- Require SISO channels and validate dimensions without imposing stability.
  Do not add anti-windup, saturation, MIMO tracking, LQR/LQG, Kalman filtering,
  or aircraft tuning yet.
- Preserve the established NumPy numerical rank convention and immutable tuple
  results.
- Preserve all existing state-space and modal results unchanged.
- Preserve all dominant-label inclusion and exclusion APIs unchanged.
- Preserve longitudinal identification and evidence APIs unchanged.

## Focused tests to add

- Verify the exact augmented equations and matrices, integral-error sign,
  state/input/output ordering, nonzero feedthrough, non-mutation, and SISO
  validation, including direct trajectory-level equation checks.
- Existing aircraft and modal behavior remains unchanged.
- Ambiguous longitudinal families retain `None` rather than being guessed.

## Commands that must pass

```bash
uv run pytest -q
.venv/bin/ruff check
git diff --check
git status
```
