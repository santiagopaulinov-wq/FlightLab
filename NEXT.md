# FlightLab session checkpoint

## Current project stage

FlightLab now supports exact zero-order-hold forced propagation directly in
physical state coordinates, in addition to its existing modal numerical
infrastructure and generic modal flight-dynamics interpretation layer. The
interpretation capability characterizes how physical state indices participate
in each individual eigenmode and groups real modes and complex-conjugate pairs
into generic modal families with family-level state-participation summaries.
Longitudinal and lateral-directional physical-mode identification v1 are both
complete at their verified conservative baselines. FlightLab has now
deliberately begun the Experimental Platform phase. Its first layer is a
generic, reproducible SISO response-result abstraction that evaluates sampled
output and reference trajectories without depending on `StateSpace`, an
aircraft model, or a controller implementation. The second layer now records
one completed computational experiment as immutable simulation provenance plus
those existing metrics, without executing the experiment. The third layer now
persists those deterministic reproducibility records in SQLite through a
generic store with explicit connection and transaction ownership. The fourth
layer executes exactly one caller-supplied generic SISO simulation, evaluates
it through the existing metrics API, and returns one validated immutable run.
The fifth layer executes a finite ordered collection of explicit experiment
cases sequentially through that same single-run boundary. The sixth layer maps
one finite ordered collection of explicit parameter values through a caller-
supplied factory into immutable experiment cases without executing them. The
seventh layer expands multiple explicit ordered axes into cases using
deterministic standard Cartesian-product order, also without execution. The
eighth layer composes that expansion with existing sequential execution to
produce immutable ordered campaign runs. The ninth layer composes explicit
sequential case execution with optional atomic persistence and returns one
minimal immutable completed-campaign result. The tenth layer assigns explicit
campaign identity and atomically persists a campaign manifest, its newly
completed runs, and their exact ordered membership. The eleventh layer
retrieves one persisted campaign as an immutable manifest plus its detached
reproducibility records in exact membership order. The twelfth layer converts
that bundle into a fresh deterministic JSON-compatible plain record without
performing I/O. The thirteenth layer extracts one explicit provenance parameter
and caller-selected existing response metrics into immutable campaign-ordered
comparison entries. The fourteenth layer transforms those entries into signed
absolute parameter and metric deltas from one explicit baseline run. The
fifteenth layer converts those deltas into immutable baseline-relative secant
slopes while representing every exact zero denominator as unavailable. The
sixteenth layer assembles explicit one-at-a-time representative secants into an
immutable response-metric-row by varied-parameter-column sensitivity matrix.
The seventeenth layer applies that matrix to one explicit aligned parameter-
change vector to produce immutable linear predicted metric changes.
The eighteenth layer applies the same matrix to a finite explicit ordered
collection of named change scenarios while retaining one immutable projection
per scenario. The nineteenth layer reduces those existing scenario projections
to deterministic per-metric finite extrema and first-attaining scenario names.
The twentieth layer checks those envelopes against explicit allowable metric-
change bounds and reports ordered deterministic margins and pass/fail results.
The twenty-first layer reduces those existing per-metric checks to one immutable
overall verdict with ordered passing, failing, and undefined metric categories.
The twenty-second layer compares one explicit named secant-matrix projection
with one caller-selected observed campaign delta and returns immutable ordered
per-metric signed residuals without recomputing either source.
The twenty-third layer checks those existing residuals against explicit
caller-ordered per-metric maximum absolute tolerances while preserving scenario
and observed-run traceability.
The twenty-fourth layer evaluates a finite explicit ordered validation set by
delegating every named scenario/observation case through the existing residual
and tolerance-check APIs.
The twenty-fifth layer reduces that validated set to one immutable deterministic
verdict with ordered passing, failing, and undefined case classifications.
The twenty-sixth layer identifies each metric's worst defined absolute residual
and first-attaining case identity across that same explicit validation set.
The twenty-seventh layer reports immutable per-metric counts, signed extrema,
signed means, and absolute-error summaries over the validated deterministic
residuals.
The twenty-eighth layer compares two explicitly identified aligned error-summary
collections through immutable right-minus-left descriptive differences. The
twenty-ninth layer applies that existing comparison to one explicit baseline
and a finite caller-ordered set of explicitly named summary collections. The
thirtieth layer reduces the stored comparison-set differences to immutable
per-metric finite extrema with first-attaining comparison identities. The
thirty-first layer checks those existing extrema against explicit aligned
allowable difference intervals and reports deterministic margins and pass/fail
states. The thirty-second layer reduces those checked fields to one immutable
overall verdict with ordered passing, failing, and undefined metric/field
identities. The thirty-third layer localizes the same validated field results
into immutable ordered per-metric verdicts. The thirty-fourth layer assembles
the checked fields and both verdict views into one immutable, traceable,
consistency-validated analytical report. The thirty-fifth layer converts that
report to a fresh deterministic JSON-compatible plain record without I/O.
The thirty-sixth layer applies that converter to a finite caller-ordered
collection of uniquely named assessment reports while preserving each name and
detached record. The thirty-seventh layer extracts their stored overall and
ordered per-metric pass states into a compact deterministic plain overview. The
thirty-eighth layer reduces the same validated named reports to one immutable
collection verdict with ordered passing, failing, and undefined report names.
The thirty-ninth layer assembles those detached named reports and that existing
verdict into one immutable, traceable, consistency-validated collection report.
The fortieth layer converts that report to a fresh deterministic JSON-compatible
plain record by reusing the existing assessment-record conventions. The
forty-first layer applies that converter to a finite caller-ordered collection
of uniquely named assessment collection reports. The forty-second layer
extracts their stored overall collection pass states into a compact ordered
deterministic overview. The forty-third layer reduces those validated named
collection reports to one immutable verdict with ordered passing, failing, and
undefined collection names. The forty-fourth layer converts that verdict to a
fresh deterministic JSON-compatible plain record without recomputing its
classifications. The forty-fifth layer applies that converter to a finite
caller-ordered collection of explicitly named assessment-collection verdicts.
The forty-sixth layer extracts their stored overall pass states into a compact
ordered deterministic overview. The forty-seventh layer reduces those
validated named verdicts to one immutable aggregate verdict with ordered
passing, failing, and undefined verdict names. The forty-eighth layer converts
that aggregate verdict to a fresh deterministic JSON-compatible plain record.
The forty-ninth layer applies the converter to a finite caller-ordered
collection of explicitly named aggregate verdicts, completing the current
aggregate-verdict record family. FlightLab has now crossed the explicit
Verification & Validation phase boundary. Its first V&V capability is one
fixed, independent closed-form verification benchmark for the existing
continuous-time linear `StateSpace` eigenvalue and physical-coordinate exact-
propagation foundation. It returns deterministic evidence through the existing
`ExperimentRun` record machinery. This is software verification against a
mathematical oracle, not physical validation of an aircraft model. The second
V&V capability is now complete: one fixed SciPy cross-check of a different
coupled oscillatory system, using SciPy only as a development/test verification
dependency and reusing the same existing evidence machinery.
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
prefilter calculation. SISO plants can also be augmented with one explicit
output-error integral state as an open controller-design model, and that model
can now be closed with caller-supplied state and integral gains while retaining
the scalar reference input and physical plant output. Explicit sets of
`n_states + 1` poles can now also be placed for that integral interconnection
through the existing NumPy-only SISO Ackermann implementation.
Both aircraft models can filter these immutable interpreted families by their
existing oscillatory status, mathematical stability, and exact dominant
state/input/output labels, with configurable global or per-category ANY, ALL,
or EXACT set matching for both inclusions and exclusions.

## Current checkpoint

- Completed capability: one fixed coupled damped-oscillator comparison of
  FlightLab eigenvalues and exact sampled zero-order-hold propagation with
  `scipy.linalg.eigvals()` and `scipy.signal.lsim(..., interp=False)`.
- Completed capability commit: this checkpoint's implementation commit
  (`feat: add scipy state-space verification benchmark`).

## Current verification baseline

- Test count: 1479 tests.
- `uv run pytest -q` passes.
- `.venv/bin/ruff check` passes.
- `git diff --check` passes.

## Experimental Platform foundation

- `flightlab.response.response_metrics(time, y, reference)` returns an
  immutable `SISOResponseMetrics` containing read-only copies of the time,
  output, reference, and tracking-error trajectories plus scalar metrics.
- Tracking error is `e = reference - y`. Final output, final reference, and
  steady-state error are the last sampled `y`, `reference`, and `e` values.
  Peak output is the largest signed output sample, and maximum absolute
  tracking error is `max(abs(e))`.
- IAE is `trapezoid(abs(e), time)`, ISE is
  `trapezoid(e**2, time)`, and the time-weighted RMS error is
  `sqrt(ISE / (time[-1] - time[0]))`; irregular time spacing is therefore
  included directly.
- Overshoot uses the final reference `r_f` as a signed target and reports
  `100 * max(0, max(sign(r_f) * y) - abs(r_f)) / abs(r_f)`. A valid response
  with no overshoot returns `0.0`.
- Settling time uses an inclusive caller-configurable relative band, defaulting
  to 2% of `abs(r_f)`, around the final reference. It is the earliest sampled
  time whose complete remaining output suffix stays within that band; there is
  no interpolation or first-entry shortcut.
- When `abs(r_f) <= 100 * machine epsilon`, percentage overshoot and the
  relative settling band have no generic unit-independent meaning, so both
  results are explicitly `None`. Settling time is also `None` when the sampled
  response never remains within its band.
- Time, output, and reference must be finite real one-dimensional arrays with
  matching lengths. Time requires at least two strictly increasing samples.
  The settling tolerance must be a finite positive real scalar. Finite inputs
  whose arithmetic would overflow to nonfinite metrics are rejected clearly.
- This first Experimental Platform capability is generic and NumPy-only. It
  performs no experiment execution, persistence, SQL, sweep execution, ML,
  controller, observer, or aircraft-specific scoring behavior itself.
- `flightlab.experiment.experiment_run(...)` returns a frozen, slotted
  `ExperimentRun` for one already-completed response. It composes the existing
  `SISOResponseMetrics` by identity and performs no metric recalculation or
  simulation execution.
- Each run records a stable string ID, aware UTC creation timestamp, simulation
  method, derived start/end/duration/sample-count values, a defensive read-only
  initial-state copy, system/controller/reference/user metadata, and metrics.
  The supplied validated time grid must exactly equal `metrics.time`, preventing
  contradictory timing provenance without storing a duplicate time vector.
- Automatic IDs are canonical UUID4 strings. Explicit IDs may be any nonblank
  caller-supplied string. Automatic timestamps use the current aware UTC time;
  explicit aware datetimes are normalized to UTC, and naive datetimes are
  rejected.
- Each metadata category is defensively copied, sorted lexically by string key,
  and exposed through a read-only mapping. Values are restricted to `None`,
  booleans, integers, finite floats, strings, or one-level tuples of those
  scalar values. NumPy scalar equivalents are normalized to Python scalars;
  mutable, nested, nonfinite, and opaque values are rejected.
- `ExperimentRun.reproducibility_record()` returns a new deterministic,
  JSON-compatible dictionary on every call. It contains run identity, an ISO
  UTC timestamp, timing data, a list-valued initial state, sorted plain metadata
  dictionaries, and all eleven scalar response metrics. Trajectory arrays are
  deliberately omitted, and mutating a returned record cannot affect the run.
- `SQLiteExperimentStore` owns one standard-library `sqlite3` connection for
  its lifetime, initializes one table and its listing index idempotently, and
  supports file-backed paths and isolated `:memory:` databases.
- `save()` accepts only `ExperimentRun`, consumes its validated deterministic
  reproducibility record, and commits one parameterized `INSERT` atomically.
  Duplicate IDs raise `DuplicateRunIDError` without overwriting the original.
- `save_many()` snapshots and validates one finite ordered run collection, then
  persists every record in caller order through the same insertion path and
  one transaction. Any failure rolls back the complete collection.
- `get()` returns a detached plain reproducibility record or `None` for an
  unknown ID. `list_runs()` returns frozen lightweight summaries ordered by
  UTC creation timestamp newest-first and then run ID ascending.
- Initial state and metadata use compact, sorted-key, Unicode-preserving JSON
  with nonfinite values disabled. All eleven response metrics are typed table
  columns; the two optional metrics preserve SQL `NULL` as Python `None`.
- This third layer adds no experiment execution, sweeps, multiprocessing,
  dataset export, ML, observer, controller, or aircraft-specific persistence
  behavior.
- `SISOSimulationResult` is an immutable three-field return contract containing
  sampled time, one-dimensional SISO output, and sampled reference trajectory.
- `execute_experiment()` invokes one caller-supplied zero-argument callable
  exactly once, requires that result contract, delegates evaluation to
  `response_metrics()`, and delegates provenance construction and validation to
  `experiment_run()`.
- Initial state, method, system, controller, descriptive reference, user
  metadata, optional settling tolerance, run ID, and creation timestamp remain
  explicit. Sampled reference data and descriptive reference metadata remain
  intentionally distinct.
- Invalid sampled results retain the existing metrics errors, invalid
  provenance retains the existing experiment-run errors, and exceptions from
  the caller's simulation propagate unchanged.
- This fourth layer performs no persistence or automatic save, retry, batch,
  sweep, parallel execution, optimization, controller synthesis, CLI workflow,
  or aircraft-specific behavior.
- `ExperimentCase` is a frozen, slotted, keyword-only, identity-based shallow
  container with every required and optional argument for one explicit
  `execute_experiment()` call.
- `execute_experiments()` snapshots one finite ordered iterable of cases,
  verifies every element before any simulation starts, delegates each case to
  `execute_experiment()` exactly once and sequentially in caller order, and
  returns the runs as an immutable tuple. Empty input returns `()`.
- Each returned run retains the existing defensive copies and deterministic
  reproducibility semantics. Case payloads remain caller-owned until their
  delegated execution so validation and copying are not duplicated or moved.
- Batch-specific malformed inputs are rejected before execution. During
  execution, the first existing validation or caller simulation exception
  propagates unchanged; earlier cases have completed, later cases do not run,
  and no partial tuple, rollback, or retry is provided.
- This fifth layer performs no case generation, parameter grids, Cartesian
  products, persistence orchestration, automatic save, concurrency,
  optimization, CLI workflow, controller synthesis, or aircraft-specific
  behavior.
- `expand_experiment_cases()` snapshots one finite ordered iterable of explicit
  parameter values and invokes one caller-supplied factory exactly once per
  value in caller order.
- Every factory result must be an `ExperimentCase`; invalid results identify
  their parameter index. Empty input returns `()`, factory exceptions propagate
  unchanged, and the result is an immutable ordered tuple.
- Parameter values remain uninterpreted and caller-owned. Expansion invokes no
  generated simulation and performs no experiment execution, Cartesian-product
  generation, persistence, automatic save, concurrency, or optimization.
- `expand_cartesian_experiment_cases()` snapshots finite explicit axes and uses
  standard-library Cartesian-product ordering: axis and value order are
  preserved and the rightmost axis varies fastest.
- The caller factory receives each combination as a tuple and is invoked
  exactly once per combination through `expand_experiment_cases()`, retaining
  its indexed result validation and exception behavior.
- Zero axes produce the single empty combination; any empty axis produces no
  combinations. Results are immutable ordered tuples of `ExperimentCase`
  objects, and generated simulations are never invoked.
- Cartesian expansion performs no experiment execution, persistence,
  automatic save, concurrency, retry, optimization, or aircraft-specific
  interpretation.
- `execute_cartesian_experiments()` is exactly the composition of
  `expand_cartesian_experiment_cases()` and `execute_experiments()`; it adds no
  combination, validation, metric, provenance, or execution path.
- All cases are generated and validated before execution. Factories and
  successful simulations run exactly once, runs preserve Cartesian order, zero
  axes execute one empty combination, and an empty axis returns `()`.
- Factory failures prevent all simulation execution. During execution, the
  first validation or simulation failure propagates unchanged after earlier
  cases complete and before later simulations run, with no retry or rollback.
- Sequential Cartesian execution performs no persistence orchestration,
  automatic save, parallel work, optimization, CLI workflow, controller
  synthesis, or aircraft-specific behavior.
- `ExperimentCampaignResult` is a frozen, slotted wrapper around one immutable
  ordered tuple of completed `ExperimentRun` objects.
- `run_experiment_campaign()` delegates explicit finite case execution to
  `execute_experiments()`. With an optional initialized
  `SQLiteExperimentStore`, it calls `save_campaign()` only after every
  experiment succeeds and returns the campaign result only after persistence
  succeeds.
- Execution failures cause no campaign persistence. Persistence errors
  propagate after execution and roll back the complete campaign transaction.
- Campaign orchestration adds no simulation, metric, provenance, serialization,
  validation, transaction, generation, retry, parallel, optimization,
  statistical-analysis, distributed, or CLI/UI implementation of its own.
- Each `ExperimentCampaignResult` has a nonblank stable campaign ID, an aware
  UTC creation timestamp, and its existing immutable ordered run tuple.
- `SQLiteExperimentStore.save_campaign()` validates every run through the
  existing reproducibility-record path and writes all runs, the campaign row,
  and zero-based positional membership rows in one transaction.
- `ExperimentCampaignManifest` is a frozen detached retrieval representation
  containing campaign ID, ISO UTC creation timestamp, and ordered run IDs.
  `get_campaign()` returns a new manifest or `None` without reconstructing full
  experiment runs.
- Duplicate campaign IDs raise `DuplicateCampaignIDError`. Any run, manifest,
  or membership failure rolls back all new rows while preserving existing data
  and store usability.
- `ExperimentCampaignBundle` is a frozen container holding an existing detached
  campaign manifest and an immutable tuple of detached reproducibility-record
  dictionaries in exact membership order.
- `get_campaign_bundle()` composes `get_campaign()` with the existing `get()`
  record retrieval path. Unknown campaigns return `None`; missing referenced
  runs raise a clear stored-state error without making the store unusable.
- Bundle retrieval reconstructs no `ExperimentRun`, performs no writes, and
  returns fresh record dictionaries on every deterministic read.
- `campaign_bundle_record()` is a pure conversion boundary that validates an
  `ExperimentCampaignBundle` and returns an exact-key plain dictionary with
  manifest metadata and ordered run reproducibility records.
- Bundle records contain only JSON-compatible dictionaries, lists, strings,
  booleans, finite numbers, and null values. Every conversion is deeply
  detached and preserves the existing run-record representation without metric
  recomputation or normalization duplication.
- `compare_campaign_runs()` validates one deterministic campaign bundle record,
  one explicit provenance category/key, and one nonempty duplicate-free ordered
  selection of existing response metric names.
- Each frozen `CampaignComparisonEntry` contains run ID, a detached scalar
  parameter value, and immutable ordered metric-name/value pairs. Comparison
  entries preserve campaign order and optional metric values without inference,
  recomputation, sorting, ranking, aggregation, or normalization.
- `campaign_metric_deltas()` requires one explicit baseline run ID and validates
  one ordered comparison for unique IDs, finite numeric parameters, and an
  identical nonempty known-metric layout across every entry.
- Each frozen `CampaignDeltaEntry` preserves run and metric order and contains
  signed `current - baseline` parameter and metric deltas. Numeric baseline
  values produce exact zeros; an optional metric delta is `None` whenever the
  current or baseline metric is unavailable.
- `campaign_secant_sensitivities()` validates an ordered delta result and
  computes every available baseline-relative slope exactly as
  `metric_delta / parameter_delta`, without revisiting source records.
- Each frozen `CampaignSensitivityEntry` retains run ID, parameter delta, and
  ordered metric sensitivities. Exact zero parameter deltas—including the
  baseline and repeated parameter values—produce `None` sensitivities without
  epsilon tolerances; optional metric deltas also remain unavailable.
- `campaign_sensitivity_matrix()` consumes explicit named parameter columns,
  full existing secant results, and caller-selected nonzero-delta representative
  run IDs without inferring or recomputing any value.
- `CampaignSensitivityMatrix` stores parameter names, metric names,
  representative run IDs, and row-major immutable values. Rows are response
  metrics, columns are varied parameters, and each element is the selected
  existing baseline-relative secant sensitivity; unavailable values remain
  `None`.
- `project_campaign_metric_changes()` requires exactly one finite numeric
  `CampaignParameterChange` per matrix column in exact name/order alignment and
  computes each available prediction as the finite sensitivity-row/change-
  vector dot product.
- `CampaignMetricChangeProjection` retains parameter and metric order, detached
  changes, and ordered predicted metric changes. Any unavailable sensitivity in
  a row makes that metric prediction unavailable rather than treating missing
  information as zero.
- `project_campaign_scenarios()` snapshots and prevalidates a finite explicit
  collection of unique named `CampaignProjectionScenario` definitions, then
  delegates each aligned vector exactly once to the existing projection API.
- Each frozen `CampaignProjectionScenarioResult` preserves caller scenario
  order and contains its name plus the existing immutable detached projection.
  Empty input returns `()`; failures produce no partial returned collection.
- `campaign_projection_envelopes()` validates one finite ordered named scenario
  result collection and reports defined finite minimum and maximum predicted
  changes for every metric in existing metric order.
- Each frozen `CampaignMetricProjectionEnvelope` retains its metric, optional
  bounds, and first-attaining scenario names. Exact ties use scenario order;
  `None` predictions are excluded, and an all-undefined metric has no bounds or
  attaining names.
- `check_campaign_projection_envelope_limits()` requires exact ordered metric
  coverage by finite explicit `CampaignMetricProjectionLimit` intervals and
  reuses existing envelope extrema without recomputation.
- Each frozen `CampaignMetricProjectionLimitResult` contains observed extrema,
  allowable bounds, exact lower/upper margins, and a pass flag requiring both
  margins to be nonnegative. Undefined envelopes retain `None` margins and do
  not pass.
- `campaign_robustness_verdict()` validates stored extrema/bounds/margin/pass
  consistency and classifies unique metrics without recomputing any upstream
  analysis.
- `CampaignRobustnessVerdict` preserves ordered passing, failing, and undefined
  metric-name tuples. Overall pass requires a nonempty input with every metric
  defined and passing; empty input is explicitly non-passing.

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
- `StateSpace.siso_integral_augmentation()` returns an immutable
  `SISOIntegralAugmentation` containing an open design model with integral-error
  convention `xi_dot = r - y`.
- Augmented state order is `[x; xi]`, input order is `[u; r]`, and output order
  is `[y; xi]`. Matrices are `A_aug = [[A, 0], [-C, 0]]`,
  `B_aug = [[B, 0], [-D, 1]]`, `C_aug = [[C, 0], [0, 1]]`, and
  `D_aug = [[D, 0], [0, 0]]`.
- Nonzero feedthrough enters exactly through `xi_dot = r - Cx - Du`. Exactly
  one plant input and output are required, but stability, controllability, and
  observability are not. One zero integrator eigenvalue is added before any
  feedback is designed.
- The original plant is not mutated. This capability performs no gain
  synthesis, integral feedback closure, prefilter combination, anti-windup,
  saturation, observer design, or aircraft-specific behavior.
- `StateSpace.siso_integral_state_feedback(K, K_i)` returns an immutable
  `SISOIntegralStateFeedbackInterconnection` containing the augmented
  closed-loop `StateSpace`, a validated copy of `K`, and the validated scalar
  `K_i`.
- The controller convention is `u = -K x + K_i xi`, the integral-error
  convention remains `xi_dot = r - y`, augmented state order is `[x; xi]`,
  the sole external input is `r`, and the sole output is the physical plant
  output `y`.
- The exact matrices are
  `A_aug = [[A-B K, B K_i], [-C+D K, -D K_i]]`,
  `B_aug = [[0], [1]]`, `C_aug = [C-D K, D K_i]`, and `D_aug = [0]`.
  Thus nonzero plant feedthrough is retained in both the integral-state
  dynamics and the exposed output.
- No algebraic inversion or special nonzero-`D` restriction is needed because
  `u` depends only on `[x; xi]`, not directly on `y` or `r`. Exactly one plant
  input and output are required; `K` must be a finite real matrix with shape
  `(1, n_states)`, and `K_i` must be a finite real scalar.
- Stability, controllability, and observability are not required. Stable and
  deliberately unstable supplied gains are both accepted, the original plant
  is not mutated, and no pole placement, LQR, gain tuning, anti-windup,
  saturation, observer, or aircraft-specific behavior is added.
- `StateSpace.place_siso_integral_poles(desired_poles)` returns an immutable
  `SISOIntegralPolePlacement` containing finite real `K`, scalar `K_i`, the
  desired poles in caller order, and the achieved poles of the actual
  integral-feedback interconnection.
- Autonomous synthesis uses the augmented pair
  `A_i = [[A, 0], [-C, 0]]`, `B_i = [[B], [-D]]`. The `-D` row follows
  directly from `xi_dot = -C x - D u` when `r = 0`, so finite nonzero plant
  feedthrough is supported without inversion or an algebraic loop.
- The existing Ackermann API supplies `K_aug` for
  `u = -K_aug [x; xi]`. The established integral-control convention therefore
  maps exactly as `K_aug = [K, -K_i]`,
  `K = K_aug[:, :n_states]`, and `K_i = -K_aug[0, -1]`.
- Exactly `n_states + 1` desired poles are required. Existing finite-value,
  one-dimensional, conjugate-closure, real-polynomial, linear-solve, and
  nonfinite-gain validation is reused rather than reimplemented. The augmented
  pair must be fully controllable under the existing numerical-rank convention.
- The returned gains are passed through `siso_integral_state_feedback()`, and
  achieved poles are checked against the desired multiset with the existing
  conjugacy tolerances. Requested poles may be stable or unstable; no automatic
  pole selection, LQR/LQI, optimization, gain scheduling, PID, or
  aircraft-specific tuning is added.
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

- Keep Experimental Platform results generic and independent of `StateSpace`,
  aircraft models, and controller implementations where practical.
- Preserve the verified response-metric definitions, final-reference target
  convention, sampled suffix settling rule, relative near-zero behavior, input
  validation, and read-only trajectory snapshots.
- Build later experiment capabilities by composing this metrics layer rather
  than introducing a parallel trajectory-evaluation implementation.
- Preserve the immutable `ExperimentRun` field meanings, exact time/metrics
  consistency check, UUID/string identity convention, aware UTC timestamp
  convention, sorted simple metadata grammar, and detached JSON-compatible
  reproducibility-record format.
- Add persistence by consuming reproducibility records; do not make the run
  abstraction execute simulations or perform its own disk I/O.
- Keep generic single-experiment execution as a thin composition of
  `SISOSimulationResult`, `response_metrics()`, and `experiment_run()`. Do not
  duplicate sampled-response or provenance validation, interpret model-specific
  outputs, or couple execution to persistence.
- Keep sequential batch execution as ordered delegation through
  `execute_experiment()`. Preserve full case preflight, input-order snapshotting,
  immutable tuple results, empty behavior, and fail-fast partial-execution
  semantics without rollback or retry.
- Keep explicit parameter-value expansion as an ordered one-to-one mapping
  through a caller factory. Preserve input snapshotting, exactly-once calls,
  indexed result validation, immutable tuple output, and complete separation
  from experiment execution and persistence.
- Keep Cartesian case expansion as a deterministic transformation of explicit
  finite axes through `itertools.product` and `expand_experiment_cases()`.
  Preserve standard zero-axis and empty-axis semantics without executing cases.
- Keep sequential Cartesian execution as direct composition of the existing
  Cartesian expansion and batch execution APIs. Preserve generation-before-
  execution and existing fail-fast partial-execution semantics.
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
- Keep required runtime dependencies unchanged, keep SciPy development/test-
  only, and avoid unrelated refactors.
- Preserve the physical-state augmented-matrix exact propagation and its
  left-endpoint zero-order-hold input convention.
- Do not resume expanding modal numerical response helpers unless a later
  interpretation capability requires it.

## Must not be added or changed next

- Do not extend the named-record, overview, aggregate-verdict, reporting, or
  serialization hierarchy for the benchmark.
- Do not make SciPy a required FlightLab runtime dependency; it is used only by
  the explicitly invoked verification runner and its development tests.
- Do not resume the previously suggested observer-based integral output
  feedback yet.
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

### Define the first published aircraft flight-dynamics benchmark

Review the completed analytical and SciPy verification evidence, then select
and specify one authoritative published linear aircraft flight-dynamics
benchmark before implementing it. Record the exact source and edition, model
matrices or coefficients, units and conventions, FlightLab quantities under
comparison, published reference quantities, tolerances, provenance, and
acceptance semantics. Explicitly classify each proposed comparison as software
verification against a published computational result or physical validation
against measured evidence; do not conflate the two. Do not add production code,
tests, data files, dependencies, or reporting types during that definition
step.

## Selected second V&V capability

### Independent SciPy damped-oscillator state-space cross-check

The second capability answers whether FlightLab and an independently maintained
scientific-computing library agree on the poles and sampled response of one
fixed, coupled, stable oscillatory continuous-time system. It must call each
library through its public API and compare their returned values directly; it
must not use the completed analytical formula as either implementation's
reference.

`flightlab.verification.run_scipy_linear_state_space_verification_benchmark()`
now executes the comparison with locked SciPy 1.18.1. The verified baseline has
maximum absolute residuals `0.0` for the eigenvalues,
`1.4432899320127035e-15` for the physical states, and
`7.771561172376096e-16` for the output. SciPy's returned grid and both initial
states are exactly equal to their fixed inputs, so the benchmark passes without
a FlightLab core correction.

This remains **software verification**. Agreement establishes evidence about
the generic linear numerical implementation for this fixed problem. It does not
validate any aircraft equations, coefficients, physical modes, controller, or
real-world behavior.

## Fixed SciPy benchmark system

Use the following two-state, one-input, one-output continuous-time realization:

```text
A = [[ 0.0,  1.0]]    B = [[0.0]]    C = [[1.0, 0.25]]    D = [[0.1]]
    [[-4.0, -0.8]]        [[1.0]]

x(0) = [0.75, -0.25]
t    = [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875,
        1.000, 1.125, 1.250, 1.375, 1.500, 1.625, 1.750, 1.875,
        2.000]
u    = [ 0.50,  0.50,  0.50,  0.50,
        -1.00, -1.00, -1.00, -1.00, -1.00, -1.00,
         0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25]
```

Every `u[k]` is held over `[t[k], t[k + 1])`; the last input sample contributes
to the last output through `D` but advances no later state. The uniform
`0.125`-second grid is required by `scipy.signal.lsim`. The input changes at
`t = 0.5` and `t = 1.25`; both libraries must use the new sample for output at
that time and the preceding sample for propagation into that time.

This system is still easy to audit as a mass-spring-damper-form oscillator, but
it is not a restatement of the first upper-triangular benchmark. Its nonzero
off-diagonal terms produce a complex-conjugate pole pair, the sampled input
exercises multiple zero-order-hold intervals and discontinuities, `C` mixes both
states, and nonzero `D` exercises same-sample feedthrough.

## FlightLab APIs under verification

- `StateSpace.eigenvalues()` on the fixed `A` matrix.
- `StateSpace.simulate(x0, u[:, None], time, method="exact")` in physical state
  coordinates, using the explicit time-varying input array so each interval
  follows FlightLab's left-endpoint zero-order-hold convention.
- Existing `response_metrics()` and `experiment_run()` only as evidence
  composition boundaries, not as independent numerical references. The
  returned run's output is FlightLab's SISO output and its sampled reference is
  SciPy's SISO output.

## Independent SciPy reference APIs

- `scipy.linalg.eigvals(A, check_finite=True)` supplies the independent
  continuous-time eigenvalues. The ordinary eigenvalue result is unordered, so
  each library's two-member conjugate pair is sorted by ascending imaginary
  part before comparison.
- `scipy.signal.lsim((A, B, C, D), U=u, T=time, X0=x0, interp=False)` supplies
  the returned time, SISO output, and physical-state trajectory. Passing the
  matrices as a tuple avoids introducing another FlightLab or benchmark model
  implementation. `interp=False` explicitly selects zero-order-hold input
  interpolation rather than SciPy's default linear interpolation.
- Do not call `scipy.linalg.expm`, numerical integration APIs, or a second
  SciPy simulation routine to construct another reference.

The implementation is based on the SciPy 1.18 public API contracts documented
for [`scipy.linalg.eigvals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigvals.html)
and [`scipy.signal.lsim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html).

## SciPy comparison quantities and residuals

Compute exactly three scalar residuals from full-precision returned values:

1. `eigenvalue_residual = max(abs(lambda_flightlab[i] -
   lambda_scipy[i]))` after sorting both two-value arrays by ascending imaginary
   part.
2. `state_residual = max(abs(x_flightlab[k, j] - x_scipy[k, j]))` over all 17
   time samples and both physical state components.
3. `output_residual = max(abs(y_flightlab[k, 0] - y_scipy[k]))` over all 17
   output samples.

Do not round, aggregate by RMS, omit transition samples, compare only final
values, or compare one implementation indirectly through response metrics.
The existing maximum-absolute-tracking-error metric must equal the independently
computed output residual; its other response fields are reproducibility data
but are not acceptance quantities.

## SciPy comparison tolerances

- Maximum absolute eigenvalue residual: `1.0e-12`.
- Maximum absolute state residual: `1.0e-10`.
- Maximum absolute output residual: `1.0e-10`.

The matrices, states, and outputs are deliberately order-one, so absolute
limits are unambiguous and no relative near-zero rule is needed. The eigenvalue
limit retains near-machine-precision expectations for a well-scaled 2-by-2
problem. The trajectory limits allow independent LAPACK, matrix-exponential,
and interval-accumulation details across SciPy builds and platforms while still
requiring roughly ten decimal digits of agreement over every state and output
sample. Equality with a limit passes.

## SciPy dependency boundary

SciPy is a **development/test-only verification dependency**, not a required
runtime dependency and not a transitive requirement for ordinary FlightLab
users. `scipy>=1.18.0,<1.19` is in `[dependency-groups].dev`, is locked to
SciPy 1.18.1 in `uv.lock`, and is absent from `[project].dependencies`. The
narrow runner imports SciPy inside the function, so importing
`flightlab.verification` and invoking the analytical benchmark still work in a
base NumPy-only installation. Calling the SciPy runner without development
dependencies installed raises the ordinary `ModuleNotFoundError` before
constructing evidence.

The lower bound fixes the reviewed `eigvals` and `lsim` contracts. The minor-
version upper bound prevents an unreviewed API generation from silently
changing the reference; the lockfile supplies the exact resolved build for the
repository verification environment. The run metadata must include the actual
`scipy.__version__` string so evidence remains traceable when the lock is
intentionally refreshed.

## SciPy acceptance and failure semantics

The benchmark passes if and only if all of these conditions hold:

1. Both eigenvalue arrays have shape `(2,)`, contain only finite values, and
   their residual is at most `1.0e-12`.
2. FlightLab and SciPy state trajectories both have shape `(17, 2)`, contain
   only finite real values, preserve `x0` exactly at index zero, and have a
   residual at most `1.0e-10`.
3. FlightLab output has shape `(17, 1)`, SciPy output has shape `(17,)`, both
   contain only finite real values, and their residual is at most `1.0e-10`.
4. SciPy's returned time has shape `(17,)`, contains only finite real values,
   and is exactly array-equal to the fixed input time grid.
5. One existing `ExperimentRun` records the fixed matrices, state, input, grid,
   SciPy API names and version, all three residuals and limits, both exact-
   initial-state flags, and an overall Boolean equal to the conjunction above.
   Its method is `exact`, its run ID is
   `verification-linear-state-space-scipy-lsim-v1`, and its creation time is
   `2026-09-02T12:00:00+00:00`. Two calls in the same locked environment must
   produce exactly equal detached JSON-compatible
   `reproducibility_record()` dictionaries.

Wrong shapes, nonfinite or complex trajectories, a mismatched SciPy time grid,
or internally inconsistent output-residual evidence raise `ValueError` before
an `ExperimentRun` is constructed. A missing SciPy installation raises
`ModuleNotFoundError`. Exceptions raised directly by either library propagate
unchanged. Well-shaped finite evidence whose residual exceeds a limit, or whose
initial state is not exactly preserved, remains meaningful failed evidence and
returns an `ExperimentRun` with `passed = False`.

## New evidence beyond the analytical benchmark

The completed analytical benchmark independently establishes the correct answer
for one coupled real-pole system with constant forcing and a nonuniform grid.
The SciPy cross-check adds agreement with a separately maintained library on a
complex-conjugate oscillatory eigensystem and a complete forced trajectory. It
also adds multiple sampled input levels, exact behavior at input transitions,
mixed-state output, nonzero direct feedthrough, and a different uniform-grid
simulation path. This is useful diversity of implementation and exercised
behavior; it does not replace the stronger mathematical independence of the
closed-form oracle or constitute physical validation.

## SciPy cross-check explicit non-goals

- No analytical oracle for the second system and no attempt to make SciPy prove
  FlightLab correct; agreement is corroborating independent-library evidence.
- No runtime SciPy requirement, optional-extra design, python-control, alternate
  SciPy simulator, generic library adapter, benchmark registry, parameterized
  system family, randomized/property testing, or tolerance policy framework.
- No new V&V result, report, verdict, serializer, database schema, persistence
  behavior, CLI, plot, or checked-in generated evidence artifact.
- No aircraft model, aerodynamic data, flight-test comparison, physical-mode
  classification, controller, observer, campaign, sweep, optimization,
  calibration, uncertainty quantification, certification, or safety claim.
- No broader verification of eigenvectors, modal participation, transfer
  functions, structural analysis, MIMO behavior, conditioning limits, arbitrary
  time grids, or other numerical integration methods.
- No change to existing StateSpace mathematics, propagation conventions,
  Experimental Platform semantics, or the completed analytical benchmark
  unless the real SciPy comparison exposes a demonstrated core discrepancy.

## Chosen first V&V capability

### Independent analytical two-state linear state-space benchmark

The first capability answers one deliberately small evidence question: for one
stable, coupled, continuous-time SISO system, do FlightLab's reported
eigenvalues and exact zero-order-hold physical-state trajectory agree with a
closed-form mathematical solution that does not use FlightLab's eigensystem,
modal, matrix-exponential, exact-step, or simulation implementation?

`flightlab.verification.run_linear_state_space_verification_benchmark()` now
executes that benchmark and returns the existing immutable `ExperimentRun`.
The verified nominal baseline has maximum absolute residuals `0.0` for the
eigenvalues and `2.220446049250313e-16` for both the state and output
trajectories, so it passes the fixed `1.0e-12` threshold without any core
state-space change. Malformed shapes and nonfinite evidence raise `ValueError`;
well-formed numerical evidence outside a limit, or failure to preserve the
initial state exactly, returns a deterministic run with `passed = False`.

The fixed benchmark is

```text
A = [[-1,  1]]    B = [[0]]    C = [[1, 0]]    D = [[0]]
    [[ 0, -2]]        [[1]]

x(0) = [1.5, -0.5]
u(t) = 2.0
t    = [0.0, 0.125, 0.5, 1.25, 2.0]
```

The nonzero off-diagonal term makes the first state depend on the second; the
nonzero constant input exercises forced propagation; and the nonuniform grid
exercises repeated exact propagation over unequal intervals. `C` exposes the
coupled first state as the single output, and `D = 0` keeps the reference
formula transparent.

## Verification target

- `StateSpace.eigenvalues()` for the fixed continuous-time `A` matrix.
- `StateSpace.simulate(x0, u, time, method="exact")` in physical coordinates,
  including both state components and the SISO output at every fixed time.
- The left-endpoint zero-order-hold convention only for the constant input;
  because the input is constant, the benchmark does not attempt to distinguish
  alternative discontinuity conventions.
- Deterministic evidence composition through the existing `ExperimentRun` and
  `reproducibility_record()` contracts. The experiment reference trajectory is
  the analytical `y(t)` and the FlightLab trajectory is the measured output.
  Existing overshoot and settling fields may be retained by that contract but
  are not V&V acceptance quantities.

This is **software verification** of the generic linear numerical foundation:
it checks implementation output against an independently derived mathematical
answer. It is not **physical validation**. No claim is made that an aircraft
model, aerodynamic derivative, controller, mode label, or simulated trajectory
represents flight-test or other real-world evidence.

## Independent reference source and selection decision

For elapsed time `tau = t - t[0]`, direct solution of the two scalar ordinary
differential equations gives

```text
lambda = {-1, -2}
x2(tau) = 1 - 1.5 exp(-2 tau)
x1(tau) = 1 - exp(-tau) + 1.5 exp(-2 tau)
y(tau)  = x1(tau)
```

The benchmark reference must encode these scalar expressions directly using
only the standard scalar exponential function and fixed constants. It must not
derive expected values with `numpy.linalg`, a matrix exponential, numerical
integration, a second `StateSpace`, or any FlightLab propagation or modal API.
The formula follows from the characteristic polynomial
`(-1 - lambda)(-2 - lambda)` and direct integrating-factor solutions of
`x2_dot = -2 x2 + 2` and `x1_dot = -x1 + x2`.

This analytical reference is selected before SciPy. It is smaller, introduces
no dependency, is inspectable line by line, and is algorithmically independent
of FlightLab's NumPy eigensolver and augmented-matrix exponential. A SciPy-first
reference would broaden system coverage but would provide less transparent
first evidence and add a dependency while likely exercising another general
matrix-exponential algorithm. Combining both now would add a second oracle
without answering a second necessary question. SciPy/python-control
cross-checks and published aircraft flight-dynamics benchmarks remain natural
later increments after this analytical seed passes.

## Exact acceptance criteria

The benchmark passes if and only if all of these conditions hold:

1. After deterministic ascending real-value sorting, FlightLab returns exactly
   two finite eigenvalues and the maximum absolute difference from
   `[-2.0, -1.0]` is at most `1.0e-12`.
2. The physical state trajectory has shape `(5, 2)`, contains only finite real
   values, preserves the specified initial state at the first sample, and its
   maximum absolute componentwise difference from the closed-form `x1` and
   `x2` values over all five samples is at most `1.0e-12`.
3. The output trajectory has shape `(5, 1)`, contains only finite real values,
   and its maximum absolute difference from analytical `y = x1` is at most
   `1.0e-12`.
4. The returned existing `ExperimentRun` records method `exact`, the fixed
   matrices/input/time/reference identity in its supported flat metadata, the
   three maximum absolute residuals, the single `1.0e-12` tolerance, and an
   overall pass Boolean equal to the conjunction of criteria 1--3. Its existing
   maximum-absolute-tracking-error metric must equal the independently computed
   output residual up to floating-point roundoff and must be at most
   `1.0e-12`.
5. Fixed run identity and fixed aware UTC creation time make two independent
   benchmark invocations produce exactly equal, JSON-compatible detached
   `reproducibility_record()` dictionaries.

No rounded display value is used for acceptance. The full-precision residuals
and fixed threshold determine the Boolean result; equality with the threshold
passes.

## Explicit non-goals

- No physical validation, calibration, uncertainty quantification,
  certification, safety claim, or comparison with flight-test data.
- No aircraft-specific longitudinal or lateral-directional model, physical-mode
  classification, controller, observer, campaign, sweep, or optimization.
- No generic V&V framework, benchmark registry, plug-in oracle interface,
  tolerance policy, new verdict/report class, new serializer, database schema,
  persistence workflow, CLI, plotting, or generated checked-in evidence file.
- No SciPy, python-control, new dependency, alternate numerical integrator, or
  published aircraft benchmark in this first increment.
- No verification of eigenvectors, modal participation, transfer functions,
  controllability/observability, numerical conditioning, MIMO behavior,
  time-varying inputs, discontinuity handling, or broad matrix families.
- No change to production state-space mathematics, simulation semantics,
  Experimental Platform reporting, or existing tests except the focused tests
  required to implement this benchmark.

## Focused tests to add

- No implementation tests are prescribed until the published benchmark source,
  evidence classification, and comparison contract are explicitly selected.

## Commands that must pass

```bash
uv run pytest -q
.venv/bin/ruff check
git diff --check
git status
```

## Restart instruction

Continue from the latest implementation commit. Read this file and inspect the
completed analytical and SciPy benchmark evidence, then perform the exact next
smallest task: **define the first published aircraft flight-dynamics
benchmark**. Do not implement it, add dependencies, conflate computational
verification with physical validation, or extend the serialization hierarchy.
Preserve the documented scope, run the required verification commands, commit
the completed capability, and do not push. Do not touch the existing untracked
`.vscode/`.
