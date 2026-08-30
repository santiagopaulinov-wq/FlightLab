# FlightLab session checkpoint

## Current project stage

FlightLab now supports exact zero-order-hold forced propagation directly in
physical state coordinates, in addition to its existing modal numerical
infrastructure and generic modal flight-dynamics interpretation layer. The
interpretation capability characterizes how physical state indices participate
in each individual eigenmode and groups real modes and complex-conjugate pairs
into generic modal families with family-level state-participation summaries.
The active development stage is now conservative longitudinal physical-mode
identification: clear short-period and phugoid families can be named from both
relative frequency scale and longitudinal state participation, while ambiguous
families remain unclassified. Each family also has a generic dynamic summary
derived from its canonical member
properties and generic physical-input and physical-output influence summaries.
These verified summaries are available together through one consolidated,
immutable generic characterization per canonical family.
The aircraft-model layer can interpret dominant generic state indices using
each model's established physical state labels without changing generic modal
results, and likewise interprets dominant input and output indices using the
models' established channel ordering.
Both aircraft models can filter these immutable interpreted families by their
existing oscillatory status, mathematical stability, and exact dominant
state/input/output labels, with configurable global or per-category ANY, ALL,
or EXACT set matching for both inclusions and exclusions.

## Checkpoint commit

- Message: `feat: consolidate aircraft modal characterization`
- Resolve the checkpoint hash from `git rev-parse HEAD` after reading this file.

## Current verification baseline

- Test count: 390 tests.
- `uv run pytest -q` passes.
- `.venv/bin/ruff check` passes.
- `git diff --check` passes.

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
  characterization by identity and attach `short_period`, `phugoid`, or `None`.
- Only oscillatory families with finite positive natural frequency and period
  are eligible. The unique fastest and slowest eligible families must have a
  natural-frequency ratio of at least 3. The fast family additionally requires
  at least 60% combined `w`/`q` participation with all dominant labels confined
  to that set; the slow family analogously requires at least 60% combined
  `u`/`theta` participation. Missing, tied, insufficiently separated, or mixed
  evidence remains unclassified.
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
  is missing, tied, insufficiently separated, or contradictory.
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

- Do not classify Dutch roll, roll subsidence, or spiral modes.
- Do not add other aircraft-specific mode names yet.
- Do not expand dominant-label filtering during the physical-identification
  stage unless identification requires it.
- Do not add real aircraft data or controllers.
- Do not change the verified numerical modal infrastructure, flight-dynamics
  equations, or sign conventions without a demonstrated inconsistency.
- Do not add dependencies or perform unrelated refactors.

## Exact next smallest task

### Longitudinal damping-evidence guard

Add a conservative damping-evidence guard to longitudinal short-period and
phugoid identification using the existing family damping ratio. Contradictory
or insufficient damping evidence must remove a tentative name rather than force
classification. Do not classify lateral-directional modes yet.

## Suggested implementation direction

- Use only the existing immutable `ModalFamilyDynamicSummary.damping_ratio`.
- Choose and document a minimal conservative rule before implementation.
- Preserve the current frequency-separation and state-participation guards.
- Preserve canonical ordering and object identity in every filtered result.
- Preserve all dominant-label inclusion and exclusion APIs unchanged.
- Add no additional aircraft-mode names or unrelated heuristics.

## Focused tests to add

- Clear synthetic short-period and phugoid cases retain their names when damping
  evidence agrees.
- A frequency/state candidate with contradictory or missing damping evidence
  remains unclassified.
- Lateral-directional families remain free of physical-mode classification.
- Ambiguous longitudinal families retain `None` rather than being guessed.

## Commands that must pass

```bash
uv run pytest -q
.venv/bin/ruff check
git diff --check
git status
```
