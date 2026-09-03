# Verification and validation scope

## Evidence already implemented

FlightLab currently has ten deterministic benchmark runners:

1. closed-form two-state eigenvalue and exact-propagation verification;
2. an independent SciPy state-space cross-check;
3. the published NASA Generic Transport Model rigid-body longitudinal modal
   benchmark at Mach 0.8;
4. the published NASA Ames unstable-roll frequency-response benchmark;
5. the MathWorks rank-deficient controllability example;
6. the MathWorks SISO pole-placement example;
7. the SciPy continuous-Lyapunov controllability-Gramian example;
8. a MathWorks-derived balanced-truncation benchmark with exact analytical
   oracles;
9. the MathWorks observer-design and Luenberger-interconnection benchmark; and
10. the MathWorks observer-based output-feedback and separation-principle
    benchmark.

Each runner produces deterministic evidence through the existing immutable
`ExperimentRun` record path. These are computational/software verification
layers. The NASA cases compare against published computational or analytical
aircraft models; they are not flight-test validation, physical model
validation, certification evidence, or safety claims.

## V&V v1 freeze

The tenth and final planned V&V v1 layer verifies the coupled compensator
topology and pole-union property that the separate controller and observer
benchmarks do not cover. It is implemented and covered by focused tests.

V&V v1 is frozen. New benchmark families are post-v1 unless a release audit
finds a demonstrated correctness gap in existing v1 behavior.

## Verification infrastructure

- `tests/test_verification.py` contains focused runner tests, including oracle,
  provenance, failure-semantics, determinism, and dependency-boundary checks.
- The broader `tests/` suite checks the underlying model, state-space,
  experiment, persistence, campaign, and analysis contracts.
- `uv.lock` fixes the development environment; NumPy is the sole runtime
  dependency, while pytest, Ruff, and SciPy are development dependencies.
- The required repository baseline is `uv run pytest -q`,
  `.venv/bin/ruff check`, and `git diff --check`.
- The locked clean-environment procedure and latest verified counts are in
  `docs/REPRODUCIBILITY.md`.
