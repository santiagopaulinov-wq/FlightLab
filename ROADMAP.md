# FlightLab v1.0 completion roadmap

## Strategy

FlightLab is in v1.0 completion mode. The goal is a coherent, reproducible
release of the platform that already exists: linear aircraft models,
state-space dynamics and structural/modal analysis, control, simulation,
experiment provenance, SQLite persistence, ordered campaigns, campaign
performance analysis, and deterministic V&V evidence.

This roadmap is ordered. New platform scope is not required for v1.0.
`NEXT.md` remains the precise handoff for only the current implementation task.

## Current state

Complete and covered by the current test suite:

- dimensional longitudinal and lateral-directional aircraft model assembly;
- generic `StateSpace` simulation, modal and structural analysis, frequency
  response, balanced realization/reduction, and control/observer primitives;
- SISO response metrics, immutable `ExperimentRun` provenance, explicit batch
  and Cartesian case execution;
- atomic SQLite run/campaign persistence and deterministic campaign retrieval;
- campaign comparison, sensitivity, projection, validation, envelope, limit,
  and verdict primitives;
- ten fixed V&V runners, including analytical and SciPy state-space checks,
  the published NASA GTM longitudinal modal benchmark, the NASA unstable-roll
  frequency-response benchmark, and focused structural/control benchmarks,
  ending with observer-based output feedback and the separation principle.

The planned V&V v1 set is complete and frozen. The repository now includes a
documented and integration-tested representative longitudinal aircraft
campaign that connects the complete FlightLab path using existing APIs.

## v1.0 completion sequence

1. **Close the existing NASA GTM longitudinal V&V work — complete.** Preserve
   its published-model reconstruction, provenance, modal checks, and tests. Do
   not reopen or broaden it without a demonstrated discrepancy.
2. **Complete the defined separation-principle/control benchmark — complete.**
   The fixed MathWorks contract and exact algebraic oracle are implemented with
   focused tests.
3. **Freeze V&V v1 — complete.** The fixed external verification set is
   complete for v1.0. Further benchmark families require post-v1 planning
   unless needed to correct a release-blocking defect.
4. **Build one representative end-to-end workflow — complete.** The
   deterministic longitudinal pitch campaign composes `aircraft model ->
   StateSpace -> modal/structural analysis -> control -> simulation ->
   ExperimentRun -> persistence -> campaign -> performance analysis` in
   `examples/longitudinal_campaign.py`, with focused integration tests.
5. **Consolidate architecture and benchmark documentation — complete.** The
   README is the entry point, architecture and V&V boundaries have dedicated
   documents, `docs/REPRODUCIBILITY.md` records the verified procedure, and
   `NEXT.md` remains the concise active checkpoint.
6. **Prove clean-environment reproducibility — complete.** Locked installation,
   package import, the representative workflow, focused tests, and the full
   suite pass from a fresh Python 3.12 temporary environment.
7. **Perform the v1.0 release audit — complete.** API/docs agreement, metadata,
   dependencies, artifacts, installed-wheel behavior, tests, hygiene, and
   release notes were audited. The demonstrated remaining blockers are an
   owner-selected license and the final version/commit/tag sequence.
8. **Prepare v1.0.0.** Resolve audit blockers, set the release version, produce
   final release notes, verify the release build/install, and leave a clean,
   tagged-ready repository. Tagging or publishing is a separate explicit
   release action.

## Explicitly after v1.0

- AI/LLM integration and repository-brain/RAG tooling
- OpenFOAM or other CFD integration
- neural operators and PhysicsNeMo
- GPU/HPC or distributed-computing expansion
- web or frontend work
- unrelated campaign-analysis expansion
- large architectural refactors
- speculative infrastructure, registries, and abstraction layers

Release-blocking corrections to existing behavior remain in scope; expanding
these deferred areas does not.
