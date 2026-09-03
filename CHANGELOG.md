# Changelog

All notable FlightLab changes intended for release are recorded here.

## 1.0.0 — 2026-09-03

### Added

- Dimensional longitudinal and lateral-directional aircraft models with
  conservative physical-mode interpretation.
- NumPy-first continuous-time state-space simulation, modal and structural
  analysis, frequency response, balanced realization/reduction, and SISO
  controller/observer design and interconnection primitives.
- Immutable SISO response metrics and `ExperimentRun` reproducibility records.
- Deterministic ordered case expansion/execution, atomic SQLite run and campaign
  persistence, and detached campaign retrieval records.
- Deterministic campaign comparison, sensitivity, projection, residual,
  envelope, limit, and verdict analysis primitives.
- Ten fixed analytical, independent-library, published-aircraft, and
  control/structural verification runners; V&V v1 is frozen.
- A representative longitudinal aircraft-to-campaign workflow with SQLite
  persistence and campaign metric comparison.
- Architecture, verification, roadmap, and clean-environment reproducibility
  documentation.

### Release status

- FlightLab 1.0.0 is prepared under the MIT License.
- The release candidate has completed its documented verification,
  reproducibility, packaging, and release-audit checkpoints.
