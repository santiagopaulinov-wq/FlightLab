# FlightLab architecture

## Purpose and boundary

FlightLab is a NumPy-first Python library for reproducible linear flight-control
experiments. It connects dimensional aircraft models to continuous-time
state-space analysis and control, simulation, immutable experiment evidence,
SQLite persistence, ordered campaigns, and deterministic campaign performance
analysis. SciPy is a development-only dependency used by one independent V&V
runner.

The architecture favors explicit, immutable data and small composable
functions. Aircraft models produce generic `StateSpace` objects; downstream
experiment and persistence layers do not depend on a particular aircraft or
controller implementation.

## Implemented layers

| Layer | Primary module | Responsibility |
| --- | --- | --- |
| Aircraft models | `longitudinal.py`, `lateral_directional.py` | Validate dimensional derivatives, construct `StateSpace`, and conservatively interpret physical modes. |
| Linear dynamics | `state_space.py` | State-space realization, simulation, modal/structural analysis, frequency response, balancing, controller and observer primitives. |
| Response evaluation | `response.py` | Deterministic sampled SISO tracking metrics. |
| Experiment evidence | `experiment.py` | Execute caller-supplied simulations and create immutable `ExperimentRun` provenance; expand and execute explicit ordered cases. |
| Persistence | `persistence.py` | Atomically store and retrieve runs, campaign manifests, membership, and detached records in SQLite. |
| Campaign orchestration | `campaign.py` | Sequentially execute explicit cases and optionally persist a complete campaign atomically. |
| Performance analysis | `analysis.py` | Pure campaign comparisons, deltas, sensitivities, projections, residual checks, envelopes, and verdict records. |
| Verification | `verification.py` | Fixed deterministic benchmarks against analytical, published, or independent-library oracles, carried as `ExperimentRun` evidence. |

The intended v1.0 integration path is:

```text
aircraft model
  -> StateSpace
  -> modal/structural analysis
  -> control
  -> simulation
  -> ExperimentRun
  -> SQLite persistence
  -> campaign
  -> performance analysis
```

Each arrow has supporting APIs and isolated tests. The representative executable
composition in `examples/longitudinal_campaign.py` joins the complete path for
the longitudinal elevator-to-pitch case without adding a workflow framework.

## Design constraints

- Keep aircraft-specific interpretation above the generic `StateSpace` layer.
- Preserve caller-visible state, input, output, case, and campaign ordering.
- Keep simulation caller-supplied at the experiment boundary; do not introduce
  a hidden controller or aircraft registry.
- Reuse `ExperimentRun.reproducibility_record()` as the evidence and storage
  boundary rather than inventing parallel result schemas.
- Keep persistence transaction ownership explicit and SQLite-local.
- Keep analysis deterministic and based on stored metrics/provenance; it does
  not rerun simulations or infer missing values.
- Keep fixed V&V runners focused on one public API boundary and independent
  oracles. A passing computational benchmark is not physical validation or a
  certification claim.

See `docs/VERIFICATION.md` for the current evidence set and `ROADMAP.md` for
the ordered v1.0 completion plan. See `docs/REPRODUCIBILITY.md` for the exact
locked clean-environment procedure.

## Public module boundaries

The package root intentionally contains no convenience re-exports. Import
public objects from the module that owns their contract:

- aircraft models and interpretation: `flightlab.longitudinal`,
  `flightlab.lateral_directional`, and `flightlab.aircraft_modal`;
- dynamics and control: `flightlab.state_space`;
- response and experiment records: `flightlab.response` and
  `flightlab.experiment`;
- storage and campaigns: `flightlab.persistence` and `flightlab.campaign`;
- campaign performance analysis: `flightlab.analysis`;
- frozen external-oracle runners: `flightlab.verification`.

`examples/longitudinal_campaign.py` is executable composition code, not a new
library layer or public workflow framework.
