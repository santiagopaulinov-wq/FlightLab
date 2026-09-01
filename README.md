# FlightLab

## Full-order balanced coordinates

For an asymptotically stable minimal continuous-time `StateSpace`,
`system.balanced_realization()` returns a `BalancedRealization` containing a
full-order balanced `StateSpace` and a real state transformation `T`. The
coordinate convention is

```text
x = T z
A_bal = T^-1 A T
B_bal = T^-1 B
C_bal = C T
D_bal = D
```

Use `np.linalg.solve(T, x)` to map an original state `x` to its balanced state
`z`. No state is removed or truncated. The balanced controllability and
observability Gramians equal `diag(system.hankel_singular_values())` up to
floating-point roundoff.

The implementation is NumPy-only. It uses Cholesky factors of the existing
infinite-horizon Gramians and an SVD of their cross product. Nonstable and
nonminimal systems are rejected. It also rejects nonfinite, non-positive-
definite, or numerically singular factorization results using an explicit
machine-epsilon-scaled singular-value threshold and inverse-consistency check.

## Explicit balanced truncation

`system.balanced_truncation(r)` performs opt-in model reduction by retaining
exactly the first `r` balanced coordinates, where `r` is an integer satisfying
`1 <= r < system.n_states`. Order `n` is deliberately excluded; use
`balanced_realization()` when no states should be removed. No order is selected
automatically.

The returned immutable `BalancedTruncation` contains the reduced `system`, its
`retained_order`, the full `balanced_transformation`, the leading
`retained_hankel_singular_values`, and two state maps:

```text
x_reduced       = projection @ x_original
x_approximately = reconstruction @ x_reduced
```

For the full-order convention `x_original = T @ z`, `projection` is the first
`r` rows of `T^-1`, and `reconstruction` is the first `r` columns of `T`.
Reconstruction sets discarded balanced coordinates to zero, so the reduced
model and reconstructed state are approximations. The reduced matrices are the
leading balanced blocks, and `D` is preserved exactly. Stability, minimality,
and numerical-factorization requirements are inherited from
`balanced_realization()`.

Each result also records `discarded_hankel_singular_values` and the diagnostic
`a_priori_error_bound`:

```text
||G - G_r||_inf <= a_priori_error_bound
a_priori_error_bound = 2 * sum(discarded_hankel_singular_values)
```

This is the classical continuous-time balanced-truncation upper bound for the
input-output induced H-infinity norm error of the stable minimal original and
its explicit-order reduced model. It is not generally an equality, does not
estimate an error from sampled responses, and is not a state-reconstruction
error bound. It is diagnostic only and does not select a retained order.

## Continuous-time frequency response

`system.frequency_response(angular_frequencies)` evaluates the NumPy-only
transfer matrix

```text
G(j omega) = C @ solve(j omega I - A, B) + D
```

Frequencies are finite real angular frequencies in rad/s. A scalar returns a
complex array with shape `(n_outputs, n_inputs)`. A nonempty one-dimensional
frequency array returns shape
`(n_frequencies, n_outputs, n_inputs)` in the supplied order. Valid zero-input
and zero-output dimensions are preserved, and `D` is included exactly.

The system need not be stable. Evaluation is valid wherever
`j omega I - A` is nonsingular. A requested frequency exactly at a pole raises
`ValueError`; nonfinite, complex, empty-vector, and higher-dimensional inputs
are rejected clearly. The method uses linear solves, selects no grid, and does
not estimate an H-infinity norm.

## Transfer-matrix singular values

`system.frequency_response_singular_values(angular_frequencies)` applies
`np.linalg.svd(..., compute_uv=False)` to each transfer matrix returned by
`frequency_response()`. Frequencies use the same rad/s units, scalar/vector
forms, ordering, validation, and exact-pole behavior.

For `k = min(n_outputs, n_inputs)`, a scalar frequency returns a real
nonnegative array with shape `(k,)`; a frequency vector returns
`(n_frequencies, k)`. Values are descending at each frequency and repeated
values retain their multiplicity. Empty input or output channels produce
`(0,)` or `(n_frequencies, 0)` as appropriate.

These values describe directional gains of `G(j omega)` only at the explicitly
requested frequencies. The method selects no grid, performs no maximization,
and does not estimate an H-infinity norm.

## Transfer-matrix singular directions

`system.frequency_response_singular_directions(angular_frequencies)` returns
an immutable `FrequencyResponseSingularDirections` with reduced-SVD singular
values, left directions `U`, and right directions `V`:

```text
G(j omega) = U @ diag(singular_values) @ V.conj().T
```

For `p` outputs, `m` inputs, and `k = min(p, m)`, scalar shapes are `(k,)`,
`(p, k)`, and `(m, k)`. A vector of `f` frequencies returns `(f, k)`,
`(f, p, k)`, and `(f, m, k)`. Empty channels use the same conventions with
`k = 0`. Rows of `U` follow output-channel order; rows of `V` follow
input-channel order. Frequencies remain in rad/s and inherit all validation and
pole behavior from `frequency_response()`.

Singular directions are not unique. Each paired direction may differ by an
arbitrary unit-magnitude complex phase, and bases within repeated-singular-value
subspaces may rotate. No phase normalization is imposed. The API selects no
grid and performs no maximization or H-infinity estimation.

## Balanced-truncation frequency-error samples

`system.balanced_truncation_frequency_response_error(r, angular_frequencies)`
returns local transfer-matrix samples

```text
E(j omega) = G(j omega) - G_r(j omega)
```

where `G_r` is the realization returned by `system.balanced_truncation(r)`.
Frequencies use the existing rad/s validation and ordering. Scalar input returns
complex shape `(n_outputs, n_inputs)`; a vector returns
`(n_frequencies, n_outputs, n_inputs)`. The direct-feedthrough term cancels
because balanced truncation preserves `D`. Empty channel axes are preserved
whenever the underlying stable-minimal truncation context is valid.

These are caller-requested input-output frequency samples, not state-
reconstruction errors. They are not generally equal to
`BalancedTruncation.a_priori_error_bound` and do not estimate the global
H-infinity error. The API selects no grid and performs no interpolation,
maximization, norm estimation, or automatic order selection.

## Singular values of balanced-truncation error samples

`system.balanced_truncation_frequency_response_error_singular_values(r,
angular_frequencies)` applies NumPy SVD to the existing sampled error matrices:

```text
singular_values(E(j omega)), where E(j omega) = G(j omega) - G_r(j omega)
```

For `k = min(n_outputs, n_inputs)`, a scalar frequency returns a real
nonnegative descending array with shape `(k,)`; a frequency vector returns
`(n_frequencies, k)` in caller order. Multiplicity and zero-channel shapes are
preserved. All order, stability, minimality, frequency, and pole errors come
unchanged from the sampled-error API.

The largest value at one frequency is the worst-case local input-output gain of
the reduction error at that frequency. These explicit samples are not an
H-infinity estimate and do not generally equal the global
`BalancedTruncation.a_priori_error_bound`. No grid, interpolation,
maximization, or automatic order selection is performed.

## Singular directions of balanced-truncation error samples

`system.balanced_truncation_frequency_response_error_singular_directions(r,
angular_frequencies)` returns an immutable
`BalancedTruncationErrorSingularDirections` with the reduced SVD

```text
E(j omega) = Ue @ diag(error_singular_values) @ Ve.conj().T
```

For `p` original outputs, `m` original inputs, and `k = min(p, m)`, scalar
shapes are `(k,)`, `(p, k)`, and `(m, k)`. A vector of `f` frequencies returns
`(f, k)`, `(f, p, k)`, and `(f, m, k)`. Empty channels retain these conventions
with `k = 0`. Rows of `Ue` follow original output-channel order; rows of `Ve`
follow original input-channel order.

Paired directions have arbitrary unit-magnitude complex phase, and bases within
repeated-singular-value subspaces may rotate. No phase normalization is
imposed. The API inherits all sampled-error validation and adds no grid,
interpolation, maximization, H-infinity estimation, or order selection.

## Static full-state feedback interconnection

`system.full_state_feedback(K)` returns a new `StateSpace` using the convention

```text
u = v - K x
```

where `x` is the plant state and `v` is the new external closed-loop input with
the same dimension and channel order as the plant input `u`. Substitution into
the plant equations gives

```text
A_cl = A - B K
B_cl = B
C_cl = C - D K
D_cl = D
```

The `C_cl` term therefore includes plant feedthrough correctly. `K` must be a
finite real matrix with shape `(n_inputs, n_states)`. With no plant input
channels, the only valid gain shape is `(0, n_states)` and the zero-dimensional
command leaves all matrices unchanged. The original plant is never mutated.
This API interconnects a caller-supplied gain only; it performs no pole
placement, LQR, tuning, reference tracking, observer design, or saturation.

## Controllable SISO pole placement

`system.place_siso_poles(desired_poles)` synthesizes a real full-state feedback
gain for the same `u = v - K x` convention. It supports continuous-time plants
with exactly one input and full controllability, and returns `K` with shape
`(1, n_states)` for direct use with `system.full_state_feedback(K)`.

For

```text
Ctrb = [B, A B, ..., A^(n-1) B]
phi(s) = product(s - desired_pole)
```

the implementation uses Ackermann's formula

```text
K = e_n^T Ctrb^-1 phi(A)
```

without forming an explicit inverse: the selector row is obtained by a linear
solve, and `phi(A)` is evaluated by Horner's method. The desired poles must be
a finite one-dimensional sequence of length `n_states`. Complex poles must be
closed under conjugation within `rtol=1e-7` and `atol=1e-10`, ensuring that the
characteristic polynomial and returned gain are real.

This API performs SISO placement only. It does not require stable requested
poles or otherwise tune the controller; closed-loop stability is the caller's
responsibility. MIMO placement, LQR, reference tracking, observer design, and
aircraft-specific tuning are outside its scope.

## Full-order Luenberger observer interconnection

`system.luenberger_observer(L)` interconnects the plant with a caller-supplied
full-order observer gain using

```text
x_hat_dot = A x_hat + B u + L (y - C x_hat - D u)
e = x - x_hat
```

It returns an immutable `LuenbergerObserverInterconnection` containing the
augmented `StateSpace` and the validated gain. The augmented state order is
`[x; x_hat]`, its external input is the known plant input `u`, and its output
order is `[y; x_hat]`. The complete realization is

```text
A_aug = [[A,   0],       B_aug = [[B],
         [L C, A-L C]]            [B]]

C_aug = [[C, 0],         D_aug = [[D],
         [0, I]]                  [0]]
```

Substituting `y = C x + D u` into the innovation cancels `D u`, giving
`x_hat_dot = L C x + (A - L C) x_hat + B u`. Consequently, the estimation
error satisfies `e_dot = (A - L C) e`; plant feedthrough remains present only
in the exposed plant output `y`.

`L` must be a finite real matrix with shape `(n_states, n_outputs)`.
Observability is not required because this API performs interconnection only.
For a plant with no output channels, the valid gain has shape `(n_states, 0)`
and there is no measurement correction. The plant is not mutated. Observer
gain synthesis, observer pole placement, Kalman filtering, output feedback,
and aircraft-specific design are outside this capability.

## Observable single-output observer pole placement

`system.place_siso_observer_poles(desired_poles)` returns a finite real
observer gain `L` with shape `(n_states, 1)` for direct use with
`system.luenberger_observer(L)`. It supports continuous-time systems with
exactly one output and full observability under the existing structural
analysis.

The method follows the duality

```text
(A - L C)^T = A^T - C^T L^T
```

and applies the existing NumPy-only SISO Ackermann implementation to the dual
pair `(A.T, C.T)`. It therefore shares the linear-solve implementation, Horner
matrix-polynomial evaluation, and desired-pole validation without duplicating
the placement algorithm.

The requested poles must be a finite one-dimensional sequence of length
`n_states`. Complex poles must be closed under conjugation within the existing
`rtol=1e-7` and `atol=1e-10` tolerance so `L` is real. Requested poles are not
required to be stable: convergence of `e_dot = (A - L C)e` is the caller's
responsibility and requires poles with strictly negative real parts. This API
does not perform multi-output placement, Kalman filtering, output-feedback
synthesis, or aircraft-specific tuning, and it does not mutate the plant.

## Observer-based dynamic output feedback

`system.observer_based_output_feedback(K, L)` interconnects caller-supplied
state-feedback and observer gains using

```text
u = v - K x_hat
x_hat_dot = A x_hat + B u + L(y - C x_hat - D u)
y = C x + D u
```

It returns an immutable `ObserverBasedOutputFeedbackInterconnection` containing
the augmented `StateSpace` and validated copies of `K` and `L`. The augmented
state order is `[x; x_hat]`, the external input is the new command `v`, and the
exposed output is the plant output `y`. Its matrices are

```text
A_aug = [[A,    -B K],       B_aug = [[B],
         [L C, A-B K-L C]]            [B]]

C_aug = [C, -D K]            D_aug = D
```

Plant feedthrough is retained in `C_aug` and `D_aug`. There is no algebraic
loop: `u` is determined by `v` and the observer state, and substituting
`y = Cx + Du` makes the identical `Du` terms cancel inside the innovation.

With `e = x - x_hat`, the equivalent `[x; e]` dynamics are

```text
x_dot = (A - B K)x + B K e + B v
e_dot = (A - L C)e
```

This block-triangular form gives the separation principle: augmented poles are
the multiset union of the controller poles from `A - BK` and observer-error
poles from `A - LC`. No stability, controllability, or observability condition
is imposed by the interconnection.

`K` and `L` must be finite real matrices shaped `(n_inputs, n_states)` and
`(n_states, n_outputs)`. Matching empty dimensions support zero-input and
zero-output plants. The API does not mutate the plant or synthesize either
gain, and it adds no reference prefilter, integral action, Kalman filtering,
saturation, or aircraft-specific tuning.

## Nominal SISO steady-state reference prefilter

`system.siso_reference_prefilter(K)` computes a finite real scalar `N` for the
existing full-state-feedback convention

```text
u = v - K x
v = N r
u = N r - K x
```

For the closed-loop realization

```text
A_cl = A - B K
B_cl = B
C_cl = C - D K
D_cl = D
```

the API evaluates the scalar DC gain using the existing NumPy-only frequency
response at zero frequency:

```text
G_cl(0) = -C_cl solve(A_cl, B_cl) + D_cl
N = 1 / G_cl(0)
```

Thus the nominal constant-reference equilibrium satisfies
`y_ss = G_cl(0) N r = r`. The calculation uses a linear solve, not an explicit
inverse, and includes nonzero plant feedthrough through both `C_cl` and `D_cl`.

The plant must have exactly one input and one output, `K` must satisfy the
existing full-state-feedback validation, and `A_cl` must be asymptotically
stable. Nonfinite or materially complex DC results and DC gains whose magnitude
is no larger than `100 * machine epsilon` are rejected as unusable. This is
nominal scaling only: it supplies no integral action, disturbance rejection,
robust tracking, or reference dynamics, and it does not mutate the plant.

## SISO output-error integral augmentation

`system.siso_integral_augmentation()` returns an immutable
`SISOIntegralAugmentation` containing an open augmented design model for

```text
x_dot = A x + B u
y = C x + D u
xi_dot = r - y
```

The augmented state order is `[x; xi]`, input order is `[u; r]`, and output
order is `[y; xi]`. The complete realization is

```text
A_aug = [[ A, 0],       B_aug = [[ B, 0],
         [-C, 0]]                [-D, 1]]

C_aug = [[C, 0],        D_aug = [[D, 0],
         [0, 1]]                 [0, 0]]
```

Nonzero plant feedthrough therefore enters the integral-error equation exactly
as `xi_dot = r - Cx - Du`. The method requires exactly one plant input and one
plant output but imposes no stability, controllability, or observability
condition. It adds one open-loop integrator state and does not mutate the
plant.

This is an auditable design augmentation only. It does not synthesize gains,
close an integral-feedback loop, combine reference prefiltering, or provide
anti-windup, saturation, observer, or aircraft-specific behavior.

## Caller-supplied SISO integral state feedback

`system.siso_integral_state_feedback(K, K_i)` closes a SISO plant with the
explicit controller and output-error integrator

```text
u = -K x + K_i xi
xi_dot = r - y
```

It returns an immutable `SISOIntegralStateFeedbackInterconnection` containing
the augmented `StateSpace`, a validated copy of `K`, and the validated scalar
`K_i`. The state order is `[x; xi]`, the sole external input is the reference
`r`, and the sole output is the physical plant output `y`. The complete
realization is

```text
A_aug = [[A-B K,  B K_i],       B_aug = [[0],
         [-C+D K, -D K_i]]               [1]]

C_aug = [C-D K, D K_i]          D_aug = [0]
```

These terms follow directly from
`y = (C-D K)x + D K_i xi` and
`xi_dot = (-C+D K)x - D K_i xi + r`. Therefore any finite nonzero scalar `D`
is supported without an algebraic inversion: `u` depends only on the augmented
state, so there is no algebraic loop.

The plant must have exactly one input and one output. `K` must be a finite real
matrix with shape `(1, n_states)`, and `K_i` must be a finite real scalar. No
stability, controllability, or observability condition is imposed. The method
does not mutate the plant, synthesize gains, select poles, perform LQR, or add
anti-windup, saturation, observer, or aircraft-specific behavior.

## SISO integral state-feedback pole placement

`system.place_siso_integral_poles(desired_poles)` computes gains for the
existing `u = -K x + K_i xi`, `xi_dot = r - y` interconnection from exactly
`n_states + 1` caller-supplied continuous-time poles. It returns an immutable
`SISOIntegralPolePlacement` with fields `K`, `K_i`, `desired_poles`, and
`achieved_poles`; the gains can be used directly as

```python
placement = system.place_siso_integral_poles(desired_poles)
closed_loop = system.siso_integral_state_feedback(
    placement.K, placement.K_i
)
```

For autonomous synthesis, set `r = 0` and write

```text
A_i = [[ A, 0],       B_i = [[ B],
       [-C, 0]]              [-D]]
```

The existing NumPy-only Ackermann implementation places poles for
`u = -K_aug [x; xi]`. Compatibility with the integral-controller convention
therefore requires

```text
K_aug = [K, -K_i]
K = K_aug[:, :n_states]
K_i = -K_aug[0, -1]
```

Substitution gives the existing closed-loop matrix

```text
A_i - B_i K_aug = [[A-B K,  B K_i],
                   [-C+D K, -D K_i]]
```

Finite nonzero `D` is supported exactly through the `-D` row of `B_i`; it may
change augmented controllability and the resulting gains but creates no
algebraic loop or inversion. The augmented pair must be controllable under the
existing numerical-rank convention. Desired poles reuse the existing finite,
one-dimensional, conjugate-closure validation, and the achieved poles of the
actual integral-feedback interconnection are checked against them with the
repository's existing tolerances.

Requested poles need not be stable. The API performs no automatic pole
selection, LQR/LQI, optimization, gain scheduling, PID design, or
aircraft-specific tuning, and it does not mutate the plant.

## Experimental Platform: SISO response metrics

`response_metrics(time, y, reference)` evaluates a finite sampled SISO
trajectory independently of any plant or controller implementation:

```python
from flightlab.response import response_metrics

metrics = response_metrics(
    time=[0.0, 0.5, 1.0, 2.0],
    y=[0.0, 0.7, 1.1, 1.0],
    reference=[1.0, 1.0, 1.0, 1.0],
)

print(metrics.rms_tracking_error)
print(metrics.settling_time)
```

The immutable result contains read-only copies of time, output, reference, and
tracking error `e = reference - y`. Scalar results include final output and
reference, final sampled error, largest sampled output, maximum absolute error,
time-weighted RMS error, trapezoidal IAE and ISE, overshoot percentage, and
settling time. RMS is `sqrt(ISE / (time[-1] - time[0]))`, so irregular sample
spacing is respected.

Overshoot uses the final reference as a signed target and reports the percentage
by which the furthest output sample in that direction exceeds its magnitude;
no overshoot is `0.0`. Settling time uses an inclusive 2% band around the final
reference and is the earliest sampled time after which every remaining output
sample stays in that band. It does not interpolate. When the final reference
magnitude is at most `100 * machine epsilon`, both percentage overshoot and the
relative settling band are undefined and their results are `None`.

## Experimental Platform: immutable experiment runs

`experiment_run(...)` combines one existing response result with the metadata
needed to describe a completed computational experiment. It records data only;
it does not execute a simulation or persist anything.

```python
from flightlab.experiment import experiment_run
from flightlab.response import response_metrics

time = [0.0, 0.5, 1.0, 2.0]
metrics = response_metrics(
    time=time,
    y=[0.0, 0.7, 1.1, 1.0],
    reference=[1.0, 1.0, 1.0, 1.0],
)
run = experiment_run(
    time=time,
    initial_state=[0.0, 0.0],
    metrics=metrics,
    method="exact",
    system={"name": "demo", "order": 2},
    controller={"type": "integral_state_feedback"},
    reference={"type": "step", "value": 1.0},
    user_metadata={"seed": 7},
)
record = run.reproducibility_record()
```

The immutable run derives start time, end time, duration, and sample count from
the validated time vector, which must exactly match `metrics.time`. Its initial
state is a defensive read-only copy. System, controller, reference, and user
metadata are defensively copied into key-sorted read-only mappings. Values may
be `None`, booleans, integers, finite floats, strings, or one-level tuples of
those scalar types; mutable, nested, and opaque values are rejected.

By default each run receives a UUID4 string and a timezone-aware UTC creation
timestamp; callers may instead supply a nonblank stable identifier and an aware
timestamp, which is normalized to UTC. `reproducibility_record()` returns a
fresh JSON-compatible dictionary with the identity, timestamp, timing data,
initial state, metadata, and every scalar response metric. It deliberately
omits trajectory arrays and performs no filesystem or database operation.

## Experimental Platform: generic experiment execution

`execute_experiment(...)` invokes one caller-supplied zero-argument simulation
callable exactly once and returns one validated `ExperimentRun`:

```python
from flightlab.experiment import SISOSimulationResult, execute_experiment


def simulate():
    return SISOSimulationResult(
        time=[0.0, 0.5, 1.0, 2.0],
        output=[0.0, 0.7, 1.1, 1.0],
        reference=[1.0, 1.0, 1.0, 1.0],
    )


run = execute_experiment(
    simulate,
    initial_state=[0.0, 0.0],
    method="exact",
    system={"name": "demo", "order": 2},
    controller={"type": "integral_state_feedback"},
    reference={"type": "step", "value": 1.0},
    user_metadata={"seed": 7},
)
```

The immutable `SISOSimulationResult` is the complete callable return contract:
sample time, one-dimensional SISO output, and the sampled reference trajectory.
Callers bind any plant, controller, input, or integration arguments themselves
through a closure or `functools.partial`; the execution layer imposes no model
or simulator signature. For an existing `StateSpace` simulation, the caller
must explicitly select its intended SISO output channel.

The sampled `SISOSimulationResult.reference` trajectory is distinct from the
`reference` mapping passed to `execute_experiment()`: the former is evaluated
numerically while the latter is descriptive reproducibility metadata.

The sampled result is passed directly to `response_metrics()`, including an
optional settling tolerance, and the resulting metrics and explicit provenance
are passed to `experiment_run()`. Those existing APIs remain the validation,
copying, immutability, and reproducibility boundaries. Simulation exceptions
propagate unchanged. Execution does not save the run, access SQLite, retry,
batch, sweep, parallelize, or synthesize a controller.

## Experimental Platform: sequential experiment cases

`ExperimentCase` explicitly describes every argument for one
`execute_experiment()` call. `execute_experiments(cases)` executes a finite
ordered iterable of those cases sequentially and returns an immutable tuple of
`ExperimentRun` objects:

```python
from flightlab.experiment import ExperimentCase, execute_experiments

cases = (
    ExperimentCase(
        simulation=simulate,
        initial_state=[0.0, 0.0],
        method="exact",
        system={"name": "demo", "order": 2},
        controller={"type": "integral_state_feedback"},
        reference={"type": "step", "value": 1.0},
        user_metadata={"case": "baseline"},
        run_id="baseline",
    ),
)
runs = execute_experiments(cases)
```

The case container is frozen, slotted, keyword-only, and intentionally shallow:
each case retains its explicit caller-owned configuration until execution.
`execute_experiment()` remains the sole validation and construction boundary,
and each returned run therefore receives the existing defensive snapshots of
sampled arrays, initial state, and metadata.

The input iterable is materialized before execution so its membership and order
cannot change between cases. Every element is verified as an `ExperimentCase`
before any simulation starts. Cases then execute exactly once each in caller
order. Empty input returns `()`. On the first simulation or validation failure,
the original exception propagates unchanged: earlier cases have completed, no
partial tuple is returned, later cases do not run, and there is no rollback or
retry. Batch execution performs no case generation, Cartesian product,
persistence, automatic saving, concurrency, or optimization.

## Experimental Platform: explicit parameter-value expansion

`expand_experiment_cases(parameter_values, case_factory)` maps one finite
ordered iterable of caller-supplied parameter values into explicit cases:

```python
from flightlab.experiment import ExperimentCase, expand_experiment_cases


def case_factory(gain):
    return ExperimentCase(
        simulation=lambda: simulate_gain(gain),
        initial_state=[0.0, 0.0],
        method="exact",
        system={"name": "demo", "order": 2},
        controller={"type": "state_feedback", "gain": gain},
        reference={"type": "step", "value": 1.0},
        user_metadata={"gain": gain},
        run_id=f"gain-{gain}",
    )


cases = expand_experiment_cases([0.5, 1.0, 1.5], case_factory)
```

The parameter iterable is materialized before factory calls, preserving its
membership and caller order. The factory is invoked exactly once per value and
must return an `ExperimentCase`; invalid results report their parameter index.
Empty input returns `()`, and factory exceptions propagate unchanged. The
result is an immutable tuple. Expansion does not invoke case simulations,
execute experiments, persist data, generate Cartesian products, or interpret
parameter values.

## Experimental Platform: Cartesian parameter-axis expansion

`expand_cartesian_experiment_cases(parameter_axes, case_factory)` expands
finite ordered axes into explicit cases without executing them. The factory
receives each parameter combination as a tuple:

```python
from flightlab.experiment import ExperimentCase, expand_cartesian_experiment_cases


def cartesian_case_factory(combination):
    gain, tolerance = combination
    return ExperimentCase(
        simulation=lambda: simulate_gain(gain),
        initial_state=[0.0, 0.0],
        method="exact",
        system={"name": "demo", "order": 2},
        controller={"type": "state_feedback", "gain": gain},
        reference={"type": "step", "value": 1.0},
        settling_tolerance=tolerance,
        run_id=f"gain-{gain}-tolerance-{tolerance}",
    )


cases = expand_cartesian_experiment_cases(
    parameter_axes=([0.5, 1.0], [0.02, 0.05]),
    case_factory=cartesian_case_factory,
)
```

Combination order follows `itertools.product`: axis and value order are
preserved, and the rightmost axis varies fastest. Thus the example combinations
are `(0.5, 0.02)`, `(0.5, 0.05)`, `(1.0, 0.02)`, and `(1.0, 0.05)`. Zero axes
produce the single empty combination `()`, while any empty axis produces no
combinations and therefore returns `()`.

Every axis is materialized before factory calls. The existing
`expand_experiment_cases()` boundary invokes the factory exactly once per
combination, validates each result as an `ExperimentCase`, preserves factory
exceptions, and returns an immutable tuple. Cartesian expansion invokes no
case simulation and performs no experiment execution or persistence.

## Experimental Platform: sequential Cartesian execution

`execute_cartesian_experiments(parameter_axes, case_factory)` is the complete
generic sequential campaign composition:

```python
from flightlab.experiment import execute_cartesian_experiments

runs = execute_cartesian_experiments(
    parameter_axes=([0.5, 1.0], [0.02, 0.05]),
    case_factory=cartesian_case_factory,
)
```

The API is deliberately equivalent to passing
`expand_cartesian_experiment_cases(parameter_axes, case_factory)` directly to
`execute_experiments()`. Case generation and validation therefore finish before
the first simulation starts. Each factory and successful simulation is invoked
exactly once, and returned runs retain deterministic Cartesian order, with the
rightmost axis varying fastest. Zero axes execute the one empty combination;
any empty axis returns `()` without calling the factory or a simulation.

Factory errors and invalid factory results occur before execution and propagate
unchanged. During execution, the first validation or simulation failure stops
the campaign: earlier simulations have completed, the failing case retains its
existing single-run behavior, later simulations do not run, and there is no
retry or rollback. Results are an immutable tuple of `ExperimentRun` objects.
The campaign API performs no persistence, automatic saving, parallel execution,
optimization, or controller synthesis.

## Experimental Platform: SQLite experiment storage

`SQLiteExperimentStore` persists the existing reproducibility record without
running a simulation or reconstructing an `ExperimentRun` on retrieval:

```python
from flightlab.experiment import experiment_run
from flightlab.persistence import SQLiteExperimentStore
from flightlab.response import response_metrics

time = [0.0, 0.5, 1.0, 2.0]
metrics = response_metrics(time, [0.0, 0.7, 1.1, 1.0], [1.0] * 4)
run = experiment_run(
    time=time,
    initial_state=[0.0, 0.0],
    metrics=metrics,
    method="exact",
    system={"name": "demo"},
    controller={"type": "integral_state_feedback"},
    reference={"type": "step", "value": 1.0},
)

store = SQLiteExperimentStore("flightlab.db")
store.initialize()
store.save(run)
record = store.get(run.run_id)
store.close()
```

Initialization is idempotent. `save()` uses one atomic parameterized `INSERT`;
an existing ID raises `DuplicateRunIDError` and is never overwritten. `get()`
returns a fresh plain record equivalent to `run.reproducibility_record()`, or
`None` for an unknown ID. `list_runs()` returns immutable lightweight summaries
ordered by creation time newest-first, with ascending run ID as the deterministic
tie-breaker.

`save_many(runs)` snapshots and validates a finite ordered collection, reuses
the same deterministic record and parameterized insertion path, and commits the
complete collection in one transaction. Duplicate IDs within the collection or
against stored data, malformed runs, schema violations, and database failures
leave none of that collection persisted.

The single `experiment_runs` table stores identity, UTC timestamp, timing, and
all scalar metrics in typed columns. Initial state and the four metadata groups
use deterministic standard-library JSON with sorted keys, compact separators,
and nonfinite values disabled; pickle is never used. One connection is retained
for each store lifetime so `:memory:` works as expected. The store is also a
context manager that initializes on entry and closes on exit; explicit
`close()` is idempotent and terminal.

## Experimental Platform: sequential campaigns with optional persistence

`run_experiment_campaign(cases, store=None)` composes the existing sequential
execution and atomic SQLite batch-persistence boundaries:

```python
from flightlab.campaign import run_experiment_campaign
from flightlab.persistence import SQLiteExperimentStore

with SQLiteExperimentStore("flightlab.db") as store:
    campaign = run_experiment_campaign(cases, store=store)

runs = campaign.runs
```

The returned frozen, slotted `ExperimentCampaignResult` contains the completed
runs as an immutable tuple in caller order. With no store, the campaign remains
in memory. With a store, every case first completes through
`execute_experiments()`; only then is the complete tuple passed once to
`SQLiteExperimentStore.save_many()`.

An execution failure propagates unchanged and prevents all campaign
persistence. A persistence failure propagates after execution and the existing
`save_many()` transaction rolls back every run from that campaign. There are no
partial campaign results, retries, automatic case generation, parallel work,
optimization, analysis, or CLI behavior.
