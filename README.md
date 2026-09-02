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

## Verification: analytical linear state-space benchmark

`run_linear_state_space_verification_benchmark()` verifies the existing
continuous-time `StateSpace` eigenvalue and exact physical-coordinate
propagation APIs against one independent closed-form two-state solution:

```python
from flightlab.verification import (
    run_linear_state_space_verification_benchmark,
)

run = run_linear_state_space_verification_benchmark()
record = run.reproducibility_record()

assert record["user_metadata"]["passed"] is True
```

The fixed coupled SISO system uses `A = [[-1, 1], [0, -2]]`,
`B = [[0], [1]]`, `C = [[1, 0]]`, `D = [[0]]`, initial state
`[1.5, -0.5]`, constant input `2.0`, and the nonuniform time grid
`[0.0, 0.125, 0.5, 1.25, 2.0]`. For elapsed time `tau`, its independently
encoded oracle is

```text
lambda = {-1, -2}
x2(tau) = 1 - 1.5 exp(-2 tau)
x1(tau) = 1 - exp(-tau) + 1.5 exp(-2 tau)
y(tau)  = x1(tau)
```

The runner computes maximum absolute eigenvalue, state, and output residuals.
Each must be at most `1e-12`, and the exact initial state must be preserved.
Malformed shapes or nonfinite evidence raise `ValueError`; well-formed evidence
outside the acceptance limits returns a run whose `user_metadata["passed"]` is
`False`.

The returned value is the existing immutable `ExperimentRun`. Its sampled
output is the FlightLab result, its sampled reference is the analytical output,
and its standard reproducibility record contains fixed identity, timestamp,
benchmark provenance, residuals, tolerance, and pass state. Repeated benchmark
calls therefore produce identical detached JSON-compatible records without a
new V&V result or serialization type. This is software verification against a
mathematical oracle, not physical validation of an aircraft or controller.

`run_scipy_linear_state_space_verification_benchmark()` adds a second fixed
software-verification check through independently maintained SciPy APIs:

```python
from flightlab.verification import (
    run_scipy_linear_state_space_verification_benchmark,
)

run = run_scipy_linear_state_space_verification_benchmark()

assert run.user_metadata["passed"] is True
assert run.reference["library"] == "scipy"
```

The benchmark compares `StateSpace.eigenvalues()` with
`scipy.linalg.eigvals()` and compares exact physical-coordinate propagation
with `scipy.signal.lsim(..., interp=False)`. Its fixed damped oscillator uses
`A = [[0, 1], [-4, -0.8]]`, `B = [[0], [1]]`, `C = [[1, 0.25]]`,
`D = [[0.1]]`, initial state `[0.75, -0.25]`, a uniform `0.125`-second grid
from `0` through `2` seconds, and three explicit zero-order-hold input levels.
This adds complex-conjugate poles, input transitions, mixed-state output, and
direct feedthrough beyond the analytical benchmark.

Maximum absolute residual limits are `1e-12` for eigenvalues and `1e-10` for
states and outputs. Exact returned-time equality and exact initial-state
preservation by both implementations are also required. Malformed reference
evidence raises `ValueError`; well-formed evidence outside a limit returns an
existing `ExperimentRun` with `passed = False`. The run records the resolved
SciPy version and fixed provenance, so repeated calls in the same locked
environment produce equal detached JSON-compatible records.

SciPy is intentionally a development/test dependency, imported only inside
this runner. Importing `flightlab.verification` and using the analytical
benchmark remain NumPy-only. This comparison is corroborating software
verification, not physical aircraft validation.

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

The `experiment_runs` table stores identity, UTC timestamp, timing, and all
scalar metrics in typed columns. Initial state and the four metadata groups use
deterministic standard-library JSON with sorted keys, compact separators, and
nonfinite values disabled; pickle is never used. Campaign manifests use one
small campaign table and one foreign-keyed membership table with explicit
zero-based positions. One connection is retained for each store lifetime so
`:memory:` works as expected. The store is also a context manager that
initializes on entry and closes on exit; explicit `close()` is idempotent and
terminal.

## Experimental Platform: sequential campaigns with optional persistence

`run_experiment_campaign(cases, store=None, *, campaign_id=None,
created_at=None)` composes the existing sequential execution and atomic SQLite
campaign-persistence boundaries:

```python
from flightlab.campaign import run_experiment_campaign
from flightlab.persistence import SQLiteExperimentStore

with SQLiteExperimentStore("flightlab.db") as store:
    campaign = run_experiment_campaign(
        cases,
        store=store,
        campaign_id="baseline-campaign",
    )
    manifest = store.get_campaign(campaign.campaign_id)

runs = campaign.runs
```

The returned frozen, slotted `ExperimentCampaignResult` contains a nonblank
campaign ID, an aware UTC creation time, and the completed runs as an immutable
tuple in caller order. IDs default to UUID4 strings and timestamps default to
the current UTC time; explicit aware timestamps are normalized to UTC. With no
store, the campaign remains in memory.

With a store, every case first completes through `execute_experiments()`. The
completed result is then passed once to `SQLiteExperimentStore.save_campaign()`.
That method reuses the existing validated run-record insertion path and writes
all new runs, one campaign row, and positional run-membership rows in a single
transaction. Empty campaigns persist a valid manifest with zero memberships.

`get_campaign(campaign_id)` returns a new frozen
`ExperimentCampaignManifest` containing the campaign ID, ISO UTC timestamp,
and ordered tuple of run IDs, or `None` when unknown. It does not reconstruct
full runs. Duplicate campaign IDs raise `DuplicateCampaignIDError`.

`get_campaign_bundle(campaign_id)` composes that manifest lookup with the
existing `get()` record retrieval for each member:

```python
bundle = store.get_campaign_bundle("baseline-campaign")
if bundle is not None:
    manifest = bundle.manifest
    records = bundle.records
```

The frozen `ExperimentCampaignBundle` contains the detached manifest and an
immutable tuple of detached reproducibility-record dictionaries in exact
membership order. Unknown campaigns return `None`. Repeated reads are
deterministic and return fresh records, so caller mutations cannot affect the
database or later reads. A manifest that references a missing run raises a
clear stored-state `ValueError`; no `ExperimentRun` is reconstructed and the
operation performs no writes.

`campaign_bundle_record(bundle)` is the pure serialization-ready boundary for
an already-retrieved bundle:

```python
from flightlab.persistence import campaign_bundle_record

plain_record = campaign_bundle_record(bundle)
```

It returns this exact plain structure:

```python
{
    "manifest": {
        "campaign_id": "baseline-campaign",
        "created_at": "2026-09-01T12:00:00+00:00",
        "run_ids": ["baseline", "candidate"],
    },
    "records": [baseline_record, candidate_record],
}
```

Manifest and record order are preserved exactly. Every nested dictionary and
list is detached on every call, and each member record retains the established
`ExperimentRun.reproducibility_record()` representation. The function validates
manifest metadata, record structure, collection lengths, and positional run-ID
agreement. It performs no SQLite access, execution, persistence, JSON writing,
or file I/O.

## Experimental Platform: ordered campaign comparisons

`compare_campaign_runs(...)` extracts one explicitly selected provenance
parameter and one or more explicitly selected existing response metrics from a
campaign bundle record:

```python
from flightlab.analysis import compare_campaign_runs

comparison = compare_campaign_runs(
    plain_record,
    parameter_category="controller",
    parameter_key="gain",
    metric_names=("iae", "overshoot_percent"),
)
```

The parameter category must be exactly `system`, `controller`, `reference`, or
`user_metadata`, and the key must exist in every run with a JSON scalar value.
Metric names form a nonempty, duplicate-free ordered selection from the eleven
existing response metrics; optional metrics retain `None`.

The result is an immutable tuple of frozen `CampaignComparisonEntry` objects in
exact campaign order. Each entry contains `run_id`, the selected
`parameter_value`, and ordered `(metric_name, value)` pairs. Values are copied
from existing provenance and metric records without inference, recomputation,
normalization, ranking, aggregation, or sorting. The operation is deterministic,
pure, and independent of subsequent source-record mutation.

`campaign_metric_deltas(comparison, baseline_run_id)` transforms an existing
ordered comparison into explicit-baseline absolute deltas:

```python
from flightlab.analysis import campaign_metric_deltas

deltas = campaign_metric_deltas(comparison, baseline_run_id="baseline")
```

The baseline ID is mandatory and must occur exactly once. Parameter values must
be finite real integers or floats (booleans are rejected), and every entry must
have the same nonempty ordered metric layout. Each frozen `CampaignDeltaEntry`
retains the run ID and contains `parameter_delta` plus ordered
`(metric_name, delta)` pairs, all computed as `current - baseline`. Campaign
order is unchanged, and every numeric baseline delta is exactly zero.

Optional metrics use one consistent rule: a metric delta is `None` whenever the
current or baseline value is `None`; otherwise both values must be finite
numeric scalars. The transformation performs no metric recomputation, sorting,
ranking, normalization, aggregation, persistence, or source mutation.

`campaign_secant_sensitivities(deltas)` converts those absolute deltas into
baseline-relative secant slopes:

```python
from flightlab.analysis import campaign_secant_sensitivities

sensitivities = campaign_secant_sensitivities(deltas)
```

For each available metric on a run with nonzero parameter delta, the value is
exactly `metric_delta / parameter_delta`. These are finite secant slopes between
the selected baseline and each run—not local derivatives, regression
coefficients, or finite-difference derivative estimates. Campaign order, metric
order, run IDs, and parameter deltas are retained in frozen
`CampaignSensitivityEntry` objects.

No epsilon tolerance is used. The baseline has exactly zero parameter delta and
therefore `None` sensitivity for every metric. Any non-baseline run with the
same parameter value also has zero parameter delta and the same explicit `None`
behavior. Optional `None` metric deltas remain `None`; nonnumeric, nonfinite,
structurally inconsistent, or overflow-producing inputs are rejected.

`campaign_sensitivity_matrix(parameters)` assembles explicit one-at-a-time
secant results into an immutable metric-by-parameter matrix. Each column is
described by a frozen `SensitivityMatrixParameter` containing a unique
parameter name, one existing ordered sensitivity result, and one explicit
nonzero-delta representative run ID:

```python
from flightlab.analysis import (
    SensitivityMatrixParameter,
    campaign_sensitivity_matrix,
)

matrix = campaign_sensitivity_matrix(
    (
        SensitivityMatrixParameter("gain", gain_sensitivities, "gain-high"),
        SensitivityMatrixParameter(
            "damping", damping_sensitivities, "damping-low"
        ),
    )
)
```

Rows are response metrics, columns are varied parameters, and element `(i, j)`
is the already-computed baseline-relative secant sensitivity of metric `i` to
parameter `j` at that parameter's explicitly selected representative run. The
frozen `CampaignSensitivityMatrix` records parameter names, metric names,
representative run IDs, and row-major values as immutable tuples. Caller column
order and existing metric order are preserved exactly; undefined sensitivities
remain `None`. Empty parameter input returns an explicit all-empty matrix.

The assembler validates unique names and representative IDs, complete source
layouts, finite values, compatible metric rows, and nonzero representative
parameter deltas. It performs no representative inference, sensitivity
recalculation, normalization, ranking, regression, aggregation, or persistence.

`project_campaign_metric_changes(matrix, parameter_changes)` applies one
explicit named parameter-change vector to an existing sensitivity matrix:

```python
from flightlab.analysis import (
    CampaignParameterChange,
    project_campaign_metric_changes,
)

projection = project_campaign_metric_changes(
    matrix,
    (
        CampaignParameterChange("gain", 0.5),
        CampaignParameterChange("damping", -0.1),
    ),
)
```

The mathematical definition is
`predicted_metric_changes = sensitivity_matrix @ parameter_changes`, or
`Δm_hat = S_secant Δp`. Each metric change is the finite row sum of
`sensitivity[i, j] * parameter_change[j]`. Parameter names and order must match
the matrix columns exactly; metric output order matches the matrix rows.

The frozen `CampaignMetricChangeProjection` retains parameter names, metric
names, the detached numeric change vector, and ordered predicted metric
changes. If any sensitivity in a metric row is `None`, that metric projection
is `None`, even when the corresponding parameter change is zero; unavailable
sensitivity is never treated as zero. Empty matrix plus empty vector returns an
explicit empty projection.

This is a linear projection built from sampled baseline-relative secant
sensitivities. It is not a nonlinear simulation, local Jacobian, regression
model, or guarantee of accuracy outside the sampled campaign region. The API
performs no inference, reordering, rescaling, simulation, or persistence.

`project_campaign_scenarios(matrix, scenarios)` applies the same matrix to a
finite explicit ordered collection of named change vectors. Each frozen
`CampaignProjectionScenario` contains one unique nonblank name and one immutable
parameter-change tuple in matrix column order:

```python
from flightlab.analysis import (
    CampaignProjectionScenario,
    project_campaign_scenarios,
)

results = project_campaign_scenarios(
    matrix,
    (
        CampaignProjectionScenario("nominal", nominal_changes),
        CampaignProjectionScenario("stress", stress_changes),
    ),
)
```

The scenario iterable is fully materialized, and every scenario definition and
change vector is validated before projection begins. Each scenario then
delegates exactly once to `project_campaign_metric_changes()`. Returned frozen
`CampaignProjectionScenarioResult` objects preserve scenario order and retain
the scenario name plus its detached immutable projection. Empty scenario input
returns `()` after validating the matrix.

The API generates or infers no scenario, silently skips no failure, and performs
no aggregation, comparison, ranking, probability modeling, Monte Carlo work,
simulation, or persistence. The first validation or projection error propagates
and no partial result is returned.

`campaign_projection_residuals(scenario_result, observed_delta)` compares one
explicit named scenario projection with one caller-selected observed
`CampaignDeltaEntry`. For every metric in exact projection order it computes:

```text
residual = observed metric delta - projected metric change
```

The frozen `CampaignProjectionResiduals` retains the scenario name, observed
run ID, and immutable ordered `CampaignMetricResidual` entries containing the
metric name, projected change, observed change, and residual. A zero residual
is exact agreement. A positive residual means the signed observed change is
greater than predicted (under-prediction by this convention); a negative
residual means it is smaller than predicted (over-prediction).

Projected and observed metric layouts must match exactly in name and order.
When either value is `None`, its residual is also `None`; missing values are
never treated as zero. All defined inputs and computed residuals must be finite.
This pure comparison reuses existing deltas and projections and performs no
simulation, metric calculation, projection, fitting, statistics, correction,
or persistence.

`check_campaign_projection_residual_tolerances(residuals, tolerances)` checks
one existing residual result against explicit caller-defined maximum absolute
residuals in exact metric order. Each frozen
`CampaignMetricResidualTolerance` names one metric and its finite nonnegative
tolerance. For every defined residual the API computes:

```text
absolute_residual = abs(residual)
margin = maximum_absolute_residual - absolute_residual
```

The metric passes exactly when its margin is nonnegative, so an exact-boundary
or zero-residual check passes. The frozen
`CampaignProjectionResidualToleranceResults` retains the scenario name and
observed run ID plus ordered `CampaignMetricResidualToleranceResult` entries
containing the signed residual, absolute residual, tolerance, margin, and pass
state. If a residual is `None`, its absolute residual and margin remain `None`
and it does not pass.

This answers whether each existing secant-based projection error remains inside
one explicit absolute tolerance. It is a deterministic approximation-quality
check, not statistical validation, uncertainty quantification, fitting,
regression, or automatic model correction.

`validate_campaign_projection_cases(cases)` evaluates a finite explicit ordered
validation set. Each frozen `CampaignProjectionValidationCase` contains a
unique nonblank case name, one existing named scenario projection, one explicit
observed `CampaignDeltaEntry`, and one immutable ordered tolerance tuple.

The outer iterable is fully materialized and all case-level names and member
types are validated before evaluation. Every case then delegates first to
`campaign_projection_residuals()` and then to
`check_campaign_projection_residual_tolerances()`; it introduces no alternate
residual or tolerance calculation. Each frozen
`CampaignProjectionValidationResult` retains the case name, scenario name,
observed run ID, residual result, and tolerance-check result. Case and metric
ordering, failures, and undefined metrics retain their existing semantics.

Empty input returns `()`. Any invalid case or delegated analysis failure
propagates without skipping a case or returning a partial result. This is pure
ordered analytical orchestration, with no inference, aggregation, scoring,
statistics, fitting, simulation, or persistence.

`campaign_projection_validation_verdict(validation_results)` reduces an
existing ordered validation set to one frozen
`CampaignProjectionValidationVerdict`. It retains `overall_passed` and ordered
tuples of passing, failing, and undefined validation-case names. A case passes
only when every metric check is defined and passing. Any defined failed metric
makes an otherwise defined case fail; any undefined metric makes a case
undefined and non-passing.

Categories are mutually exclusive. Undefined takes deterministic precedence
when one case contains both a failed defined metric and an undefined metric;
the case appears only in `undefined_cases`. Original case order is preserved
within every category. Overall pass requires at least one case and every case
passing, so empty input returns a non-passing verdict with empty categories.

Before classification, the API validates case uniqueness, all nested
scenario/run identities, metric layouts, residual values, absolute residuals,
tolerances, margins, optional states, and stored pass flags. It uses those
existing results without recomputing projections, observations, deltas, or
residuals. This is a deterministic summary against caller-defined residual
tolerances—not external physical validation, probabilistic certification, or a
safety proof.

`campaign_projection_validation_residual_envelopes(validation_results)` finds
the worst defined absolute projection residual for each metric across one
explicit ordered validation set. Each frozen
`CampaignMetricValidationResidualEnvelope` retains the metric name, maximum
stored absolute residual, and the attaining validation-case, scenario, and
observed-run identities. Metric layout order is preserved exactly across all
cases.

Only defined stored absolute residuals participate. If every case is undefined
for one metric, its maximum and all three attaining identities are `None`.
Exact ties select the first attaining validation case in caller order. Empty
input returns `()`.

The API shares the complete nested validation used by the validation verdict,
including identity, layout, finite-value, absolute-residual, optional-state,
margin, and pass-state consistency checks. It does not recalculate a projection,
observation, delta, residual, or tolerance. The resulting envelope identifies
the worst observed projection error in this deterministic validation set; it is
not a probabilistic error bound, confidence interval, or external physical-
validation guarantee.

`campaign_projection_error_summaries(validation_results)` produces immutable
descriptive error records for the same validated deterministic case set. Each
frozen `CampaignMetricProjectionErrorSummary` retains the metric name, total
case count, defined and undefined residual counts, minimum and maximum signed
residuals, mean signed residual, mean absolute residual, and maximum absolute
residual.

Undefined residuals contribute only to the undefined count. When a metric has
no defined residual, every numeric summary is `None`. Metric order and layout
must match across all cases. The maximum absolute residual is taken from
`campaign_projection_validation_residual_envelopes()`, so its semantics remain
identical to the established worst-case envelope. Empty input returns `()` and
nonfinite aggregate arithmetic is rejected.

These values describe observed deterministic projection errors across the
explicit validation set. They are not statistical validation, confidence
intervals, probability estimates, external physical validation, calibration,
regression, model fitting, ranking, or automatic model correction.

`compare_campaign_projection_error_summaries(left_name, left, right_name,
right)` compares two explicitly identified existing summary collections in
exact metric order. Each frozen
`CampaignMetricProjectionErrorSummaryComparison` retains both collection names,
the metric name, detached copies of the left and right summaries, and
right-minus-left differences for defined/undefined counts, signed extrema,
mean signed residual, mean absolute residual, and maximum absolute residual.

Every optional difference is `None` when either source value is `None`; missing
values are never treated as zero. Both collections are materialized and fully
validated before comparison, including unique metric names, common case counts,
count totals, defined/undefined consistency, finite values, signed extrema, and
absolute-error relationships. Empty plus empty returns `()`; one empty and one
nonempty collection is incompatible.

The comparison exposes deterministic descriptive changes only. A negative
absolute-error difference describes a smaller supplied summary value, but the
API does not infer that a collection is better or perform scoring, ranking,
normalization, significance testing, fitting, calibration, or acceptance.

`compare_campaign_projection_error_summary_collections(baseline_name,
baseline, comparison_collections)` applies that pairwise comparison to an
explicit finite ordered set. Each comparison is supplied as a frozen
`CampaignProjectionErrorSummaryCollection` with a unique nonblank name and an
ordered summary tuple. Each frozen
`CampaignProjectionErrorSummaryComparisonSetResult` retains the baseline and
comparison collection names plus the delegated pairwise results.

The complete input is materialized and validated before any pairwise
evaluation. The baseline is reused unchanged for every delegated comparison,
comparison collection order and metric order are preserved exactly, and the
pairwise API is called once per collection. Empty comparison input returns
`()`; malformed entries, invalid or conflicting names, malformed summaries,
incompatible metric layouts, and delegated failures raise without returning a
partial result. This orchestration remains purely analytical and adds no
aggregation, ranking, scoring, persistence, simulation, or model correction.

`campaign_projection_error_comparison_set_metric_envelopes(results)` reduces
those existing ordered comparison-set results without recomputing summaries,
residuals, projections, observations, or comparisons. It returns frozen
`CampaignProjectionErrorSummaryDifferenceEnvelope` records in metric-major
order and, within each metric, the documented comparison-difference field
order. Every record retains the metric and difference-field names, finite
minimum and maximum stored differences, and the first comparison collection
attaining each extremum.

The complete finite input is materialized and validated before reduction.
Baseline identity and summaries, comparison identities, nested records, and
metric layouts must be consistent. Defined differences must be finite numeric
scalars and cannot be booleans. `None` values are ignored when defined values
exist; if a field is undefined across every collection, both extrema and both
attaining names are `None`. Exact ties retain the first collection in original
order. Empty input returns `()`.

These envelopes describe only finite extrema of already-computed
projection-error summary differences across explicit comparison collections.
They are not statistical confidence bounds, probabilistic uncertainty,
ranking, scoring, validation thresholds, calibration, regression, fitting,
optimization, or physical-certification bounds.

`check_campaign_projection_error_comparison_envelope_limits(envelopes,
limits)` checks each existing metric/difference-field envelope against one
explicit aligned `CampaignProjectionErrorSummaryDifferenceLimit`. Limits must
provide exact metric and difference-field coverage in envelope order, with
finite non-boolean bounds satisfying `allowable_minimum_difference <=
allowable_maximum_difference`.

Each frozen `CampaignProjectionErrorSummaryDifferenceLimitResult` retains the
metric and field identity, observed extrema, allowable interval, margins, and
pass state. The margins are `observed_minimum - allowable_minimum` and
`allowable_maximum - observed_maximum`; a defined field passes only when both
are nonnegative. An undefined envelope produces `None` margins and does not
pass. Empty envelopes plus empty limits return `()`.

This pure check only determines whether observed finite comparison-envelope
differences remain inside explicit caller-defined deterministic intervals. It
is not statistical significance testing, a confidence interval, probabilistic
uncertainty, physical validation, automatic acceptance criteria, calibration,
regression, fitting, optimization, or certification.

`campaign_projection_error_comparison_envelope_limit_verdict(limit_results)`
reduces one existing ordered limit-result collection to a frozen
`CampaignProjectionErrorComparisonEnvelopeLimitVerdict`. Each category stores
immutable `CampaignProjectionErrorMetricFieldIdentity` values so metric and
difference-field traceability remain together.

Defined passing results enter `passing_identities`, defined non-passing results
enter `failing_identities`, and structurally undefined non-passing results enter
`undefined_identities`. Original order is preserved within every mutually
exclusive category. `overall_passed` is true only for nonempty input when every
result is defined and passing; empty input returns an explicit non-passing
verdict with three empty categories.

The API fully validates the stored result layout, identities, finite interval
and envelope values, optional states, margins, and pass states without
recomputing any of them. This verdict only summarizes deterministic
caller-defined comparison-envelope interval checks. It is not statistical or
physical validation, probabilistic or safety certification, scoring, ranking,
calibration, regression, fitting, optimization, or automatic model correction.

`campaign_projection_error_comparison_envelope_metric_verdicts(limit_results)`
uses the same complete stored-result validation and returns one frozen
`CampaignProjectionErrorMetricEnvelopeLimitVerdict` per metric in original
metric order. Every verdict retains its metric name, metric-level pass state,
and ordered passing, failing, and undefined
`CampaignProjectionErrorMetricFieldIdentity` values.

A metric passes only when its complete nonempty difference-field layout is
defined and every stored field result passes. Defined non-passing fields and
structurally undefined non-passing fields remain in separate mutually exclusive
categories. Field order is preserved within each category. Empty input returns
`()`.

These per-metric verdicts only localize existing deterministic caller-defined
comparison-envelope interval checks. They are not statistical or physical
validation, probabilistic or safety certification, scoring, ranking,
calibration, regression, fitting, optimization, or automatic model correction.

`campaign_projection_error_comparison_envelope_assessment_report(limit_results)`
assembles one frozen
`CampaignProjectionErrorComparisonEnvelopeAssessmentReport` containing detached
ordered copies of the checked limit results, the existing overall verdict, and
the existing ordered per-metric verdicts. The overall and per-metric verdict
APIs are each invoked exactly once; classification logic is not duplicated.

Before returning, the assembly verifies exact metric/field coverage and order,
mutually exclusive categories, agreement between global and per-metric
classifications, and agreement between global, metric, and field-level pass
states. Every retained limit result preserves its observed extrema, allowable
interval, margins, and stored pass state. Empty input produces a report with no
results, the existing explicit non-passing overall verdict, and no metric
verdicts.

This report is a deterministic analytical assembly of already-computed
comparison-envelope checks and verdicts. It is not external physical or
statistical validation, certification, confidence scoring, model calibration,
regression, fitting, optimization, or automatic model correction.

`campaign_projection_error_comparison_envelope_assessment_record(report)`
validates one existing assessment report and returns a fresh deterministic
JSON-compatible plain dictionary. The record contains ordered limit-result
dictionaries, the overall verdict and its ordered metric/field identity
dictionaries, and ordered per-metric verdict dictionaries with their ordered
field-name categories.

Undefined observed extrema and margins remain `None`; they are never converted
to zero. Defined numeric values must remain finite and non-boolean, and all
stored report identities, categories, ordering, margins, and pass states must
be internally consistent. Every call creates new nested dictionaries and lists,
so caller mutation cannot affect the source report or later records. The empty
report is represented by an empty `limit_results` list, the explicit non-passing
overall verdict with empty identity lists, and an empty `metric_verdicts` list.

This API creates a deterministic JSON-compatible representation only. It does
not write JSON, save files, persist to SQLite, communicate over a network, or
alter analytical results.

`campaign_projection_error_comparison_envelope_named_assessment_records(entries)`
converts an explicit finite ordered collection of frozen
`CampaignProjectionErrorNamedAssessmentReport` entries. Every entry requires a
unique nonblank caller-supplied name and one existing assessment report. The
complete collection is materialized and all names and member types are checked
before conversion begins.

The existing single-report record converter is invoked exactly once per entry
in caller order. The result is a fresh JSON-compatible list whose entries have
stable `name` and `report` fields. Empty input returns `[]`. Invalid collection
entries are never skipped, delegated report validation failures propagate, and
no partial list is returned. Returned nested structures are detached from both
the named entries and their source reports.

`campaign_projection_error_comparison_envelope_verdict_overview(entries)`
extracts a deliberately compact plain overview from the same explicitly named
assessment reports. Each output dictionary contains only `name`, the stored
`overall_passed` state, and an ordered `metrics` list of `{metric, passed}`
dictionaries.

The complete named collection and every report's stored cross-view verdict
structure are validated before extraction. Report order and per-report metric
order are preserved exactly. Empty input returns `[]`, and every call returns
fresh detached JSON-compatible dictionaries and lists. The overview does not
include limit results, extrema, margins, residuals, or identity-category lists,
and it infers no scores, rankings, probabilities, or acceptance conclusions.

`campaign_projection_error_comparison_envelope_assessment_collection_verdict(
entries)` reduces the validated ordered named collection to one frozen
`CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict`. It
retains the overall pass state plus ordered passing, failing, and undefined
report-name tuples.

A report is undefined when its stored overall verdict contains any undefined
metric/field identity; this classification takes precedence over an ordinary
failure. Otherwise, its stored overall pass state determines passing or
failing. The collection passes only when it is nonempty and every report is
defined and passing. Empty input returns an explicit non-passing verdict with
three empty categories. This verdict summarizes deterministic caller-defined
assessment reports only; it is not external or statistical validation,
certification, scoring, ranking, calibration, fitting, regression,
optimization, or automatic model correction.

`campaign_projection_error_comparison_envelope_assessment_collection_report(
entries)` assembles the validated ordered named reports and their existing
collection verdict into one frozen
`CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport`. The
collection-verdict API is invoked exactly once; collection classification is
not duplicated and no analytical values are recomputed.

The report retains detached deep copies of every named assessment report in
caller order, including all nested metric and difference-field ordering, plus
the existing overall collection verdict. Before returning, it checks exact
report-name coverage and source order, mutually exclusive categories,
undefined precedence, agreement with every stored assessment verdict, and the
collection pass state. Empty input retains `()` named reports and the existing
explicit non-passing empty verdict. This is orchestration of already-validated
analytical evidence only, not validation, certification, scoring, ranking,
calibration, regression, fitting, optimization, or model correction.

`campaign_projection_error_comparison_envelope_assessment_collection_record(
report)` validates one existing collection report and returns a fresh
deterministic JSON-compatible dictionary. Its `named_reports` list reuses the
existing named/single-assessment conversion path, while `collection_verdict`
contains the stored overall pass state and ordered passing, failing, and
undefined report-name lists.

The conversion preserves report order, every nested metric and difference-field
order, and all undefined analytical values as `None`. It rejects malformed or
internally inconsistent collection reports before conversion. Every call
returns fresh nested dictionaries and lists; the empty report becomes an empty
`named_reports` list plus the explicit non-passing verdict with empty category
lists. The API writes no JSON, performs no I/O or persistence, and recomputes no
analysis or verdict classification.

`campaign_projection_error_comparison_envelope_named_assessment_collection_records(
entries)` applies that existing collection-report converter to a finite ordered
set of frozen `CampaignProjectionErrorNamedAssessmentCollectionReport` entries.
Every entry requires a unique nonblank caller-supplied name and one existing
collection report.

The complete outer collection is materialized and its names and member types
are validated before conversion. The existing converter is then invoked
exactly once per entry in caller order, producing fresh plain `{name, report}`
dictionaries. Empty input returns `[]`; delegated structural failures propagate
without a partial result. All nested ordering, undefined `None` values, and
collection-verdict categories remain those of the existing converter. This API
performs no I/O, persistence, scoring, ranking, or analytical recomputation.

`campaign_projection_error_comparison_envelope_assessment_collection_verdict_overview(
entries)` extracts an intentionally compact overview from the same finite
ordered named collection reports. Each fresh plain dictionary contains exactly
`name` and the already-stored collection-verdict `overall_passed` boolean.

The complete outer collection is materialized and checked for unique nonblank
names and valid member types, then every nested collection report is fully
validated before extraction. Caller order is preserved and empty input returns
`[]`. No nested reports, verdict categories, metrics, fields, margins, extrema,
or residuals are included, and no verdict or analytical value is recomputed.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict(
entries)` reduces those validated named collection reports to one frozen
`CampaignProjectionErrorNamedAssessmentCollectionReportVerdict`. A collection
is undefined when its stored collection verdict contains an undefined report
name; otherwise its stored overall pass state determines whether it is passing
or failing. Undefined takes precedence over ordinary failure.

The result retains caller-ordered passing, failing, and undefined collection-
name tuples. It passes overall only for nonempty input when every collection is
defined and passing; empty input returns an explicit non-passing verdict with
empty categories. This API recomputes no nested verdict or analytical value
and infers no score, rank, confidence, probability, certification, or
acceptance policy.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_record(
verdict)` validates one existing named assessment-collection report verdict and
returns a fresh deterministic JSON-compatible dictionary. It contains the
stored `overall_passed` boolean and caller-ordered
`passing_collection_names`, `failing_collection_names`, and
`undefined_collection_names` lists.

The converter validates the verdict type, Boolean pass state, tuple category
structure, nonblank unique mutually exclusive names, and overall-pass
consistency. It does not recompute any classification. The empty non-passing
verdict becomes the same explicit schema with three empty lists, and every call
returns detached plain data without I/O or persistence.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_records(
entries)` applies that converter to a finite caller-ordered collection of
frozen `CampaignProjectionErrorNamedAssessmentCollectionVerdict` entries. Each
entry contains one unique nonblank name and one existing assessment-collection
report verdict.

The complete collection and all verdicts are validated before conversion. The
single-verdict record converter is then invoked exactly once per entry, yielding
fresh plain `{name, verdict}` dictionaries in caller order while preserving
every stored inner category order. Empty input returns `[]`; no classification,
pass state, or analytical value is recomputed.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_overview(
entries)` extracts an intentionally compact overview from the same validated
named verdict collection. Each fresh plain dictionary contains exactly `name`
and the verdict's stored `overall_passed` Boolean.

The complete input is materialized and every name and verdict is validated
before extraction. Caller order is preserved, empty input returns `[]`, and no
stored category names, classifications, pass states, or analytical values are
recomputed. Repeated calls return equal but fully detached lists and
dictionaries.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_collection_verdict(
entries)` reduces the same validated named verdicts to one frozen
`CampaignProjectionErrorNamedAssessmentCollectionVerdictCollectionVerdict`.
Each named verdict is undefined when its stored `undefined_collection_names`
tuple is nonempty; otherwise its stored overall pass state determines passing
or failing. Undefined takes precedence over ordinary failure.

The aggregate retains caller-ordered passing, failing, and undefined verdict-
name tuples and passes only for nonempty input when every verdict is defined
and passing. Empty input returns an explicit non-passing verdict with empty
categories. No nested classification, pass state, or analytical value is
recomputed.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_collection_verdict_record(
verdict)` validates one existing aggregate verdict and returns a fresh plain
dictionary containing its stored `overall_passed` Boolean and ordered
`passing_verdict_names`, `failing_verdict_names`, and
`undefined_verdict_names` lists.

The conversion validates the frozen verdict's type, Boolean state, tuple
categories, nonblank unique mutually exclusive names, and overall-pass
consistency. It invokes no aggregation or lower-level classification. The
empty non-passing verdict produces the same schema with three empty lists, and
every call returns equal but fully detached JSON-compatible data.

`campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_collection_verdict_records(
entries)` applies that existing converter exactly once to each member of a
finite ordered collection of frozen
`CampaignProjectionErrorNamedAssessmentCollectionAggregateVerdict` entries.
Every entry supplies one unique nonblank name and one existing aggregate
verdict.

The complete collection and all nested verdicts are validated before any
conversion. The returned fresh `{name, verdict}` dictionaries preserve caller
order and every stored inner category order, contain only JSON-compatible plain
values, and are detached from their sources. Empty input returns `[]`. This
completes the current aggregate-verdict record family without adding another
serialization or aggregation layer.

`campaign_projection_envelopes(scenario_results)` reduces one explicit finite
ordered scenario set to immutable per-metric predicted-change bounds:

```python
from flightlab.analysis import campaign_projection_envelopes

envelopes = campaign_projection_envelopes(results)
```

Each frozen `CampaignMetricProjectionEnvelope` retains the metric name, minimum
predicted change and attaining scenario name, and maximum predicted change and
attaining scenario name. Metrics remain in projection order. Only defined
finite predictions participate; `None` is never treated as zero. If every
scenario is undefined for a metric, both bounds and both scenario names are
`None`.

Exact ties select the first attaining scenario in caller scenario order. The
envelope therefore reports the best/worst predicted metric excursions across
the explicit finite scenario set, but does not rank scenarios globally or
claim probabilistic robustness. Empty scenario input returns `()`. The pure
analysis performs no projection recomputation, simulation, sampling,
optimization, plotting, or persistence.

`check_campaign_projection_envelope_limits(envelopes, limits)` checks those
ordered deterministic envelopes against explicit caller-defined allowable
predicted-change intervals. Each frozen `CampaignMetricProjectionLimit` names
one metric and its finite lower/upper bounds in exact envelope order.

For a defined envelope, the API computes:

```text
lower_margin = observed_minimum - allowable_lower
upper_margin = allowable_upper - observed_maximum
```

The frozen `CampaignMetricProjectionLimitResult` preserves metric order,
observed extrema, allowable limits, both margins, and `passed`. A metric passes
only when both margins are nonnegative; equality is a zero-margin pass. If the
envelope is undefined, observed extrema and margins remain `None` and the metric
does not pass. Missing information is never treated as zero or success.

This is a deterministic requirement/robustness check asking whether predicted
excursions from the explicit scenario set remain inside caller-supplied bounds.
It is not a probabilistic safety guarantee. Empty envelopes plus empty limits
return `()`. The API infers no limit and performs no projection recomputation,
ranking, probability modeling, simulation, or persistence.

`campaign_robustness_verdict(limit_results)` reduces an existing ordered set of
per-metric limit checks to one frozen `CampaignRobustnessVerdict`. It retains
`overall_passed` plus ordered tuples of passing, failing, and undefined metric
names. Defined results use their existing `passed` flag; undefined extrema and
margins are classified separately rather than as ordinary failures.

Overall pass is true only when at least one metric was checked and every metric
is defined and passing. Any failing or undefined metric makes it false. Empty
input returns an explicit non-passing verdict with all category tuples empty.
Before classification, the API validates exact relationships among observed
extrema, allowable bounds, stored margins, and pass state without recomputing
envelopes or projections.

This is a deterministic verdict over explicit scenario projections and
caller-defined limits. It is not a probabilistic certification, formal safety
proof, weighted score, or nonlinear robustness guarantee. No metric is ranked
or assigned an implicit priority.

An execution failure propagates unchanged and prevents all campaign
persistence. Any run, manifest, or membership failure propagates after
execution and rolls back every newly inserted campaign row. Existing records
remain unchanged. There are no partial campaign results, retries, automatic
case generation, parallel work, optimization, analysis, or CLI behavior.
