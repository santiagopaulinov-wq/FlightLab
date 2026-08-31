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
