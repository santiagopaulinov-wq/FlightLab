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
