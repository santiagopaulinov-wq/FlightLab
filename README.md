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
