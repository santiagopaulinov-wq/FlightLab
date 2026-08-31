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
