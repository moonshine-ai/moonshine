# Eigen

A C++ template library for linear algebra: matrices, vectors, numerical
solvers, and related algorithms.

- **Upstream:** <https://gitlab.com/libeigen/eigen>
- **Version:** 3.4.0
- **License:** MPL-2.0 (see `COPYING.MPL2`)

This is an MPL-2.0-only subset of upstream Eigen 3.4.0. All sparse-matrix
modules, ordering methods, iterative sparse solvers, and third-party sparse
solver wrappers (CHOLMOD, METIS, SuperLU, UMFPACK, KLU, SuiteSparseQR, Pardiso,
PaStiX) have been removed because they include LGPL- or GPL-licensed code.

Moonshine only needs the Dense stack (`Core`, `LU`, `Cholesky`, `QR`, `SVD`,
`Geometry`, `Eigenvalues`, `Jacobi`, `Householder`). cpp-annote compiles with
`EIGEN_MPL2_ONLY=1`.
