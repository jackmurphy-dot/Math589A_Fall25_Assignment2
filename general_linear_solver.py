import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution(A: Array, P: np.ndarray, Q: np.ndarray, b: Array, r: int) -> Array:
    """
    Solve L y = P b for first r entries (L has implicit unit diagonal).
    L[i, j] for j<i is stored in A at (P[i], Q[j]).
    """
    y = np.zeros(len(P), dtype=float)
    bp = b[P]
    for i in range(r):
        if i > 0:
            y[i] = bp[i] - np.dot(A[P[i], Q[:i]], y[:i])
        else:
            y[i] = bp[i]
    return y

def _backsolve_U_on_factored_A(A: Array, P: np.ndarray, Q: np.ndarray, rhs: Array, r: int, tol: float) -> Array:
    """
    Solve U z = rhs where U is the r x r upper-triangular block read from A
    using rows P[:r] and columns Q[:r] (no dense reconstruction of U_B).
    """
    z = np.zeros(r, dtype=float)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular U encountered in backsolve.")
        if i < r - 1:
            rhs_i = rhs[i] - np.dot(A[P[i], Q[i+1:r]], z[i+1:r])
        else:
            rhs_i = rhs[i]
        z[i] = rhs_i / piv
    return z

def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b via PAQ = LU with full pivoting.
    Returns:
        c : particular solution (n,), free variables set to 0
        N : nullspace basis (n x (n - r))
    Implementation mirrors derivation.tex, but *does not* rebuild dense U_B/U_F.
    Instead, it backsolves directly against the factored A with P,Q for stability.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol=tol)

    # 1) L y = P b, only first r entries matter downstream
    y = _forward_substitution(A_fac, P, Q, b, r)

    # 2) Particular solution: U z = y[:r], then place z into the pivot variables
    c = np.zeros(n, dtype=float)
    if r > 0:
        z_basic = _backsolve_U_on_factored_A(A_fac, P, Q, y[:r], r, tol)
        c[Q[:r]] = z_basic  # x = Q [z; 0]

    # 3) Nullspace: for each free column j, solve U z = U(:, free_j) on pivot rows, then top is -z, bottom is e_j
    num_free = n - r
    if num_free > 0:
        N = np.zeros((n, num_free), dtype=float)
        for j in range(num_free):
            col_idx = Q[r + j]              # physical column index of the j-th free var
            rhs = A_fac[P[:r], col_idx]     # this is U(:, free) restricted to pivot rows
            z = _backsolve_U_on_factored_A(A_fac, P, Q, rhs, r, tol)
            # Place into N in original variable order:
            N[Q[:r], j] = -z
            N[col_idx, j] = 1.0
    else:
        N = np.zeros((n, 0), dtype=float)

    return c, N
