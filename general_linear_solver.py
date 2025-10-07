# general_linear_solver.py
# Solve A x = b using PAQ = LU factorization.
# Returns (c, N) with c = particular solution or None, N = nullspace basis.

import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution_L(A: Array, P: Array, Q: Array, b: Array, r: int) -> Array:
    """Solve L y = P b (L unit lower)."""
    m, _ = A.shape
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]
    for i in range(m):
        s = 0.0
        for j in range(min(i, r)):
            s += A[P[i], Q[j]] * y[j]
        y[i] = b_perm[i] - s
    return y


def _back_substitution_U(A: Array, P: Array, Q: Array, y: Array, r: int, tol: float) -> Array:
    """Solve U_bb x_b = y_b (U upper)."""
    x = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, r):
            s += A[P[i], Q[j]] * x[j]
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Near-singular pivot.")
        x[i] = (y[i] - s) / piv
    return x


def _solve_Ubb(A: Array, P: Array, Q: Array, rhs: Array, r: int, tol: float) -> Array:
    """Helper: solve U_bb x = rhs for nullspace computation."""
    x = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, r):
            s += A[P[i], Q[j]] * x[j]
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular U block.")
        x[i] = (rhs[i] - s) / piv
    return x


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b  →  (c, N)
      c : particular solution (None if inconsistent)
      N : nullspace basis (n×k)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b.")

    # Factor on a private copy
    A_work = np.array(A, dtype=np.float64, copy=True, order="C")
    A_fac, P, Q, pivot_cols, r = paq_lu(A_work, tol=tol)
    logger.debug(f"PAQ-LU complete, rank={r}")

    # Forward solve
    y = _forward_substitution_L(A_fac, P, Q, b, r)

    # Check consistency: rows beyond rank should have zero residual
    consistent = True
    if r < m and np.max(np.abs(y[r:])) > tol:
        consistent = False

    # Nullspace basis
    k = max(n - r, 0)
    N = np.zeros((n, k), dtype=np.float64)
    if k > 0:
        if r > 0:
            for f in range(k):
                rhs = -A_fac[P[:r], Q[r + f]]
                x_b = _solve_Ubb(A_fac, P, Q, rhs, r, tol)
                col = np.zeros(n, dtype=np.float64)
                col[Q[:r]] = x_b
                col[Q[r + f]] = 1.0
                N[:, f] = col
        else:
            N[:, :] = np.eye(n)

    if not consistent:
        return None, N

    # Particular solution
    if r == 0:
        return np.zeros(n, dtype=np.float64), N

    y_b = y[:r]
    x_b = _back_substitution_U(A_fac, P, Q, y_b, r, tol)
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b
    return c, N
