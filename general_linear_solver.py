# general_linear_solver.py
# Solves A x = b using PAQ = LU decomposition.
# Returns (c, N) where:
#   - c is a particular solution (or None if inconsistent)
#   - N is a basis of the nullspace of A
# Uses in-place factorization on a private copy of A.

import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray


def _forward_substitution_L(A: Array, P: Array, Q: Array, b: Array, r: int) -> Array:
    """Solve L y = P b for y (L has implicit unit diagonal)."""
    m, n = A.shape
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]
    for i in range(m):
        s = 0.0
        upto = min(i, r)
        if upto > 0:
            ai = A[P[i]]
            for j in range(upto):
                s += ai[Q[j]] * y[j]
        y[i] = b_perm[i] - s
    return y


def _back_substitution_U_basic(A: Array, P: Array, Q: Array, y: Array, r: int, tol: float) -> Array:
    """Solve U_bb x_b = y_b for the leading r×r upper-triangular block."""
    x_b = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        api = A[P[i]]
        for j in range(i + 1, r):
            s += api[Q[j]] * x_b[j]
        Uii = A[P[i], Q[i]]
        if abs(Uii) < tol:
            raise np.linalg.LinAlgError("Singular U block encountered.")
        x_b[i] = (y[i] - s) / Uii
    return x_b


def _solve_Ubb(A: Array, P: Array, Q: Array, rhs: Array, r: int, tol: float) -> Array:
    """Helper for nullspace basis: solve U_bb x = rhs."""
    x = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        api = A[P[i]]
        for j in range(i + 1, r):
            s += api[Q[j]] * x[j]
        Uii = A[P[i], Q[i]]
        if abs(Uii) < tol:
            raise np.linalg.LinAlgError("Singular U block in nullspace solve.")
        x[i] = (rhs[i] - s) / Uii
    return x


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b returning (c, N)
      c : particular solution or None if inconsistent
      N : nullspace basis (n×k matrix)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    # Work on a private copy to preserve caller’s A
    A_work = np.array(A, dtype=np.float64, order="C", copy=True)

    # Factorization
    A_fac, P, Q, pivot_cols, r = paq_lu(A_work, tol=tol)
    logger.debug(f"LU factorization complete: rank={r}")

    # Forward substitution
    y = _forward_substitution_L(A_fac, P, Q, b, r)

    # Consistency check
    consistent = True
    if r < m and np.max(np.abs(y[r:])) > tol:
        consistent = False

    # Build nullspace basis N
    k = max(n - r, 0)
    N = np.zeros((n, k), dtype=np.float64)
    if k > 0 and r > 0:
        for f in range(k):
            rhs = np.zeros(r, dtype=np.float64)
            for i in range(r):
                rhs[i] = -A_fac[P[i], Q[r + f]]
            x_b = _solve_Ubb(A_fac, P, Q, rhs, r, tol)
            col = np.zeros(n, dtype=np.float64)
            col[Q[:r]] = x_b
            col[Q[r + f]] = 1.0
            N[:, f] = col
    elif k > 0 and r == 0:
        N[:, :] = np.eye(n, dtype=np.float64)

    # Particular solution
    if not consistent:
        c = None
        return c, N

    if r == 0:
        c = np.zeros(n, dtype=np.float64)
        return c, N

    y_b = y[:r].copy()
    x_b = _back_substitution_U_basic(A_fac, P, Q, y_b, r, tol)

    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b
    return c, N
