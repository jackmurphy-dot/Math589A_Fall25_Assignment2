# general_linear_solver.py
import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution_L(A: Array, P: Array, Q: Array, b: Array, r: int) -> np.ndarray:
    """Solve (L @ y = P@b) for y, where L has implicit 1s on diag."""
    m = len(P)
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]
    for i in range(m):
        s = 0.0
        for j in range(min(i, r)):
            s += A[P[i], Q[j]] * y[j]
        y[i] = b_perm[i] - s
    return y


def _back_substitution_U(A: Array, P: Array, Q: Array, y: Array, r: int, tol: float) -> np.ndarray:
    """Solve U x = y for leading r×r upper-triangular block."""
    x = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, r):
            s += A[P[i], Q[j]] * x[j]
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Zero pivot in U")
        x[i] = (y[i] - s) / piv
    return x


def _solve_nullspace(A: Array, P: Array, Q: Array, r: int, n: int, tol: float) -> np.ndarray:
    """Construct nullspace basis from U_bb."""
    k = max(n - r, 0)
    N = np.zeros((n, k), dtype=np.float64)
    if k == 0:
        return N
    if r == 0:
        N[:, :] = np.eye(n)
        return N
    for f in range(k):
        rhs = -A[P[:r], Q[r + f]]
        x_b = np.zeros(r)
        for i in range(r - 1, -1, -1):
            s = np.dot(A[P[i], Q[i + 1:r]], x_b[i + 1:r])
            piv = A[P[i], Q[i]]
            if abs(piv) < tol:
                raise np.linalg.LinAlgError("Singular U_bb")
            x_b[i] = (rhs[i] - s) / piv
        col = np.zeros(n)
        col[Q[:r]] = x_b
        col[Q[r + f]] = 1.0
        N[:, f] = col
    return N


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Solve A x = b  → (c, N)
      c : particular solution (None if inconsistent)
      N : nullspace basis
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    A_copy = np.array(A, dtype=np.float64, copy=True, order="C")
    A_fac, P, Q, _, r = paq_lu(A_copy, tol=tol)
    logger.debug(f"PAQ-LU rank={r}")

    # Forward substitution: L y = P b
    y = _forward_substitution_L(A_fac, P, Q, b, r)

    # Check consistency
    if r < m and np.max(np.abs(y[r:])) > tol:
        return None, _solve_nullspace(A_fac, P, Q, r, n, tol)

    # Particular solution: U x_b = y[:r]
    x_b = _back_substitution_U(A_fac, P, Q, y[:r], r, tol)
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b

    # Nullspace
    N = _solve_nullspace(A_fac, P, Q, r, n, tol)
    return c, N
