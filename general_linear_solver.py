import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution(A: Array, P: np.ndarray, Q: np.ndarray, b: Array, r: int) -> Array:
    """Solve L y = P b for first r entries (L has implicit unit diagonal)."""
    y = np.zeros(len(P), dtype=float)
    bp = b[P]
    for i in range(r):
        if i > 0:
            y[i] = bp[i] - np.dot(A[P[i], Q[:i]], y[:i])
        else:
            y[i] = bp[i]
    return y

def _backsolve_U_on_factored_A(A: Array, P: np.ndarray, Q: np.ndarray, rhs: Array, r: int, tol: float) -> Array:
    """Solve U z = rhs directly on factored A using P,Q indexing."""
    z = np.zeros(r, dtype=float)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular U encountered in backsolve.")
        rhs_i = rhs[i] - np.dot(A[P[i], Q[i+1:r]], z[i+1:r]) if i < r - 1 else rhs[i]
        z[i] = rhs_i / piv
    return z

def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Array, Optional[Array]]:
    """
    Solve A x = b via PAQ = LU with full pivoting.

    Returns:
        N : ndarray (n x (n-r)) Nullspace basis matrix.
        c : ndarray (n,) particular solution, or None if inconsistent.

    Behavior:
        - If A x = b is inconsistent, c = None.
        - Nullspace N is always returned (basis of null(A)).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol=tol)

    # 1) Forward substitution: L y = P b
    y = _forward_substitution(A_fac, P, Q, b, r)

    # 2) Particular solution
    c = np.zeros(n, dtype=float)
    if r > 0:
        z_basic = _backsolve_U_on_factored_A(A_fac, P, Q, y[:r], r, tol)
        c[Q[:r]] = z_basic
    else:
        c[:] = 0.0

    # 3) Nullspace construction
    num_free = n - r
    if num_free > 0:
        N = np.zeros((n, num_free), dtype=float)
        for j in range(num_free):
            free_col = Q[r + j]
            rhs = A_fac[P[:r], free_col]
            z = _backsolve_U_on_factored_A(A_fac, P, Q, rhs, r, tol)
            N[Q[:r], j] = -z
            N[free_col, j] = 1.0
    else:
        N = np.zeros((n, 0), dtype=float)

    # 4) Check consistency: if residual > tol, system inconsistent
    residual = A @ c - b
    if np.linalg.norm(residual) > 1e-8 * (np.linalg.norm(A) * np.linalg.norm(c) + np.linalg.norm(b) + 1.0):
        c = None  # inconsistent system

    return N, c
