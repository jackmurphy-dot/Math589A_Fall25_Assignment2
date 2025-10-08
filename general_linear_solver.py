import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray


def _forward_substitution(A: Array, P: np.ndarray, Q: np.ndarray, b: Array, r: int) -> Array:
    """Solve L y = P b for first r entries (L has implicit unit diag)."""
    y = np.zeros(len(P), dtype=float)
    bp = b[P]
    for i in range(r):
        y[i] = bp[i] - (np.dot(A[P[i], Q[:i]], y[:i]) if i > 0 else 0.0)
    return y


def _backsolve_U_on_factored_A(A: Array, P: np.ndarray, Q: np.ndarray, rhs: Array, r: int, tol: float) -> Array:
    """Solve U z = rhs using rows P[:r] and cols Q[:r] (no dense rebuild)."""
    z = np.zeros(r, dtype=float)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) <= tol:
            raise np.linalg.LinAlgError("Singular U encountered in backsolve.")
        s = np.dot(A[P[i], Q[i + 1:r]], z[i + 1:r]) if i < r - 1 else 0.0
        z[i] = (rhs[i] - s) / piv
    return z


def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Array, Optional[Array]]:
    """
    Solve A x = b via PAQ = LU with row+column pivoting.

    Returns:
        N : (n x (n-r)) nullspace basis (always 2-D)
        c : (n,) particular solution with free vars = 0, or None if inconsistent
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol=tol)

    # 1) Forward: L y = P b
    y = _forward_substitution(A_fac, P, Q, b, r)

    # 2) Particular solution
    c = np.zeros(n, dtype=float)
    if r > 0:
        z_basic = _backsolve_U_on_factored_A(A_fac, P, Q, y[:r], r, tol)
        c[Q[:r]] = z_basic

    # 3) Nullspace basis
    num_free = n - r
    N = np.zeros((n, num_free), dtype=float)
    for j in range(num_free):
        free_col = Q[r + j]
        rhs = A_fac[P[:r], free_col]
        z = _backsolve_U_on_factored_A(A_fac, P, Q, rhs, r, tol) if r > 0 else np.zeros(0)
        N[Q[:r], j] = -z
        N[free_col, j] = 1.0

    # --- Enforce N as 2-D (even if nullspace is empty) ---
    N = np.asarray(N, dtype=float)
    if N.ndim == 1:
        # happens if n==r and N collapses to 1D
        N = N.reshape(n, 0)
    elif N.size == 0 or N.shape[1] != num_free or N.shape[0] != n:
        N = np.zeros((n, max(num_free, 0)), dtype=float)

    # 4) Consistency check
    denom = np.linalg.norm(A, ord=np.inf) * (np.linalg.norm(c, ord=np.inf) + 1.0) \
            + np.linalg.norm(b, ord=np.inf) + 1.0
    consistent = np.linalg.norm(A @ c - b, ord=np.inf) <= 1e-8 * denom if denom else True
    if not consistent:
        c = None

    return N, c
