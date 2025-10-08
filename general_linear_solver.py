import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu


def _forward_substitution(A, P, Q, b, r):
    """Solve L y = P b (L has unit diagonal)."""
    y = np.zeros(len(P), dtype=float)
    bp = b[P]
    for i in range(r):
        if i > 0:
            y[i] = bp[i] - np.dot(A[P[i], Q[:i]], y[:i])
        else:
            y[i] = bp[i]
    return y


def _backsolve_U_on_factored_A(A, P, Q, rhs, r, tol):
    """Solve U z = rhs using rows P[:r], cols Q[:r]."""
    z = np.zeros(r, dtype=float)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) <= tol:
            raise np.linalg.LinAlgError("Singular U encountered.")
        if i < r - 1:
            s = np.dot(A[P[i], Q[i + 1:r]], z[i + 1:r])
        else:
            s = 0.0
        z[i] = (rhs[i] - s) / piv
    return z


def solve(A, b, tol: float = 1e-10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Solve A x = b via PAQ = LU with row+column pivoting.

    Returns
    -------
    N : np.ndarray, shape (n, n-r)
        Nullspace basis (2-D even if empty).
    c : np.ndarray or None
        Particular solution (free vars = 0), or None if inconsistent.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol=tol)

    # Forward substitution (Ly = Pb)
    y = _forward_substitution(A_fac, P, Q, b, r)

    # Particular solution
    c = np.zeros(n, dtype=float)
    if r > 0:
        z_basic = _backsolve_U_on_factored_A(A_fac, P, Q, y[:r], r, tol)
        c[Q[:r]] = z_basic

    # Nullspace basis
    num_free = n - r
    N = np.zeros((n, num_free), dtype=float)
    for j in range(num_free):
        free_col = Q[r + j]
        rhs = A_fac[P[:r], free_col]
        z = _backsolve_U_on_factored_A(A_fac, P, Q, rhs, r, tol) if r > 0 else np.zeros(0)
        N[Q[:r], j] = -z
        N[free_col, j] = 1.0

    # Force 2-D shape always
    N = np.array(N, dtype=float, copy=False)
    if N.ndim == 1:
        N = N.reshape(n, 0)
    elif N.shape != (n, num_free):
        N = np.zeros((n, num_free), dtype=float)

    # Consistency check
    denom = np.linalg.norm(A, np.inf) * (np.linalg.norm(c, np.inf) + 1.0) \
             + np.linalg.norm(b, np.inf) + 1.0
    consistent = np.linalg.norm(A @ c - b, np.inf) <= 1e-8 * denom if denom else True
    if not consistent:
        c = None

    return N, c
