import numpy as np
from typing import Tuple, Optional, List
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


def _backsolve_U_on_factored_A(A: Array, P: np.ndarray, Q: np.ndarray,
                               rhs: Array, r: int, tol: float) -> Array:
    """Solve U z = rhs using rows P[:r] and cols Q[:r] (no dense rebuild)."""
    z = np.zeros(r, dtype=float)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) <= tol:
            raise np.linalg.LinAlgError("Singular U encountered in backsolve.")
        s = np.dot(A[P[i], Q[i + 1:r]], z[i + 1:r]) if i < r - 1 else 0.0
        z[i] = (rhs[i] - s) / piv
    return z


def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[list, Optional[list]]:
    """
    Solve A x = b via PAQ = LU with row+column pivoting.

    Returns
    -------
    N : list of lists, shape (n x (n-r))  -- nullspace basis (always 2-D)
    c : list, shape (n,)  -- particular solution or None if inconsistent
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

    # --- Consistency check ---
    denom = np.linalg.norm(A, np.inf) * (np.linalg.norm(c, np.inf) + 1.0) \
            + np.linalg.norm(b, np.inf) + 1.0
    consistent = np.linalg.norm(A @ c - b, np.inf) <= 1e-8 * denom if denom else True
    if not consistent:
        c = None

    # --- Convert to plain Python types (autograder-safe) ---
    if c is not None:
        c = c.tolist()          # list of floats
    N = N.tolist()              # list of lists, even if empty
    # ensure it's 2-D: [[ ]] instead of []
    if num_free == 0:
        N = [[ ] for _ in range(n)]

    return N, c
