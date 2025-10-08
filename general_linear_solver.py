import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def forward_substitution(A, P, Q, b, r):
    """Solve L y = P b for y, where L has implicit unit diagonal."""
    m = len(P)
    y = np.zeros(m)
    b_perm = b[P]
    for i in range(r):
        y[i] = b_perm[i] - np.dot(A[P[i], Q[:i]], y[:i])
    return y

def backward_substitution(A, P, Q, y, r, tol=1e-10):
    """Solve U z = y for z (upper triangular system)."""
    z = np.zeros(r)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular matrix.")
        z[i] = (y[i] - np.dot(A[P[i], Q[i + 1:r]], z[i + 1:r])) / piv
    return z

def build_nullspace(A, P, Q, r, n, tol=1e-10):
    """Construct a nullspace basis for A."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0))

    N = np.zeros((n, num_free))
    # Free columns correspond to Q[r:]
    for j, free_col in enumerate(Q[r:]):
        e = np.zeros(n)
        e[free_col] = 1.0
        # Backward substitution for dependent vars
        z = -backward_substitution(A, P, Q, A[P[:r], free_col], r, tol)
        e[Q[:r]] = z
        N[:, j] = e
    return N

def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Optional[Array], Array]:
    """
    General linear solver using PLU decomposition.
    Returns (particular_solution, nullspace_basis).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol)

    # Forward substitution: solve L y = P b
    y = forward_substitution(A_fac, P, Q, b, r)

    # Compute particular solution
    if r > 0:
        z_basic = backward_substitution(A_fac, P, Q, y[:r], r, tol)
        c = np.zeros(n)
        c[Q[:r]] = z_basic
    else:
        c = np.zeros(n)

    # Nullspace basis
    N = build_nullspace(A_fac, P, Q, r, n, tol)

    # If non-singular (r == n), nullspace is empty
    if r == n:
        N = np.zeros((n, 0))

    return c, N
