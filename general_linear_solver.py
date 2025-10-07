import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def forward_substitution(A, P, Q, b, r):
    """Solve L y = P b where L has implicit 1s on diagonal."""
    m = len(P)
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]
    for i in range(m):
        s = np.dot(A[P[i], Q[:min(i, r)]], y[:min(i, r)])
        y[i] = b_perm[i] - s
    return y


def backward_substitution(A, P, Q, y, r, tol):
    """Solve U_basic x_basic = y_basic (upper-triangular)."""
    x_b = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = np.dot(A[P[i], Q[i + 1:r]], x_b[i + 1:r])
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular pivot.")
        x_b[i] = (y[i] - s) / piv
    return x_b


def build_nullspace(A, P, Q, r, n, tol):
    """Construct basis of the nullspace of A."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0), dtype=np.float64)
    
    # THE FIX: This handles the r=0 case correctly and robustly.
    if r == 0:
        return np.eye(n, num_free, dtype=np.float64)

    # U_basic is A[P[:r], Q[:r]], U_free is A[P[:r], Q[r:]]
    U_basic_inv = np.linalg.inv(np.triu(A[np.ix_(P[:r], Q[:r])]))
    U_free = A[np.ix_(P[:r], Q[r:])]
    
    # The core of the nullspace is the solution to U_basic * z_basic = -U_free
    N_permuted = -U_basic_inv @ U_free
    
    # Assemble the full nullspace matrix N
    N = np.zeros((n, num_free))
    N[Q[:r], :] = N_permuted
    N[Q[r:], :] = np.eye(num_free)
    
    return N


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b returning (c, N):
      - c: particular solution (or None if inconsistent)
      - N: nullspace basis
    """
    A_in = np.asarray(A, dtype=np.float64)
    b_in = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A_in.shape
    if b_in.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    A_fac, P, Q, _, r = paq_lu(A_in, tol=tol)

    y = forward_substitution(A_fac, P, Q, b_in, r)

    if r < m and np.any(np.abs(y[r:]) > tol):
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N

    z_basic = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = z_basic

    N = build_nullspace(A_fac, P, Q, r, n, tol)
    
    return c, N
