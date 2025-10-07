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
    """Solve U x_b = y_b (upper-triangular)."""
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

    # Extract the U_basic and U_free submatrices from the factored matrix A
    U_basic = np.triu(A[np.ix_(P[:r], Q[:r])])
    U_free = A[np.ix_(P[:r], Q[r:])]

    # The nullspace vectors in the permuted space are found by solving
    # U_basic * N_basic = -U_free
    if np.min(np.abs(np.diag(U_basic))) < tol:
        raise np.linalg.LinAlgError("Singular U_basic matrix in nullspace.")
    
    N_basic = -np.linalg.inv(U_basic) @ U_free
    
    # Assemble the full nullspace matrix N in the original coordinate system
    N = np.zeros((n, num_free))
    N[Q[:r], :] = N_basic         # Basic variable rows
    N[Q[r:], :] = np.eye(num_free) # Free variable rows (identity matrix)
    
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
