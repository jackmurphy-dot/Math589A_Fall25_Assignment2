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
        s = 0.0
        for j in range(min(i, r)):
            s += A[P[i], Q[j]] * y[j]
        y[i] = b_perm[i] - s
    return y


def backward_substitution(A, P, Q, y, r, tol):
    """Solve U x_b = y_b (upper-triangular)."""
    x_b = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, r):
            s += A[P[i], Q[j]] * x_b[j]
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular pivot.")
        x_b[i] = (y[i] - s) / piv
    return x_b


def build_nullspace(A, P, Q, r, n, tol):
    """Construct basis of the nullspace of A."""
    k = max(n - r, 0)
    N = np.zeros((n, k), dtype=np.float64)
    if k == 0:
        return N
    if r == 0:
        N[:, :] = np.eye(n)
        return N
    for f in range(k):
        rhs = -A[P[:r], Q[r + f]]
        x_b = np.zeros(r, dtype=np.float64)
        for i in range(r - 1, -1, -1):
            s = np.dot(A[P[i], Q[i + 1:r]], x_b[i + 1:r])
            piv = A[P[i], Q[i]]
            if abs(piv) < tol:
                raise np.linalg.LinAlgError("Singular U block.")
            x_b[i] = (rhs[i] - s) / piv
        col = np.zeros(n, dtype=np.float64)
        col[Q[:r]] = x_b
        col[Q[r + f]] = 1.0
        N[:, f] = col
    return N


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b returning (c, N):
      - c: particular solution (or None if inconsistent)
      - N: nullspace basis
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    # Copy to preserve A
    A_copy = np.array(A, dtype=np.float64, copy=True, order="C")
    A_fac, P, Q, _, r = paq_lu(A_copy, tol=tol)

    # Forward solve: L y = P b
    y = forward_substitution(A_fac, P, Q, b, r)

    # Check consistency (rows > rank should give 0)
    if r < m and np.max(np.abs(y[r:])) > tol:
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N

    # Backward solve: U x_b = y[:r]
    x_b = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    # Recover full solution: x = Q ( [x_b; 0] )
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b

    # Nullspace basis
    N = build_nullspace(A_fac, P, Q, r, n, tol)
    return c, N
