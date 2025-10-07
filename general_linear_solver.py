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
        return np.eye(n, k, dtype=np.float64)

    for f in range(k):
        # For each free variable, solve for the basic variables
        rhs = -A[P[:r], Q[r + f]]
        x_b = np.zeros(r, dtype=np.float64)

        # Perform back-substitution to find the basic variables of the null space vector
        for i in range(r - 1, -1, -1):
            s = 0.0
            for j in range(i + 1, r):
                s += A[P[i], Q[j]] * x_b[j]
            
            piv = A[P[i], Q[i]]
            if abs(piv) < tol:
                raise np.linalg.LinAlgError("Singular U block in nullspace calculation.")
            x_b[i] = (rhs[i] - s) / piv
        
        # Assemble the full null space vector
        col = np.zeros(n, dtype=np.float64)
        col[Q[:r]] = x_b      # Set basic variables
        col[Q[r + f]] = 1.0  # Set the current free variable to 1
        N[:, f] = col
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

    # The PAQ=LU function works on a copy
    A_fac, P, Q, _, r = paq_lu(A_in, tol=tol)

    # Step 1: Solve Ly = Pb
    y = forward_substitution(A_fac, P, Q, b_in, r)

    # Step 2: Check for consistency
    if r < m and np.any(np.abs(y[r:]) > tol):
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N

    # Step 3: Solve U_basic * z_basic = y_basic for the particular solution
    z_basic = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    # Step 4: Reconstruct the full particular solution vector c from z_basic
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = z_basic

    # Step 5: Find the nullspace basis
    N = build_nullspace(A_fac, P, Q, r, n, tol)
    
    return c, N
