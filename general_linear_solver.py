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
        # The sum is over j < i and j < r
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
            raise np.linalg.LinAlgError("Singular pivot encountered in U.")
        x_b[i] = (y[i] - s) / piv
    return x_b


def build_nullspace(A, P, Q, r, n, tol):
    """Construct a basis for the nullspace of A."""
    num_free_vars = n - r
    if num_free_vars <= 0:
        return np.zeros((n, 0), dtype=np.float64)

    N = np.zeros((n, num_free_vars), dtype=np.float64)
    
    # Each column of N corresponds to setting one free variable to 1 and others to 0.
    for k in range(num_free_vars):
        free_col_idx = r + k
        # Right-hand side for the back substitution is -U_free's k-th column
        rhs = -A[P[:r], Q[free_col_idx]]
        
        # Solve U_basic * x_basic = rhs
        x_b = np.zeros(r, dtype=np.float64)
        for i in range(r - 1, -1, -1):
            s = np.dot(A[P[i], Q[i + 1:r]], x_b[i + 1:r])
            piv = A[P[i], Q[i]]
            if abs(piv) < tol:
                raise np.linalg.LinAlgError("Singular U block in nullspace calculation.")
            x_b[i] = (rhs[i] - s) / piv

        # Assemble the full nullspace vector
        null_vec = np.zeros(n, dtype=np.float64)
        null_vec[Q[:r]] = x_b          # Basic variables
        null_vec[Q[free_col_idx]] = 1.0  # One free variable set to 1
        N[:, k] = null_vec
        
    return N


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b returning (c, N):
      - c: particular solution (or None if inconsistent)
      - N: nullspace basis matrix
    """
    A_in = np.asarray(A, dtype=np.float64)
    b_in = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A_in.shape
    if b_in.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    # 1. Compute PAQ = LU decomposition
    A_fac, P, Q, _, r = paq_lu(A_in, tol=tol)

    # 2. Solve Ly = Pb using forward substitution
    y = forward_substitution(A_fac, P, Q, b_in, r)

    # 3. Check for consistency
    if r < m and np.any(np.abs(y[r:]) > tol):
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N # System is inconsistent

    # 4. Solve U_basic * x_basic = y_basic using backward substitution
    # This gives the basic variables for the particular solution (free variables are 0)
    x_b = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    # 5. Construct the particular solution vector `c`
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b

    # 6. Construct the nullspace basis `N`
    N = build_nullspace(A_fac, P, Q, r, n, tol)
    
    return c, N
