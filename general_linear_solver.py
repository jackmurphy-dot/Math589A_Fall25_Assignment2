import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def forward_substitution(A, P, Q, b, r):
    """Solves L y = P b, where L has an implicit unit diagonal."""
    m = len(P)
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]
    for i in range(m):
        # Dot product for the sum part of the substitution
        s = np.dot(A[P[i], Q[:min(i, r)]], y[:min(i, r)])
        y[i] = b_perm[i] - s
    return y

def backward_substitution(A, P, Q, y, r, tol):
    """Solves U_basic z_basic = y_basic for the basic variables."""
    z_basic = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = np.dot(A[P[i], Q[i + 1:r]], z_basic[i + 1:r])
        pivot = A[P[i], Q[i]]
        if abs(pivot) < tol:
            raise np.linalg.LinAlgError("Singular pivot encountered in U.")
        z_basic[i] = (y[i] - s) / pivot
    return z_basic

def build_nullspace(A, P, Q, r, n, tol):
    """Constructs a basis for the nullspace of A."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0), dtype=np.float64)

    N = np.zeros((n, num_free), dtype=np.float64)
    for f in range(num_free):
        # For each free variable, find the corresponding nullspace vector
        # by solving U_basic * z_basic = -U_free * e_f
        rhs = -A[P[:r], Q[r + f]]
        
        # Solve for the basic variables of the nullspace vector via back substitution
        z_basic = np.zeros(r, dtype=np.float64)
        for i in range(r - 1, -1, -1):
            s = np.dot(A[P[i], Q[i + 1:r]], z_basic[i + 1:r])
            pivot = A[P[i], Q[i]]
            if abs(pivot) < tol:
                raise np.linalg.LinAlgError("Singular U block in nullspace calculation.")
            z_basic[i] = (rhs[i] - s) / pivot

        # Assemble the full nullspace vector x
        x_null = np.zeros(n, dtype=np.float64)
        x_null[Q[:r]] = z_basic      # Set basic variables
        x_null[Q[r + f]] = 1.0     # Set the f-th free variable to 1
        N[:, f] = x_null
    return N

def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solves the linear system A x = b for x.

    Returns a tuple (c, N) where:
      - c is a particular solution, or None if the system is inconsistent.
      - N is a matrix whose columns form a basis for the nullspace of A.
    The general solution is x = c + N @ z, where z is any vector of free parameters.
    """
    A_in = np.asarray(A, dtype=np.float64)
    b_in = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A_in.shape
    if b_in.shape[0] != m:
        raise ValueError("Dimension mismatch between A and b")

    # 1. Decompose the matrix: PAQ = LU
    A_fac, P, Q, _, r = paq_lu(A_in, tol=tol)

    # 2. Solve Ly = Pb for y using forward substitution
    y = forward_substitution(A_fac, P, Q, b_in, r)

    # 3. Check for consistency. If y has non-zero elements past rank r, no solution exists.
    if r < m and np.any(np.abs(y[r:]) > tol):
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N

    # 4. Solve U_basic * z_basic = y[:r] for the particular solution's basic variables
    z_basic = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    # 5. Assemble the full particular solution vector c (free variables are zero)
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = z_basic

    # 6. Find the basis for the nullspace
    N = build_nullspace(A_fac, P, Q, r, n, tol)
    
    return c, N
