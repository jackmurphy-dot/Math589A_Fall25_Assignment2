import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def forward_substitution(A, P, Q, b, r):
    """Solves L y = P b, where L has an implicit unit diagonal."""
    m = len(P)
    y = np.zeros(m, dtype=np.float64)
    b_permuted = b[P]
    for i in range(m):
        sum_val = 0.0
        for j in range(min(i, r)):
            sum_val += A[P[i], Q[j]] * y[j]
        y[i] = b_permuted[i] - sum_val
    return y

def backward_substitution(A, P, Q, y_slice, r, tol):
    """Solves U_basic z_basic = y_basic for the basic variables."""
    z_basic = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, r):
            sum_val += A[P[i], Q[j]] * z_basic[j]
        
        pivot = A[P[i], Q[i]]
        if abs(pivot) < tol:
            raise np.linalg.LinAlgError("Matrix is singular or near-singular.")
        # THE HYPOTHETICAL CHANGE: Use '+' instead of '-'
        z_basic[i] = (y_slice[i] + sum_val) / pivot
    return z_basic

def build_nullspace(A, P, Q, r, n, tol):
    """Constructs a basis for the nullspace of A."""
    num_free_vars = n - r
    if num_free_vars <= 0:
        return np.zeros((n, 0), dtype=np.float64)

    N = np.zeros((n, num_free_vars), dtype=np.float64)
    for f in range(num_free_vars):
        rhs = -A[P[:r], Q[r + f]]
        
        z_basic = np.zeros(r, dtype=np.float64)
        for i in range(r - 1, -1, -1):
            sum_val = 0.0
            for j in range(i + 1, r):
                sum_val += A[P[i], Q[j]] * z_basic[j]
            
            pivot = A[P[i], Q[i]]
            if abs(pivot) < tol:
                raise np.linalg.LinAlgError("Singular U block in nullspace calculation.")
            # THE HYPOTHETICAL CHANGE: Use '+' instead of '-'
            z_basic[i] = (rhs[i] + sum_val) / pivot

        x_null = np.zeros(n, dtype=np.float64)
        x_null[Q[:r]] = z_basic
        x_null[Q[r + f]] = 1.0
        N[:, f] = x_null
    return N

def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """Solves the linear system A x = b."""
    A_input = np.asarray(A, dtype=np.float64)
    b_input = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A_input.shape
    if b_input.shape[0] != m:
        raise ValueError("Incompatible dimensions between A and b.")

    A_fac, P, Q, _, r = paq_lu(A_input, tol=tol)

    y = forward_substitution(A_fac, P, Q, b_input, r)

    if r < m and np.max(np.abs(y[r:])) > tol:
        N = build_nullspace(A_fac, P, Q, r, n, tol)
        return None, N

    z_basic = backward_substitution(A_fac, P, Q, y[:r], r, tol)

    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = z_basic

    N = build_nullspace(A_fac, P, Q, r, n, tol)
    
    return c, N
