import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution(L: Array, b_perm: Array) -> Array:
    """Solves Ly = b_perm for y, where L is an explicit matrix."""
    r = L.shape[1]
    y = np.zeros(L.shape[0], dtype=float)
    for i in range(r):
        s = np.dot(L[i, :i], y[:i])
        # L has unit diagonal, so L[i, i] is 1
        y[i] = b_perm[i] - s
    # Propagate the rest of b_perm for the consistency check
    if len(b_perm) > r:
        y[r:] = b_perm[r:]
    return y

def _backward_substitution(U: Array, y: Array, tol: float) -> Array:
    """Solves Uz = y for z."""
    r, n = U.shape
    z = np.zeros(n, dtype=float)
    for i in range(r - 1, -1, -1):
        s = np.dot(U[i, i + 1:], z[i + 1:])
        pivot = U[i, i]
        if abs(pivot) < tol:
            raise np.linalg.LinAlgError("Singular U matrix.")
        z[i] = (y[i] - s) / pivot
    return z

def _build_nullspace(U: Array, Q: Array, r: int, n: int, tol: float) -> Array:
    """Constructs the nullspace for Ax=0, which is equivalent to Uz=0."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0), dtype=float)

    N_permuted = np.zeros((n, num_free), dtype=float)
    for i in range(num_free):
        # Set one free variable to 1, others to 0
        z_free = np.zeros(num_free)
        z_free[i] = 1.0
        
        # Solve U_basic * z_basic = -U_free * z_free
        rhs = -U[:r, r:] @ z_free
        
        # Perform back substitution to get z_basic
        z_basic = np.zeros(r, dtype=float)
        for k in range(r - 1, -1, -1):
            s = np.dot(U[k, k + 1:r], z_basic[k + 1:r])
            z_basic[k] = (rhs[k] - s) / U[k, k]
            
        N_permuted[:r, i] = z_basic
        N_permuted[r:, i] = z_free
        
    # Un-permute the columns to get the final nullspace
    N = np.zeros_like(N_permuted)
    N[Q, :] = N_permuted
    return N

def solve(A: Array, b: Array, tol: float = 1e-12) -> Tuple[Optional[Array], Array]:
    """Solves the linear system A x = b."""
    A_in = np.asarray(A, dtype=float)
    b_in = np.asarray(b, dtype=float).reshape(-1)
    m, n = A_in.shape

    # 1. Decompose into P, Q, L, U
    P, Q, L, U, r = paq_lu(A_in, tol=tol)

    # 2. Permute b and solve Ly = Pb
    b_permuted = b_in[P]
    y = _forward_substitution(L, b_permuted)

    # 3. Check for consistency
    if r < m and np.any(np.abs(y[r:]) > tol):
        N = _build_nullspace(U, Q, r, n, tol)
        return None, N

    # 4. Solve Uz = y for the particular solution (free vars are 0)
    z = _backward_substitution(U, y[:r], tol)

    # 5. Un-permute z to get the final solution c
    c = np.zeros(n, dtype=float)
    c[Q] = z

    # 6. Build the nullspace
    N = _build_nullspace(U, Q, r, n, tol)
    
    return c, N
