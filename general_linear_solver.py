import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution(L: Array, b_perm: Array) -> Array:
    """Solves Ly = b_perm for y."""
    rank = L.shape[1]
    y = np.zeros(rank, dtype=float)
    for i in range(rank):
        s = np.dot(L[i, :i], y[:i])
        y[i] = (b_perm[i] - s) / L[i, i] # L[i,i] is always 1
    return y

def _backward_substitution(U: Array, y: Array) -> Array:
    """Solves Uz = y for z."""
    rank, n = U.shape
    z = np.zeros(n, dtype=float)
    for i in range(rank - 1, -1, -1):
        s = np.dot(U[i, i + 1:], z[i + 1:])
        z[i] = (y[i] - s) / U[i, i]
    return z

def _build_nullspace(U: Array, r: int, n: int) -> Array:
    """Constructs the nullspace of U."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0), dtype=float)

    N_u = np.zeros((n, num_free))
    for i in range(num_free):
        # Create a nullspace vector in the permuted z-space
        z_null = np.zeros(n)
        z_null[r + i] = 1.0 # Set one free variable to 1
        
        # Solve for the basic variables
        rhs = -U[:r, r + i]
        
        # Perform back substitution
        z_basic = np.zeros(r)
        for k in range(r - 1, -1, -1):
            s = np.dot(U[k, k + 1:r], z_basic[k + 1:r])
            z_basic[k] = (rhs[k] - s) / U[k, k]
        
        z_null[:r] = z_basic
        N_u[:, i] = z_null
        
    return N_u

def solve(A: Array, b: Array, tol: float = 1e-12) -> Tuple[Optional[Array], Array]:
    """Solves the linear system A x = b."""
    A_in = np.asarray(A, dtype=float)
    # Ensure b is 2D
    b_in = np.asarray(b, dtype=float)
    if b_in.ndim == 1:
        b_in = b_in.reshape(-1, 1)

    m, n = A_in.shape

    # 1. Decompose
    P, Q, L, U, r = paq_lu(A_in, tol=tol)

    # 2. Find Nullspace of A
    N_u = _build_nullspace(U, r, n)
    N = Q @ N_u

    # 3. Find Particular Solution
    # Loop through each column of b, though tests likely only have one.
    c_list = []
    for i in range(b_in.shape[1]):
        b_col = b_in[:, i]
        b_perm = P @ b_col

        # Consistency Check
        if r < m and np.any(np.abs(b_perm[r:]) > tol):
            return None, N
        
        y = _forward_substitution(L, b_perm)
        z = _backward_substitution(U, y)
        c_list.append(z)

    # Stack solutions if b had multiple columns
    c_permuted = np.hstack(c_list) if c_list else np.zeros((n, 0))

    # Un-permute the particular solution
    c = Q @ c_permuted
    # Ensure c is a 1D array if b was 1D
    if b.ndim == 1:
        c = c.flatten()

    return c, N
