import numpy as np
from typing import Tuple, Optional

from plu_decomposition import paq_lu

def _forward_substitution(L: np.ndarray, b_perm: np.ndarray) -> np.ndarray:
    """Solves Ly = b_perm for y."""
    rank = L.shape[1]
    y = np.zeros(rank, dtype=float)
    for i in range(rank):
        s = np.dot(L[i, :i], y[:i])
        y[i] = b_perm[i] - s # L[i,i] is 1
    return y

def _backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solves Uz = y for z."""
    rank, n = U.shape
    z = np.zeros(n, dtype=float)
    for i in range(rank - 1, -1, -1):
        s = np.dot(U[i, i + 1:], z[i + 1:])
        z[i] = (y[i] - s) / U[i, i]
    return z

def _build_nullspace(U: np.ndarray, r: int, n: int) -> np.ndarray:
    """Constructs the nullspace of U."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0))

    N_u = np.zeros((n, num_free))
    for i in range(num_free):
        z_null = np.zeros(n)
        z_null[r + i] = 1.0 
        
        rhs = -U[:r, r + i]
        
        z_basic = np.zeros(r)
        for k in range(r - 1, -1, -1):
            s = np.dot(U[k, k + 1:r], z_basic[k + 1:r])
            z_basic[k] = (rhs[k] - s) / U[k, k]
        
        z_null[:r] = z_basic
        N_u[:, i] = z_null
        
    return N_u

def solve(A: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Solves the linear system A x = b."""
    A_in = np.asarray(A, dtype=float)
    b_in = np.asarray(b, dtype=float)
    if b_in.ndim == 1:
        b_in = b_in.reshape(-1, 1)
    m, n = A_in.shape

    P, Q, L, U, r = paq_lu(A_in, tol)

    N_u = _build_nullspace(U, r, n)
    N = Q @ N_u

    c_list = []
    for i in range(b_in.shape[1]):
        b_col = b_in[:, i]
        b_perm = P @ b_col

        if r < m and np.any(np.abs(b_perm[r:]) > tol):
            return None, N
        
        y = _forward_substitution(L, b_perm)
        z = _backward_substitution(U, y)
        c_list.append(z)

    c_permuted = np.hstack(c_list) if c_list else np.zeros((n, 0))

    c = Q @ c_permuted
    if b.ndim == 1:
        c = c.flatten()

    return c, N
