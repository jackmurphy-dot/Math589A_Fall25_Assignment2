import numpy as np
from typing import Tuple, Optional

from plu_decomposition import paq_lu

def solve_lower(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solves Ly = b for y."""
    rank = L.shape[1]
    y = np.zeros(rank, dtype=float)
    for i in range(rank):
        s = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - s) / L[i, i] # L[i,i] is always 1
    return y

def solve_upper(U: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Solves Uz = y for z."""
    rank, n = U.shape
    z = np.zeros(n, dtype=float)
    for i in range(rank - 1, -1, -1):
        s = np.dot(U[i, i + 1:], z[i + 1:])
        pivot = U[i, i]
        if abs(pivot) < tol:
             raise np.linalg.LinAlgError("Singular matrix in back substitution.")
        z[i] = (y[i] - s) / pivot
    return z

def get_null_space(U: np.ndarray, r: int, n: int, tol: float) -> np.ndarray:
    """Constructs the nullspace basis from U."""
    num_free = n - r
    if num_free <= 0:
        return np.zeros((n, 0))

    N_permuted = np.zeros((n, num_free))
    for i in range(num_free):
        z_vec = np.zeros(n)
        z_vec[r + i] = 1.0 
        
        # Solve for the basic variables
        for k in range(r - 1, -1, -1):
            s = np.dot(U[k, k+1:], z_vec[k+1:])
            pivot = U[k, k]
            if abs(pivot) < tol:
                 raise np.linalg.LinAlgError("Singular matrix in nullspace calculation.")
            z_vec[k] = -s / pivot
        
        N_permuted[:, i] = z_vec
        
    return N_permuted

def solve(A: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Solves the linear system A x = b."""
    A_in = np.asarray(A, dtype=float)
    b_in = np.asarray(b, dtype=float)
    if b_in.ndim == 1:
        b_in = b_in.reshape(-1, 1)
    m, n = A_in.shape

    P, Q, L, U, r = paq_lu(A_in, tol)

    # First, construct the nullspace of A, which is Q @ Nullspace(U)
    N_u = get_null_space(U, r, n, tol)
    N = Q @ N_u

    # Second, find the particular solution c
    c_list = []
    for i in range(b_in.shape[1]):
        b_col = b_in[:, i]
        b_perm = P @ b_col # Permute b

        # Check for consistency
        if r < m and np.any(np.abs(b_perm[r:]) > tol):
            return None, N
        
        # Solve Ly=Pb and Uz=y
        y = solve_lower(L, b_perm)
        z = solve_upper(U, y, tol)
        c_list.append(z)
        
    z_solution = np.hstack(c_list) if c_list else np.zeros((n, 0))

    # Un-permute the solution: c = Qz
    c = Q @ z_solution
    if b.ndim == 1:
        c = c.flatten()

    return c, N
