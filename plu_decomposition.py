import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-12) -> Tuple[Array, Array, Array, Array, int]:
    """
    Computes PAQ=LU decomposition, returning separate L and U matrices.
    """
    A_U = np.asarray(A, dtype=float, copy=True)
    m, n = A_U.shape
    
    P = np.arange(m)
    Q = np.arange(n)
    L = np.zeros((m, min(m, n)), dtype=float)
    
    r = 0
    for k in range(min(m, n)):
        # 1. Find the pivot (largest element) in the remaining submatrix of U
        sub_matrix = A_U[k:, k:]
        i_rel, j_rel = np.unravel_index(np.argmax(np.abs(sub_matrix)), sub_matrix.shape)
        pivot_row = k + i_rel
        pivot_col = k + j_rel
        
        if np.abs(A_U[pivot_row, pivot_col]) < tol:
            break # No more pivots, rank is k.
            
        # 2. Perform swaps on matrices and permutation vectors
        A_U[[k, pivot_row], :] = A_U[[pivot_row, k], :] # Swap rows in U
        L[[k, pivot_row], :k] = L[[pivot_row, k], :k] # Swap corresponding rows in L
        P[[k, pivot_row]] = P[[pivot_row, k]]         # Update row permutation vector

        A_U[:, [k, pivot_col]] = A_U[:, [pivot_col, k]] # Swap columns in U
        Q[[k, pivot_col]] = Q[[pivot_col, k]]         # Update column permutation vector
        
        # 3. Set diagonal of L and compute multipliers
        L[k, k] = 1.0
        for i in range(k + 1, m):
            L[i, k] = A_U[i, k] / A_U[k, k]
            # Elimination step on U
            A_U[i, k:] -= L[i, k] * A_U[k, k:]
        
        r += 1
        
    # Trim matrices to final rank
    L = L[:, :r]
    U = A_U[:r, :]
    
    return P, Q, L, U, r
