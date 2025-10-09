import numpy as np
from typing import Tuple

def paq_lu(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Computes PAQ=LU decomposition, returning separate L and U matrices,
    and P and Q as permutation matrices.
    """
    m, n = A.shape
    U = A.astype(np.float64, copy=True)
    max_rank = min(m, n)
    
    # Initialize P and Q as identity MATRICES
    P = np.eye(m, dtype=np.float64)
    Q = np.eye(n, dtype=np.float64)
    L = np.zeros((m, max_rank), dtype=np.float64)
    
    rank = max_rank
    for k in range(max_rank):
        # Full pivot search on the remaining submatrix of U
        idx = np.argmax(np.abs(U[k:, k:]))
        pivot_row = k + (idx // U[k:, k:].shape[1])
        pivot_col = k + (idx % U[k:, k:].shape[1])

        if np.abs(U[pivot_row, pivot_col]) < tol:
            rank = k
            break
            
        # Perform physical swaps on the matrices
        U[[pivot_row, k], :] = U[[k, pivot_row], :]
        P[[pivot_row, k], :] = P[[k, pivot_row], :]
        U[:, [pivot_col, k]] = U[:, [k, pivot_col]]
        Q[:, [pivot_col, k]] = Q[:, [k, pivot_col]]
        if k > 0:
            L[[pivot_row, k], :k] = L[[k, pivot_row], :k]
        
        # Elimination
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    # Add unit diagonal to L and trim matrices to final rank
    for i in range(rank):
        L[i, i] = 1.0
        
    L = L[:, :rank]
    U = U[:rank, :]
    
    return P, Q, L, U, rank
