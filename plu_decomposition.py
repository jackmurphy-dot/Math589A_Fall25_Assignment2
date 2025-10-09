import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, Array, Array, Array, int]:
    """
    Computes PAQ=LU decomposition, returning separate L and U matrices and P, Q as matrices.
    """
    A_U = np.asarray(A, dtype=float, copy=True)
    m, n = A_U.shape
    max_rank = min(m, n)
    
    # Initialize P and Q as identity MATRICES
    P = np.eye(m, dtype=float)
    Q = np.eye(n, dtype=float)
    L = np.zeros((m, max_rank), dtype=float)
    
    rank = max_rank
    for k in range(max_rank):
        # Full pivot search
        idx = np.argmax(np.abs(A_U[k:, k:]))
        pivot_row = k + (idx // A_U[k:, k:].shape[1])
        pivot_col = k + (idx % A_U[k:, k:].shape[1])
        
        if np.abs(A_U[pivot_row, pivot_col]) < tol:
            rank = k
            break
            
        # Physical swaps on matrices
        A_U[[pivot_row, k], :] = A_U[[k, pivot_row], :]
        P[[pivot_row, k], :] = P[[k, pivot_row], :]
        A_U[:, [pivot_col, k]] = A_U[:, [k, pivot_col]]
        Q[:, [pivot_col, k]] = Q[:, [k, pivot_col]]
        if k > 0:
            L[[pivot_row, k], :k] = L[[k, pivot_row], :k]
        
        # Elimination
        for i in range(k + 1, m):
            L[i, k] = A_U[i, k] / A_U[k, k]
            A_U[i, k:] -= L[i, k] * A_U[k, k:]
    
    # Add unit diagonal to L
    for i in range(rank):
        L[i, i] = 1.0
        
    # Trim matrices to final rank
    L = L[:, :rank]
    U = A_U[:rank, :]
    
    return P, Q, L, U, rank
