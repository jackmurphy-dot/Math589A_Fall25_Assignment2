import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Compute PAQ = LU with full pivoting.
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    # k will be the number of pivots found, i.e., the rank.
    k = 0
    while k < min(m, n):
        # Find the absolute maximum element in the active submatrix A[P[k:], Q[k:]].
        sub_matrix_rows = P[k:]
        sub_matrix_cols = Q[k:]
        sub_matrix = A[np.ix_(sub_matrix_rows, sub_matrix_cols)]
        
        # Find the relative indices of the maximum value
        i_rel, j_rel = np.unravel_index(np.argmax(np.abs(sub_matrix)), sub_matrix.shape)
        
        # Convert to absolute indices in P and Q
        i_abs, j_abs = k + i_rel, k + j_rel
        
        pivot_val = A[P[i_abs], Q[j_abs]]

        # If the best pivot is close to zero, the remaining matrix is rank-deficient.
        if np.abs(pivot_val) < tol:
            break # Stop decomposition. The rank is k.

        # Perform row and column swaps to bring the pivot to position k.
        P[[k, i_abs]] = P[[i_abs, k]]
        Q[[k, j_abs]] = Q[[j_abs, k]]

        # Get physical indices of the pivot element for elimination
        pk, qk = P[k], Q[k]
        
        # Perform elimination on the rows below the pivot
        for i_elim in range(k + 1, m):
            pi = P[i_elim]
            multiplier = A[pi, qk] / pivot_val
            A[pi, qk] = multiplier # Store L factor
            # Update the rest of the row
            A[pi, Q[k+1:]] -= multiplier * A[pk, Q[k+1:]]
        
        k += 1

    r = k # The rank is the number of pivots found.
    return A, P, Q, list(Q[:r]), r
