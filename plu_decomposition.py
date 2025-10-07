import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Compute PAQ = LU with partial pivoting and column exchanges.
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    k = 0
    while k < min(m, n):
        # Find the best pivot row in the current column k
        i_rel = np.argmax(np.abs(A[P[k:], Q[k]]))
        i_max = k + i_rel
        
        # If the pivot is too small, find the first available substitute column
        if np.abs(A[P[i_max], Q[k]]) < tol:
            found_swap = False
            for j_swap in range(k + 1, n):
                # Find best pivot in the potential swap column
                i_rel_new = np.argmax(np.abs(A[P[k:], Q[j_swap]]))
                i_max_new = k + i_rel_new
                if np.abs(A[P[i_max_new], Q[j_swap]]) >= tol:
                    # Found a suitable column. Perform the swap.
                    Q[[k, j_swap]] = Q[[j_swap, k]]
                    i_max = i_max_new # Update the pivot row for the new column
                    found_swap = True
                    break
            if not found_swap:
                break # No more pivots found anywhere. Rank is k.
        
        # Bring the pivot row to the current position k
        P[[k, i_max]] = P[[i_max, k]]
        
        # Perform elimination
        pk, qk = P[k], Q[k]
        pivot = A[pk, qk]
        
        # Calculate and store multipliers in the pivot column
        A[P[k+1:], qk] /= pivot
        
        # Update the rest of the submatrix using an outer product
        if k + 1 < n:
            A[np.ix_(P[k+1:], Q[k+1:])] -= np.outer(A[P[k+1:], qk], A[pk, Q[k+1:]])
        
        k += 1
    
    r = k
    return A, P, Q, list(Q[:r]), r
