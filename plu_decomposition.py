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
    
    # k will be the final rank
    k = 0
    while k < min(m, n):
        # Find best pivot in current column k
        pivot_row_local = np.argmax(np.abs(A[P[k:], Q[k]]))
        pivot_row_global = k + pivot_row_local
        
        # If pivot is too small, find a better column
        if np.abs(A[P[pivot_row_global], Q[k]]) < tol:
            best_j = -1
            max_pivot_val = 0.0
            best_i_global_for_swap = -1

            # Search remaining columns for the best possible pivot
            for j_search in range(k + 1, n):
                i_local = np.argmax(np.abs(A[P[k:], Q[j_search]]))
                current_pivot_val = np.abs(A[P[k + i_local], Q[j_search]])
                if current_pivot_val > max_pivot_val:
                    max_pivot_val = current_pivot_val
                    best_j = j_search
                    best_i_global_for_swap = k + i_local
            
            if max_pivot_val >= tol:
                Q[[k, best_j]] = Q[[best_j, k]]
                pivot_row_global = best_i_global_for_swap
            else:
                # No suitable pivot found; rank is k.
                break

        # Perform row swap
        P[[k, pivot_row_global]] = P[[pivot_row_global, k]]
        
        # Elimination
        pk, qk = P[k], Q[k]
        pivot_element = A[pk, qk]
        
        # Update pivot column (L factors)
        A[P[k+1:], qk] /= pivot_element
        
        # Update submatrix (Schur complement)
        if k + 1 < n:
            sub_L_col = A[P[k+1:], qk].reshape(-1, 1)
            sub_U_row = A[pk, Q[k+1:]].reshape(1, -1)
            A[np.ix_(P[k+1:], Q[k+1:])] -= sub_L_col @ sub_U_row
        
        k += 1
    
    r = k
    return A, P, Q, list(Q[:r]), r
