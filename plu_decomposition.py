import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Computes PAQ=LU decomposition using partial pivoting with column exchanges.
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    k = 0
    while k < min(m, n):
        i_rel = np.argmax(np.abs(A[P[k:], Q[k]]))
        i_max = k + i_rel
        
        if np.abs(A[P[i_max], Q[k]]) < tol:
            found_better_col = False
            for j_search in range(k + 1, n):
                i_rel_new = np.argmax(np.abs(A[P[k:], Q[j_search]]))
                pivot_row_in_new_col = k + i_rel_new
                if np.abs(A[pivot_row_in_new_col, Q[j_search]]) >= tol:
                    Q[[k, j_search]] = Q[[j_search, k]]
                    i_max = pivot_row_in_new_col
                    found_better_col = True
                    break
            
            if not found_better_col:
                break

        P[[k, i_max]] = P[[i_max, k]]
        
        pivot_element = A[P[k], Q[k]]
        for i in range(k + 1, m):
            multiplier = A[P[i], Q[k]] / pivot_element
            A[P[i], Q[k]] = multiplier
            for j in range(k + 1, n):
                A[P[i], Q[j]] -= multiplier * A[P[k], Q[j]]
                
        k += 1

    r = k
    return A, P, Q, list(Q[:r]), r
