import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-12) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Computes PAQ=LU decomposition using full pivoting (row and column exchanges).
    This version correctly performs the in-place factorization.
    """
    A = np.asarray(A, dtype=float, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    r = 0
    for k in range(min(m, n)):
        # 1. Find the pivot (largest element) in the remaining submatrix.
        sub_matrix_view = A[np.ix_(P[k:], Q[k:])]
        i_rel, j_rel = np.unravel_index(np.argmax(np.abs(sub_matrix_view)), sub_matrix_view.shape)
        
        # 2. Convert relative indices to absolute indices in P and Q.
        pivot_row_idx = k + i_rel
        pivot_col_idx = k + j_rel
        
        pivot_element = A[P[pivot_row_idx], Q[pivot_col_idx]]

        # 3. If the best pivot is close to zero, the rank is k.
        if abs(pivot_element) < tol:
            break

        # 4. Perform virtual row and column swaps.
        P[[k, pivot_row_idx]] = P[[pivot_row_idx, k]]
        Q[[k, pivot_col_idx]] = Q[[pivot_col_idx, k]]

        # 5. Correctly perform elimination on the rows below the pivot.
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= A[P[k], Q[k]]
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]
        
        r += 1

    return A, P, Q, list(Q[:r]), r
