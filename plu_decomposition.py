import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Computes PAQ=LU decomposition using full pivoting (row and column exchanges).
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    # k tracks the number of pivots found, which determines the rank.
    k = 0
    while k < min(m, n):
        # 1. Find the absolute maximum element in the active submatrix A[P[k:], Q[k:]]
        sub_matrix_view = A[np.ix_(P[k:], Q[k:])]
        i_rel, j_rel = np.unravel_index(np.argmax(np.abs(sub_matrix_view)), sub_matrix_view.shape)
        
        # 2. Check if the best pivot is too small
        if np.abs(sub_matrix_view[i_rel, j_rel]) < tol:
            break # No more pivots found, rank is k.

        # 3. Perform virtual swaps to bring the pivot to the k-th position
        # Convert relative indices to absolute indices in P and Q
        pivot_row_idx = k + i_rel
        pivot_col_idx = k + j_rel
        
        P[[k, pivot_row_idx]] = P[[pivot_row_idx, k]]
        Q[[k, pivot_col_idx]] = Q[[pivot_col_idx, k]]

        # 4. Elimination
        # Get the physical indices of the pivot row/column after swapping
        pivot_phys_row = P[k]
        pivot_phys_col = Q[k]
        pivot_element = A[pivot_phys_row, pivot_phys_col]
        
        # Update rows below the pivot
        for i in range(k + 1, m):
            elim_phys_row = P[i]
            
            # Calculate multiplier and store it in-place in the L-factor part
            multiplier = A[elim_phys_row, pivot_phys_col] / pivot_element
            A[elim_phys_row, pivot_phys_col] = multiplier
            
            # Update the remainder of the row
            update_cols = Q[k+1:]
            A[elim_phys_row, update_cols] -= multiplier * A[pivot_phys_row, update_cols]
        
        k += 1

    r = k
    return A, P, Q, list(Q[:r]), r
