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
    
    # k tracks the number of pivots found, which determines the rank.
    k = 0
    while k < min(m, n):
        # --- 1. Partial Pivot Search in Current Column `k` ---
        # Find the row with the max value in the current logical column.
        max_val_in_col = 0.0
        pivot_row_idx = k
        for i in range(k, m):
            current_val = np.abs(A[P[i], Q[k]])
            if current_val > max_val_in_col:
                max_val_in_col = current_val
                pivot_row_idx = i

        # --- 2. Check Pivot and Perform Column Exchange if Needed ---
        # If the best pivot in this column is below tolerance, search other columns.
        if max_val_in_col < tol:
            found_better_col = False
            for j_search in range(k + 1, n):
                # Find the best pivot in this new candidate column
                max_val_in_new_col = 0.0
                pivot_row_in_new_col = k
                for i in range(k, m):
                    val = np.abs(A[P[i], Q[j_search]])
                    if val > max_val_in_new_col:
                        max_val_in_new_col = val
                        pivot_row_in_new_col = i
                
                # If this column has a usable pivot, swap it in and stop searching.
                if max_val_in_new_col >= tol:
                    Q[[k, j_search]] = Q[[j_search, k]]
                    pivot_row_idx = pivot_row_in_new_col
                    found_better_col = True
                    break
            
            # If no suitable pivot was found in any column, the rank is k.
            if not found_better_col:
                break # Exit the main while loop.

        # --- 3. Perform Row Swap ---
        P[[k, pivot_row_idx]] = P[[pivot_row_idx, k]]
        
        # --- 4. Elimination ---
        pivot_element = A[P[k], Q[k]]
        for i in range(k + 1, m):
            multiplier = A[P[i], Q[k]] / pivot_element
            A[P[i], Q[k]] = multiplier # Store L-factor
            # Update the rest of the row
            for j in range(k + 1, n):
                A[P[i], Q[j]] -= multiplier * A[P[k], Q[j]]
                
        k += 1 # Move to the next pivot

    r = k # The rank is the number of pivots found
    return A, P, Q, list(Q[:r]), r
