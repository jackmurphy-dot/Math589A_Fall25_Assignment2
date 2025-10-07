import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Compute PAQ = LU with partial pivoting and column exchanges.
    This ensures pivot columns are moved before non-pivot columns.

    Returns:
      A (modified, holds L and U),
      P, Q (permutation vectors),
      pivot_cols (list of pivot column indices, Q[:r]),
      r (rank)
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    # Loop over potential pivot positions
    while r < min(m, n):
        # --- Partial Pivot Search ---
        # Find the best pivot row in the current column `r` (from rows `r` to `m-1`).
        i_rel = np.argmax(np.abs(A[P[r:], Q[r]]))
        i_max = r + i_rel
        pivot_val = A[P[i_max], Q[r]]

        # --- Column Exchange ---
        # If the pivot in the current column is too small, search subsequent
        # columns for a suitable pivot.
        if abs(pivot_val) < tol:
            found_new_pivot = False
            for j_search in range(r + 1, n):
                i_rel_new = np.argmax(np.abs(A[P[r:], Q[j_search]]))
                i_max_new = r + i_rel_new
                if abs(A[P[i_max_new], Q[j_search]]) >= tol:
                    # Found a suitable pivot column. Swap columns `r` and `j_search`.
                    Q[[r, j_search]] = Q[[j_search, r]]
                    i_max = i_max_new  # Update the pivot row index for the new column
                    found_new_pivot = True
                    break
            
            # If no suitable pivot was found in any remaining column, the rank is `r`.
            if not found_new_pivot:
                break # Exit main loop

        # --- Row Exchange ---
        # Swap the current row with the pivot row.
        P[[r, i_max]] = P[[i_max, r]]

        # --- Gaussian Elimination ---
        pr, qr = P[r], Q[r]
        pivot_element = A[pr, qr]

        # Calculate multipliers and update the submatrix.
        for i in range(r + 1, m):
            pi = P[i]
            multiplier = A[pi, qr] / pivot_element
            A[pi, qr] = multiplier  # Store L factor
            # Update the rest of the row
            A[pi, Q[r + 1:]] -= multiplier * A[pr, Q[r + 1:]]
        
        r += 1

    pivot_cols = list(Q[:r])
    return A, P, Q, pivot_cols, r
