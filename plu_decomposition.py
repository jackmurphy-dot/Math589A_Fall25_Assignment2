import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Computes the PAQ=LU decomposition of a matrix A using partial pivoting
    with column exchanges to handle rank deficiency.

    Args:
        A (Array): The m x n matrix to decompose.
        tol (float): The tolerance below which a pivot is considered zero.

    Returns:
        A: The matrix A overwritten with L (strictly lower) and U (upper) factors.
        P: The row permutation vector.
        Q: The column permutation vector.
        list(Q[:r]): A list of the pivot columns.
        r: The computed rank of the matrix.
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    
    # k is the current pivot number, which will become the rank r
    k = 0
    while k < min(m, n):
        # --- Partial Pivot Search ---
        # Find the best pivot row in the current logical column k.
        i_rel = np.argmax(np.abs(A[P[k:], Q[k]]))
        i_max = k + i_rel
        
        # --- Column Exchange if Necessary ---
        # If the best pivot in this column is too small, search for a new pivot column.
        if np.abs(A[P[i_max], Q[k]]) < tol:
            found_new_col = False
            for j_new in range(k + 1, n):
                # Check if this new column has a potential pivot
                i_rel_new = np.argmax(np.abs(A[P[k:], Q[j_new]]))
                if np.abs(A[P[k + i_rel_new], Q[j_new]]) >= tol:
                    # Found a suitable column. Swap logical columns k and j_new.
                    Q[[k, j_new]] = Q[[j_new, k]]
                    # The pivot row for this new column is now i_max
                    i_max = k + i_rel_new
                    found_new_col = True
                    break
            
            # If no suitable pivot was found in any remaining column, the rank is k.
            if not found_new_col:
                break # Exit the main while loop

        # --- Row Exchange ---
        # Swap the current row with the chosen pivot row.
        P[[k, i_max]] = P[[i_max, k]]
        
        # --- Elimination ---
        pk, qk = P[k], Q[k]
        pivot_element = A[pk, qk]
        
        # Update the pivot column with L-factors and the submatrix
        for i in range(k + 1, m):
            pi = P[i]
            multiplier = A[pi, qk] / pivot_element
            A[pi, qk] = multiplier
            A[pi, Q[k+1:]] -= multiplier * A[pk, Q[k+1:]]
            
        k += 1

    r = k
    return A, P, Q, list(Q[:r]), r
