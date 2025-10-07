import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Computes a standard PLU decomposition (row pivots only).
    This simplified version is for debugging the non-singular case (P2.2).
    """
    A = np.asarray(A, dtype=np.float64, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n) # Q will remain the identity matrix [0, 1, 2, ...]

    # Loop over each pivot position
    for k in range(min(m, n)):
        # --- Partial Pivoting (Rows Only) ---
        # Find the row with the largest value in the current column k
        pivot_row_candidate = k + np.argmax(np.abs(A[P[k:], Q[k]]))
        
        # Perform the virtual row swap in P
        P[[k, pivot_row_candidate]] = P[[pivot_row_candidate, k]]
        
        # --- Elimination ---
        pivot_element = A[P[k], Q[k]]
        
        # If the pivot is zero, the matrix is singular. Stop.
        if np.abs(pivot_element) < tol:
            # This indicates the rank is k.
            r = k
            return A, P, Q, list(Q[:r]), r

        # Update all rows below the pivot
        for i in range(k + 1, m):
            # Calculate the multiplier
            multiplier = A[P[i], Q[k]] / pivot_element
            # Store the multiplier in the L factor part of A
            A[P[i], Q[k]] = multiplier
            # Update the rest of the row
            A[P[i], Q[k+1:]] -= multiplier * A[P[k], Q[k+1:]]

    r = min(m, n)
    return A, P, Q, list(Q[:r]), r
