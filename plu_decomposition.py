import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-10) -> Tuple[Array, np.ndarray, np.ndarray, int]:
    """
    Simplified PLU decomposition with partial pivoting (rows only).
    Returns (A, P, Q, r) where r is the numerical rank.
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)

    for k in range(min(m, n)):
        # Find pivot row (partial pivoting)
        pivot_row = k + np.argmax(np.abs(A[P[k:], Q[k]]))
        pivot_val = A[P[pivot_row], Q[k]]

        if abs(pivot_val) < tol:
            # Rank deficiency detected
            return A, P, Q, k

        # Swap rows in permutation vector
        P[[k, pivot_row]] = P[[pivot_row, k]]

        # Elimination
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= A[P[k], Q[k]]
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]

    r = min(m, n)
    return A, P, Q, r
