import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-10) -> Tuple[Array, np.ndarray, np.ndarray, int]:
    """
    In-place PAQ = LU with partial pivoting over both rows and columns.
    - Row exchanges are simulated with P.
    - Column exchanges are virtual via Q (A is not column-swapped).
    Returns:
        A : stores L (strict lower, unit diag implicit) and U (upper)
        P : row permutation indices
        Q : column permutation indices, with pivot columns first
        r : numerical rank
    """
    A = np.asarray(A, dtype=float, order="C").copy()
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    for k in range(min(m, n)):
        # Find absolute max pivot in remaining submatrix
        sub = np.abs(A[P[k:, None], Q[None, k:]])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_piv = k + i_rel
        j_piv = k + j_rel
        piv_val = A[P[i_piv], Q[j_piv]]

        if abs(piv_val) < tol:
            break

        # Simulate row swap; virtual column swap
        if i_piv != k:
            P[[k, i_piv]] = P[[i_piv, k]]
        if j_piv != k:
            Q[[k, j_piv]] = Q[[j_piv, k]]

        piv = A[P[k], Q[k]]
        # Eliminate below pivot in column Q[k]
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= piv
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]

        r += 1

    return A, P, Q, r
