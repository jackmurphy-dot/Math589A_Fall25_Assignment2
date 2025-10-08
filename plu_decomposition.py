import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-8) -> Tuple[Array, np.ndarray, np.ndarray, int]:
    """
    In-place PAQ = LU with partial pivoting over both rows and columns.
    - Row exchanges are simulated with the permutation vector P.
    - Column exchanges are virtual via the permutation vector Q (A itself is not column-swapped).
    Returns:
        A : ndarray (m x n) storing L (strict lower, unit diag implicit) and U (upper)
        P : ndarray (m,) row permutation indices
        Q : ndarray (n,) column permutation indices: pivot columns first
        r : int       numerical rank
    """
    A = np.asarray(A, dtype=float, order="C").copy()
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    # Work over the active k..m-1 by k..n-1 submatrix
    for k in range(min(m, n)):
        # Find pivot (i_rel, j_rel) by max-abs in remaining submatrix
        # NOTE: do not slice A directly; respect P and Q permutations.
        sub = np.abs(A[P[k:, None], Q[None, k:]])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_piv = k + i_rel
        j_piv = k + j_rel
        piv_val = A[P[i_piv], Q[j_piv]]

        if abs(piv_val) < tol:
            # No more usable pivots -> stop; r is current k
            break

        # Bring pivot row/col into position k (simulate row swap; virtual col swap)
        if i_piv != k:
            P[[k, i_piv]] = P[[i_piv, k]]
        if j_piv != k:
            Q[[k, j_piv]] = Q[[j_piv, k]]

        # Eliminate entries below the pivot in column Q[k]
        piv = A[P[k], Q[k]]
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= piv
            # Update remaining columns Q[k+1:]
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]

        r += 1

    return A, P, Q, r
