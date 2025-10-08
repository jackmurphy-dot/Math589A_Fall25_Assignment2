import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-10) -> Tuple[Array, np.ndarray, np.ndarray, int]:
    """
    In-place PAQ = LU with row (simulated) and column (virtual) pivoting.
    Returns:
        A : stores L (strict lower, unit diag implicit) and U (upper)
        P : row permutation indices
        Q : column permutation indices (pivot columns first)
        r : numerical rank
    """
    A = np.asarray(A, dtype=float, order="C").copy()
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    for k in range(min(m, n)):
        # Active submatrix absolute values
        sub = np.abs(A[P[k:, None], Q[None, k:]])
        if sub.size == 0:
            break
        sub_max = sub.max()

        # If the ENTIRE active submatrix is (numerically) zero, stop.
        if sub_max <= max(tol, 1e-12 * np.max(np.abs(A))):

            break

        # Pick (global) max pivot within the active submatrix
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_piv = k + i_rel
        j_piv = k + j_rel

        # Bring pivot into (k,k) via simulated row swap and virtual column swap
        if i_piv != k:
            P[[k, i_piv]] = P[[i_piv, k]]
        if j_piv != k:
            Q[[k, j_piv]] = Q[[j_piv, k]]

        piv = A[P[k], Q[k]]

        # Guard (paranoid): if this pivot is still too small, stop
        if abs(piv) <= tol:
            break

        # Eliminate below
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= piv
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]

        r += 1

    return A, P, Q, r
