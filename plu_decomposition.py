import numpy as np
from typing import Tuple
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-10) -> Tuple[Array, np.ndarray, np.ndarray, int]:
    """
    Perform an in-place PAQ = LU decomposition with both row and column pivoting.
    Returns (A, P, Q, r) where:
      A  - contains L (unit diag implicit) and U
      P  - row permutation indices
      Q  - column permutation indices (virtual)
      r  - numerical rank
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    for k in range(min(m, n)):
        # Find pivot (row, col) among remaining submatrix
        submat = np.abs(A[P[k:], :][:, Q[k:]])
        i_rel, j_rel = np.unravel_index(np.argmax(submat), submat.shape)
        pivot_val = A[P[k + i_rel], Q[k + j_rel]]

        # Check for numerical singularity
        if abs(pivot_val) < tol:
            break

        # Swap rows and columns in permutation vectors (simulate)
        i_piv = k + i_rel
        j_piv = k + j_rel
        P[[k, i_piv]] = P[[i_piv, k]]
        Q[[k, j_piv]] = Q[[j_piv, k]]

        # Gaussian elimination step
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= A[P[k], Q[k]]
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]

        r += 1

    return A, P, Q, r
