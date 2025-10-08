import numpy as np
from logging_setup import logger


def paq_lu(A, tol: float = 1e-12):
    """
    Compute LU factorization with both row and column pivoting (PAQ = LU).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Input matrix.
    tol : float, optional
        Numerical tolerance for rank determination.

    Returns
    -------
    A_fac : ndarray
        In-place modified copy of A containing combined L and U factors.
        L has implicit unit diagonal below; U occupies the upper triangle.
    P : ndarray of int
        Row permutation indices such that P @ A @ Q = L @ U.
    Q : ndarray of int
        Column permutation indices.
    r : int
        Numerical rank of A.
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0

    for k in range(min(m, n)):
        # Find the pivot (largest abs entry) in remaining submatrix
        sub = np.abs(A[P[k:], Q[k:]])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_piv, j_piv = k + i_rel, k + j_rel
        piv = A[P[i_piv], Q[j_piv]]

        if abs(piv) <= tol:
            # Matrix is rank-deficient below tolerance
            break

        # Swap pivot row and column into position
        P[[k, i_piv]] = P[[i_piv, k]]
        Q[[k, j_piv]] = Q[[j_piv, k]]

        # Elimination below pivot
        for i in range(k + 1, m):
            A[P[i], Q[k]] /= A[P[k], Q[k]]
            A[P[i], Q[k + 1:]] -= A[P[i], Q[k]] * A[P[k], Q[k + 1:]]
        r += 1

    logger.debug(f"PAQ_LU done: rank={r}")
    return A, P, Q, r
