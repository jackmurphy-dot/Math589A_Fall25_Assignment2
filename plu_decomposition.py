import numpy as np
from typing import Tuple, List
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Compute PAQ = LU with full pivoting (global row+column pivot).
    Returns:
      A (modified, holds L and U),
      P, Q (permutation vectors),
      pivot_cols (Q[:r]),
      r (rank)
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    P = np.arange(m)
    Q = np.arange(n)
    r = 0
    min_mn = min(m, n)

    while r < min_mn:
        # Full pivot search on submatrix A[P[r:], Q[r:]]
        sub = np.abs(A[np.ix_(P[r:], Q[r:])])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i_rel, j_rel] < tol:
            break
        i, j = r + i_rel, r + j_rel

        # Row and column swaps
        if i != r:
            P[[r, i]] = P[[i, r]]
        if j != r:
            Q[[r, j]] = Q[[j, r]]

        pr, qr = P[r], Q[r]
        piv = A[pr, qr]

        # Eliminate below pivot
        for ii in range(r + 1, m):
            pi = P[ii]
            Lij = A[pi, qr] / piv
            A[pi, qr] = Lij
            A[pi, Q[r + 1:]] -= Lij * A[pr, Q[r + 1:]]
        r += 1

    pivot_cols = list(Q[:r])
    return A, P, Q, pivot_cols, r
