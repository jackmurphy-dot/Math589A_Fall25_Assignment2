import numpy as np
from typing import Tuple

def paq_lu(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Complete pivoting LU factorization:  PAQ = LU

    Returns:
        P : (m x m) permutation matrix (rows)
        L : (m x r) lower-triangular with unit diagonal on L[:r,:r]
        U : (r x n) upper-triangular
        Q : (n x n) permutation matrix (columns)
        r : numerical rank
    """
    m, n = A.shape
    U = A.astype(float).copy()
    P = np.eye(m, dtype=float)
    Q = np.eye(n, dtype=float)

    k_max = min(m, n)
    L = np.zeros((m, k_max), dtype=float)
    r = k_max

    for k in range(k_max):
        # Pick pivot by absolute value in the active submatrix
        sub = U[k:, k:]
        idx = int(np.argmax(np.abs(sub)))
        pr = k + idx // sub.shape[1]
        pc = k + idx %  sub.shape[1]

        # Rank check
        if abs(U[pr, pc]) < tol:
            r = k
            break

        # Row swap (U, P, and the past columns of L)
        U[[k, pr], :] = U[[pr, k], :]
        P[[k, pr], :] = P[[pr, k], :]
        if k > 0:
            L[[k, pr], :k] = L[[pr, k], :k]

        # Column swap (U, Q)
        U[:, [k, pc]] = U[:, [pc, k]]
        Q[:, [k, pc]] = Q[:, [pc, k]]

        # Eliminate below the pivot
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    # Put unit diagonal on the leading r x r block of L
    for i in range(r):
        L[i, i] = 1.0

    # Trim to the numerical rank
    L = L[:, :r]
    U = U[:r, :]

    return P, L, U, Q, r
