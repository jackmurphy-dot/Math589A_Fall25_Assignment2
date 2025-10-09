import numpy as np
from typing import Tuple

def paq_lu(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Computes the PAQ = LU decomposition of A using complete (full) pivoting.

    Returns:
        P : (m x m) permutation matrix for row swaps
        L : (m x r) lower-triangular with unit diagonal
        U : (r x n) upper-triangular
        Q : (n x n) permutation matrix for column swaps
        r : numerical rank of A
    """
    m, n = A.shape
    U_matrix = A.astype(float).copy()
    P_matrix = np.eye(m)
    Q_matrix = np.eye(n)
    k_max = min(m, n)
    L_matrix = np.zeros((m, k_max))
    rank = k_max

    for k in range(k_max):
        # Find pivot with largest absolute value in submatrix
        sub = U_matrix[k:, k:]
        i_rel, j_rel = divmod(np.argmax(np.abs(sub)), sub.shape[1])
        i, j = k + i_rel, k + j_rel

        if abs(U_matrix[i, j]) < tol:
            rank = k
            break

        # Row swap
        U_matrix[[k, i], :] = U_matrix[[i, k], :]
        P_matrix[[k, i], :] = P_matrix[[i, k], :]
        if k > 0:
            L_matrix[[k, i], :k] = L_matrix[[i, k], :k]

        # Column swap
        U_matrix[:, [k, j]] = U_matrix[:, [j, k]]
        Q_matrix[:, [k, j]] = Q_matrix[:, [j, k]]

        # Eliminate below pivot
        for row in range(k + 1, m):
            L_matrix[row, k] = U_matrix[row, k] / U_matrix[k, k]
            U_matrix[row, k:] -= L_matrix[row, k] * U_matrix[k, k:]

    # Set unit diagonal in L
    for i in range(rank):
        L_matrix[i, i] = 1.0

    # Trim to rank
    L_matrix = L_matrix[:, :rank]
    U_matrix = U_matrix[:rank, :]

    return P_matrix, L_matrix, U_matrix, Q_matrix, rank
