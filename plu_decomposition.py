import numpy as np
from typing import Tuple

def paq_lu(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Computes PAQ = LU with full (complete) pivoting.
    Returns:
        P : row permutation matrix (m x m)
        L : lower-triangular matrix (m x r) with unit diagonal
        U : upper-triangular matrix (r x n)
        Q : column permutation matrix (n x n)
        r : numerical rank of A
    """
    m, n = A.shape
    U_matrix = A.astype(np.float64, copy=True)
    k_max = min(m, n)

    P_matrix = np.eye(m, dtype=np.float64)
    Q_matrix = np.eye(n, dtype=np.float64)
    L_matrix = np.zeros((m, k_max), dtype=np.float64)

    rank = k_max
    for k in range(k_max):
        # Find pivot in remaining submatrix
        sub_matrix = U_matrix[k:, k:]
        pivot_idx = np.argmax(np.abs(sub_matrix))
        p_row = k + (pivot_idx // sub_matrix.shape[1])
        p_col = k + (pivot_idx % sub_matrix.shape[1])

        if np.abs(U_matrix[p_row, p_col]) < tol:
            rank = k
            break

        # Swap rows (P and L)
        U_matrix[[p_row, k], :] = U_matrix[[k, p_row], :]
        P_matrix[[p_row, k], :] = P_matrix[[k, p_row], :]
        if k > 0:
            L_matrix[[p_row, k], :k] = L_matrix[[k, p_row], :k]

        # Swap columns (Q)
        U_matrix[:, [p_col, k]] = U_matrix[:, [k, p_col]]
        Q_matrix[:, [p_col, k]] = Q_matrix[:, [k, p_col]]

        # Eliminate below pivot
        for i in range(k + 1, m):
            L_matrix[i, k] = U_matrix[i, k] / U_matrix[k, k]
            U_matrix[i, k:] -= L_matrix[i, k] * U_matrix[k, k:]

    # Set unit diagonal on L
    for i in range(rank):
        L_matrix[i, i] = 1.0

    # Trim to rank
    L_matrix = L_matrix[:, :rank]
    U_matrix = U_matrix[:rank, :]

    # ✅ Corrected return order
    return P_matrix, L_matrix, U_matrix, Q_matrix, rank
