# plu_decomposition.py
# In-place PAQ = LU with simulated row exchanges and virtual column exchanges.

from typing import Tuple, List
import numpy as np
from logging_setup import logger

Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=np.float64)
    A = A.astype(np.float64, copy=False)

    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)
    r = 0
    min_mn = min(m, n)

    while r < min_mn:
        best_val = 0.0
        piv_i = -1
        piv_j = -1
        # Full pivot search (max abs value)
        for j in range(r, n):
            cj = Q[j]
            for i in range(r, m):
                val = abs(A[P[i], cj])
                if val > best_val:
                    best_val, piv_i, piv_j = val, i, j

        if best_val < tol or piv_i < 0 or piv_j < 0:
            break

        if piv_i != r:
            P[r], P[piv_i] = P[piv_i], P[r]
        if piv_j != r:
            Q[r], Q[piv_j] = Q[piv_j], Q[r]

        pr, qr = P[r], Q[r]
        piv = A[pr, qr]

        for i in range(r + 1, m):
            pi = P[i]
            if abs(piv) >= tol:
                Lij = A[pi, qr] / piv
            else:
                Lij = 0.0
            A[pi, qr] = Lij
            if abs(Lij) > 0:
                A[pi, Q[r + 1 : n]] -= Lij * A[pr, Q[r + 1 : n]]

        r += 1

    pivot_cols = list(Q[:r])
    return A, P, Q, pivot_cols, r
