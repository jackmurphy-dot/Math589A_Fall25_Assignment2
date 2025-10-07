# plu_decomposition.py
# In-place PAQ = LU with simulated row exchanges and virtual column exchanges.
# Works for rectangular A (m x n). Double precision; default TOL=1e-6.

from typing import Tuple, List
import numpy as np
from logging_setup import logger


Array = np.ndarray

def paq_lu(A: Array, tol: float = 1e-6) -> Tuple[Array, np.ndarray, np.ndarray, List[int], int]:
    """
    Perform an in-place PAQ = LU factorization on A (m x n) using:
      - simulated row swaps via permutation vector P (do NOT physically swap rows),
      - virtual column swaps via permutation vector Q (do NOT physically swap columns),
      - partial pivoting in rows, with a column search step so pivot columns are moved before non-pivot columns.

    Storage convention in returned A (interpreted with P, Q):
      - U: upper-triangular r x r block   U_ij = A[P[i], Q[j]] for 0<=i<=j<r
      - L: unit-diagonal, strict lower triangle
            L_ij = A[P[i], Q[j]] for 0<=j<i<r  and for i>=r, entries L_ij may be nonzero but diagonal is implicitly 1

    Returns
    -------
    A : np.ndarray
        Overwritten with L (strict lower) and U (upper) in-place.
    P : np.ndarray, shape (m,)
        Row permutation vector. Access a logical row i via A[P[i], :].
    Q : np.ndarray, shape (n,)
        Column permutation vector. Access a logical column j via A[:, Q[j]].
    pivot_cols : list[int]
        The first r entries of Q (i.e., Q[:r]); included here for convenience.
    r : int
        Numerical rank determined by |pivot| >= tol.
    """
    if not isinstance(A, np.ndarray):
        A = np.asarray(A, dtype=np.float64)
    else:
        A = A  # in-place

    A = A.astype(np.float64, copy=False)
    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)

    r = 0  # rank / step counter
    min_mn = min(m, n)

    # Main elimination loop
    while r < min_mn:
        # Find the best pivot among columns j >= r: choose column whose best pivot (by abs) is largest
        best_val = 0.0
        piv_i = -1
        piv_j = -1
        for j in range(r, n):
            cj = Q[j]
            # max over rows r..m-1 of |A[P[i], cj]|
            # track both the value and the row index
            col_best_val = 0.0
            col_best_i = -1
            for i in range(r, m):
                val = abs(float(A[P[i], cj]))
                if val > col_best_val:
                    col_best_val = val
                    col_best_i = i
            if col_best_val > best_val:
                best_val = col_best_val
                piv_i = col_best_i
                piv_j = j

        # No pivot found above tolerance => stop; rank-deficient
        if best_val < tol or piv_i < 0 or piv_j < 0:
            break

        # Bring this pivot to position (r, r) virtually:
        # simulated row swap: swap P[r] <-> P[piv_i]
        if piv_i != r:
            P[r], P[piv_i] = P[piv_i], P[r]
        # virtual column swap: swap Q[r] <-> Q[piv_j]
        if piv_j != r:
            Q[r], Q[piv_j] = Q[piv_j], Q[r]

        # Pivot value
        pr = P[r]
        qr = Q[r]
        piv = float(A[pr, qr])

        # Gaussian elimination below the pivot (update A in-place via true rows and virtual columns)
        # Store multipliers into the strictly-lower positions (L)
        for i in range(r + 1, m):
            pi = P[i]
            if abs(piv) >= tol:
                Lij = A[pi, qr] / piv
            else:
                Lij = 0.0
            A[pi, qr] = Lij  # store L_ij

            # Row update on the trailing columns (j > r)
            if Lij != 0.0:
                for j in range(r + 1, n):
                    pj = Q[j]
                    A[pi, pj] -= Lij * A[pr, pj]

        r += 1

    pivot_cols = list(Q[:r])
    return A, P, Q, pivot_cols, r
