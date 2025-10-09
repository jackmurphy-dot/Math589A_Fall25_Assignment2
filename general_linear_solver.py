import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Optional[Array], Array]:
    """
    Solves the linear system A x = b using the PAQ=LU decomposition.
    """
    A_in = np.asarray(A, dtype=float)
    b_in = np.asarray(b, dtype=float).reshape(-1)
    m, n = A_in.shape

    # 1. Decompose the matrix to get PAQ = LU
    A_fac, P, Q, r, pivot_cols = paq_lu(A_in, tol=tol)

    # 2. Solve Ly = Pb using forward substitution
    y = np.zeros(m, dtype=float)
    b_permuted = b_in[P]
    for i in range(m):
        # The sum is over the columns of L that have been computed
        s = np.dot(A_fac[P[i], Q[:min(i, r)]], y[:min(i, r)])
        y[i] = b_permuted[i] - s

    # 3. Solve for the particular solution
    c = np.zeros(n, dtype=float)
    if r > 0:
        # Solve U_basic * z_basic = y[:r] using backward substitution
        z_basic = np.zeros(r, dtype=float)
        for i in range(r - 1, -1, -1):
            s = np.dot(A_fac[P[i], Q[i + 1:r]], z_basic[i + 1:r])
            pivot = A_fac[P[i], Q[i]]
            z_basic[i] = (y[i] - s) / pivot
        c[Q[:r]] = z_basic

    # 4. Construct the nullspace
    num_free = n - r
    N = np.zeros((n, num_free), dtype=float)
    if num_free > 0:
        for j in range(num_free):
            # For each free variable, find the corresponding nullspace vector
            rhs = -A_fac[P[:r], Q[r + j]]
            
            # Solve U_basic * z_basic = rhs using back substitution
            z_basic_null = np.zeros(r, dtype=float)
            for i in range(r - 1, -1, -1):
                s = np.dot(A_fac[P[i], Q[i + 1:r]], z_basic_null[i + 1:r])
                pivot = A_fac[P[i], Q[i]]
                z_basic_null[i] = (rhs[i] - s) / pivot
            
            # Assemble the full nullspace vector
            N[Q[:r], j] = z_basic_null
            N[Q[r + j], j] = 1.0

    # 5. Check for consistency
    if r < m and np.max(np.abs(y[r:])) > tol:
        c = None
    
    return c, N
