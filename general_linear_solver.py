import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def forward_substitution(A, P, Q, b, r):
    """Solve L y = P b for y (L has implicit unit diagonal)."""
    m = len(P)
    y = np.zeros(m)
    b_perm = b[P]
    for i in range(r):
        y[i] = b_perm[i] - np.dot(A[P[i], Q[:i]], y[:i])
    return y

def backward_substitution(A, P, Q, y, r, tol=1e-10):
    """Solve U z = y for z (upper triangular)."""
    z = np.zeros(r)
    for i in range(r - 1, -1, -1):
        piv = A[P[i], Q[i]]
        if abs(piv) < tol:
            raise np.linalg.LinAlgError("Singular matrix in backward substitution.")
        z[i] = (y[i] - np.dot(A[P[i], Q[i + 1:r]], z[i + 1:r])) / piv
    return z

def solve(A: Array, b: Array, tol: float = 1e-10) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b using the PAQ = LU factorization.
    Returns (particular_solution, nullspace_basis).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol)

    # Forward substitution (L y = P b)
    y = forward_substitution(A_fac, P, Q, b, r)

    # Extract U_B and U_F blocks (pivot and free columns)
    U_B = np.zeros((r, r))
    for i in range(r):
        U_B[i, :] = A_fac[P[i], Q[:r]]
    U_B = np.triu(U_B)

    U_F = np.zeros((r, n - r))
    if n > r:
        for i in range(r):
            U_F[i, :] = A_fac[P[i], Q[r:]]

    # Particular solution: U_B * x_B = y[:r]
    c = np.zeros(n)
    if r > 0:
        x_B = np.linalg.solve(U_B, y[:r])
        c[Q[:r]] = x_B

    # Nullspace matrix N = Q [ -U_B^{-1} U_F ; I ]
    if n > r:
        UB_inv_UF = np.linalg.solve(U_B, U_F)
        top = -UB_inv_UF
        bottom = np.eye(n - r)
        N_reordered = np.vstack((top, bottom))  # shape (n, n-r) but permuted
        N = np.zeros((n, n - r))
        N[Q, :] = N_reordered  # undo Q permutation
    else:
        N = np.zeros((n, 0))

    return c, N
