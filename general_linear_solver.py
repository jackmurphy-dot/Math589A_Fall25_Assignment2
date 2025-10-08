import numpy as np
from typing import Tuple, Optional
from logging_setup import logger
from plu_decomposition import paq_lu

Array = np.ndarray

def _forward_substitution_Ly_eq_Pb(A: Array, P: np.ndarray, Q: np.ndarray, b: Array, r: int) -> Array:
    """
    Solve L y = P b for the first r components (L has unit diagonal).
    IMPORTANT: L is stored in A on columns Q[:r]; the strictly lower triangle
    of L at step i is in A[P[i], Q[:i]].
    """
    m = len(P)
    y = np.zeros(m, dtype=float)
    b_perm = b[P]
    for i in range(r):
        # L[i, :i] dot y[:i]  where L[i, j] is stored at A[P[i], Q[j]]
        if i > 0:
            y[i] = b_perm[i] - np.dot(A[P[i], Q[:i]], y[:i])
        else:
            y[i] = b_perm[i]
    return y

def _extract_UB_UF(A: Array, P: np.ndarray, Q: np.ndarray, r: int, n: int) -> Tuple[Array, Array]:
    """
    Build U_B (r x r) and U_F (r x (n-r)) from the factored A using P and Q.
    U_B is the leading r pivot-columns restricted to the first r pivot-rows,
    and is upper-triangular (up to small roundoff).
    """
    if r == 0:
        return np.zeros((0, 0)), np.zeros((0, n))

    # U_B rows are P[:r], cols are Q[:r]
    U_B = A[P[:r], :][:, Q[:r]].copy()
    # Ensure upper-triangular numerically
    U_B = np.triu(U_B)

    # U_F rows are same pivot rows, cols are Q[r:]
    if n > r:
        U_F = A[P[:r], :][:, Q[r:]].copy()
    else:
        U_F = np.zeros((r, 0))

    return U_B, U_F

def solve(A: Array, b: Array, tol: float = 1e-8) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b using PAQ = LU with BOTH row and column pivoting.
    Returns:
        c : one particular solution (n,) with free vars set to zero; or None if no pivot found
        N : nullspace basis (n x (n - r)) mapping free params to full solutions
    Notes:
        From derivation.tex:
            x = Q [ U_B^{-1} (L^{-1} P b)_pivot ; 0 ] + Q [ -U_B^{-1} U_F ; I ] x_free
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    A_fac, P, Q, r = paq_lu(A, tol=tol)

    # If rank is 0, particular solution is zero only if b is zero in pivot rows (none exist);
    # we return c = zeros (conventional) and N spans entire space.
    # The general code below already handles this.
    # 1) Ly = P b  (compute the first r entries of y consistently with L storage)
    y = _forward_substitution_Ly_eq_Pb(A_fac, P, Q, b, r)

    # 2) Extract U_B (pivot block) and U_F (free block)
    U_B, U_F = _extract_UB_UF(A_fac, P, Q, r, n)

    # 3) Particular solution: x_B solves U_B x_B = y[:r]
    c = np.zeros(n, dtype=float)
    if r > 0:
        # Solve upper-triangular system robustly
        # (np.linalg.solve will also work since U_B is triangular and invertible)
        x_B = np.linalg.solve(U_B, y[:r])
        # Map from reordered (pivot-first) coordinates back to original variable order: x = Q x'
        # Here c' = [x_B ; 0], so c[Q[:r]] = x_B
        c[Q[:r]] = x_B
        # The free positions Q[r:] remain zero -> that's fine for a particular solution

    # 4) Nullspace: N = Q [ -U_B^{-1} U_F ; I_{n-r} ]
    if n > r:
        UB_inv_UF = np.linalg.solve(U_B, U_F) if r > 0 else np.zeros((0, n - r))
        top = -UB_inv_UF                      # shape (r, n-r)
        bottom = np.eye(n - r)                # shape (n-r, n-r)
        N_reordered = np.vstack([top, bottom])  # shape (n, n-r) in pivot-first variable order
        # Map back to original column order: x = Q x'
        N = np.zeros((n, n - r), dtype=float)
        N[Q, :] = N_reordered
    else:
        N = np.zeros((n, 0), dtype=float)

    return c, N
