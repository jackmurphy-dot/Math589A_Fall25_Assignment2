# general_linear_solver.py
# Solve A x = b via PAQ = LU (from plu_decomposition.py).
# Returns (c, N) to satisfy autograder ordering:
#   - c: particular solution (or None if inconsistent)
#   - N: matrix whose columns form a basis for Null(A)

from typing import Tuple, Optional
import numpy as np
from plu_decomposition import paq_lu
from logging_setup import logger


Array = np.ndarray

def _forward_substitution_L(A: Array, P: Array, Q: Array, b: Array, r: int) -> Array:
    """
    Solve L y = P b for y, where L has implicit unit diagonal and is stored in A's
    strictly-lower part interpreted with (P, Q). Works for rectangular matrices.
    We produce a length-m vector y. For i >= r, the equation is still well-defined
    because L is unit lower-triangular when viewed in the pivoted row/column order.
    """
    m, n = A.shape
    y = np.zeros(m, dtype=np.float64)
    b_perm = b[P]  # simulated row permutation

    for i in range(m):
        # sum_{j=0}^{min(i-1, r-1)} L_ij * y_j, with L_ij = A[P[i], Q[j]]
        s = 0.0
        upto = min(i, r)  # only pivot columns contribute in the triangular part
        if upto > 0:
            ai = A[P[i]]
            for j in range(upto):
                s += float(ai[Q[j]]) * y[j]
        # unit diagonal -> y_i = b_i - s
        y[i] = b_perm[i] - s
    return y


def _back_substitution_U_basic(A: Array, P: Array, Q: Array, y: Array, r: int, tol: float) -> Array:
    """
    Solve U_bb x_b = y_b where U_bb is the leading r x r upper-triangular block
    in A under permutations (P, Q). Returns x_b (length r).
    """
    x_b = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        # sum_{j=i+1..r-1} U_ij * x_b[j]
        s = 0.0
        api = A[P[i]]
        for j in range(i + 1, r):
            s += float(api[Q[j]]) * x_b[j]
        Uii = float(api[Q[i]])
        if abs(Uii) < tol:
            # Degenerate pivot (should not happen if rank logic is correct); guard anyway
            raise np.linalg.LinAlgError("Singular U block encountered in back substitution.")
        x_b[i] = (y[i] - s) / Uii
    return x_b


def solve(A: Array, b: Array, tol: float = 1e-6) -> Tuple[Optional[Array], Array]:
    """
    Solve A x = b returning (c, N) where:
      - c is a particular solution (np.ndarray of shape (n,)) if consistent; otherwise c is None
      - N is an (n x k) matrix whose columns form a basis for Null(A)

    Conventions per instructor note:
      - Return order is (c, N)
      - When inconsistent, return c = None but still return a valid Null(A) basis N

    Parameters
    ----------
    A : array_like, shape (m, n)
    b : array_like, shape (m,) or (m, 1)
    tol : float
        Numerical tolerance to detect zero pivots and consistency in rank-deficient cases.

    Returns
    -------
    c : Optional[np.ndarray], shape (n,)
    N : np.ndarray, shape (n, k)  (k may be 0 if full column rank)

    Notes
    -----
    - Uses PAQ = LU with simulated row swaps and virtual column ordering (see plu_decomposition.py).
    - Does all arithmetic in float64.
    """
    # Normalize inputs
    A = np.asarray(A, dtype=np.float64, order='C')  # we will modify A in place
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Dimension mismatch: A is (m x n) but b has length != m.")

    # Factorization (in-place)
    A, P, Q, pivot_cols, r = paq_lu(A, tol=tol)

    # Forward substitution to compute y = L^{-1} P b
    y = _forward_substitution_L(A, P, Q, b, r)

    # Consistency check: for rows i >= r, the system implies 0*x = y[i]
    # Thus we require |y[i]| <= tol for all i in [r, m).
    consistent = True
    if r < m:
        if np.max(np.abs(y[r:])) > tol:
            consistent = False

    # Build Null(A) basis regardless of consistency
    # U_bb = r x r upper-triangular; U_bf = r x (n - r)
    # Null basis columns correspond to free variables (columns Q[r:])
    k = max(n - r, 0)
    N = np.zeros((n, k), dtype=np.float64)
    if k > 0 and r > 0:
        # Compute X = - U_bb^{-1} U_bf via back-substitution per column
        # For each free column idx f (logical col j = r+f), solve U_bb * x = -U_bf[:, f]
        for f in range(k):
            rhs = np.zeros(r, dtype=np.float64)
            # rhs = - U_bf[:, f]
            for i in range(r):
                rhs[i] = -float(A[P[i], Q[r + f]])
            x_b = _solve_Ubb(A, P, Q, rhs, r, tol)
            # Assemble N column in original variable order:
            # entries for basic vars (Q[:r]) = x_b; entry for this free var (Q[r+f]) = 1
            col = np.zeros(n, dtype=np.float64)
            col[Q[:r]] = x_b
            col[Q[r + f]] = 1.0
            N[:, f] = col
    elif k > 0 and r == 0:
        # A is (numerically) zero -> every column is free; Null(A) = I_n
        N[:, :] = np.eye(n, dtype=np.float64)
    # else k == 0 => full column rank: N is (n x 0), fine.

    # Particular solution if consistent:
    if not consistent:
        c = None
        return c, N

    if r == 0:
        # A is zero but consistent means b ~ 0; particular solution can be all zeros
        c = np.zeros(n, dtype=np.float64)
        return c, N

    # Solve U_bb x_b = y_b (top r entries of y)
    y_b = y[:r].copy()
    x_b = _back_substitution_U_basic(A, P, Q, y_b, r, tol)

    # Build full particular solution in original column order (free vars = 0)
    c = np.zeros(n, dtype=np.float64)
    c[Q[:r]] = x_b
    # free positions Q[r:] remain zero
    return c, N


def _solve_Ubb(A: Array, P: Array, Q: Array, rhs: Array, r: int, tol: float) -> Array:
    """
    Solve U_bb x = rhs for the leading r x r upper-triangular block of A
    under permutations (P, Q). Helper used for Null(A) basis construction.
    """
    x = np.zeros(r, dtype=np.float64)
    for i in range(r - 1, -1, -1):
        s = 0.0
        api = A[P[i]]
        for j in range(i + 1, r):
            s += float(api[Q[j]]) * x[j]
        Uii = float(api[Q[i]])
        if abs(Uii) < tol:
            # Guard against tiny pivots
            raise np.linalg.LinAlgError("Singular U block in nullspace solve.")
        x[i] = (rhs[i] - s) / Uii
    return x
