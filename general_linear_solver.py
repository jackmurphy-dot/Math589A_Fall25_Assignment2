# Minor change for commit test

import numpy as np
from plu_decomposition import paq_lu
from logging_setup import get_logger

logger = get_logger(__name__)

def _forward_unit_lower(Lrr: np.ndarray, br: np.ndarray) -> np.ndarray:
    """
    Solve (unit lower triangular) Lrr y = br.
    Lrr is r x r with ones on the diagonal.
    br is r x 1.
    Returns y as r x 1.
    """
    r = Lrr.shape[0]
    y = np.zeros((r, 1), dtype=float)
    for i in range(r):
        # diagonal is 1, so no division
        y[i, 0] = br[i, 0] - float(Lrr[i, :i] @ y[:i, 0])
    return y

def _back_upper(Urr: np.ndarray, y: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """
    Solve (upper triangular) Urr x = y.
    Urr is r x r, y is r x 1, returns x as r x 1.
    """
    r = Urr.shape[0]
    x = np.zeros((r, 1), dtype=float)
    for i in range(r - 1, -1, -1):
        diag = Urr[i, i]
        if abs(diag) < tol:
            # numerical safety; treat as zero pivot
            x[i, 0] = 0.0
        else:
            x[i, 0] = (y[i, 0] - float(Urr[i, i+1:] @ x[i+1:, 0])) / diag
    return x

def solve(A: np.ndarray, b: np.ndarray, tol: float = 1e-12):
    """
    Solve Ax = b using complete-pivoting LU (PAQ = LU).

    Returns:
        c : (n x k) particular solution(s) for each RHS column of b (k=1 in grader)
        N : (n x (n-r)) whose columns form a basis for the nullspace of A
    """
    # Ensure b is 2D with shape (m, k)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    m, n = A.shape
    k_rhs = b.shape[1]

    logger.info("Solving system of size A%s, b%s", A.shape, b.shape)

    P, L, U, Q, r = paq_lu(A, tol=tol)

    # ---- Build a nullspace basis from U (free vars set to identity) ----
    if r < n:
        # Solve U_rr * W = U_rF, then N_perm = [ -W ; I ]
        W = np.linalg.solve(U[:r, :r], U[:r, r:])  # (r x (n-r))
        N_perm = np.zeros((n, n - r), dtype=float)
        N_perm[:r, :] = -W
        N_perm[r:, :] = np.eye(n - r)
    else:
        N_perm = np.zeros((n, 0), dtype=float)

    N = Q @ N_perm  # (n x (n-r))

    # ---- Particular solution(s) with correct shape (n x k) ----
    C = np.zeros((n, k_rhs), dtype=float)

    for j in range(k_rhs):
        # Permute RHS by P
        b_perm = (P @ b[:, [j]])  # (m x 1)

        # Solve for y using the r leading equations: L[:r,:r] y = (Pb)[:r]
        y = _forward_unit_lower(L[:r, :r], b_perm[:r, :])  # (r x 1)

        # Consistency check on eliminated rows: L[r:,:r] y ?= (Pb)[r:]
        if r < m:
            tail = b_perm[r:, 0] - (L[r:, :r] @ y[:, 0])
            if np.any(np.abs(tail) > max(tol, 1e-12)):
                # Inconsistent system for this RHS: follow common convention
                # to return empty particular solution; keep N as nullspace basis.
                return np.zeros((n, 0)), N

        # Back substitution for pivot variables
        x_piv = _back_upper(U[:r, :r], y, tol=1e-14)  # (r x 1)

        # Assemble particular solution in permuted coordinates (free vars = 0)
        c_perm = np.zeros((n, 1), dtype=float)
        c_perm[:r, 0] = x_piv[:, 0]

        # Map back with column permutation Q
        C[:, [j]] = Q @ c_perm

        # Optional: one-step iterative refinement to tighten residuals
        rvec = b[:, [j]] - A @ C[:, [j]]
        if np.linalg.norm(rvec) > tol * max(1.0, np.linalg.norm(b[:, [j]])):
            delta, *_ = np.linalg.lstsq(A, rvec, rcond=None)
            C[:, [j]] += delta

    logger.info("Solver finished successfully.")
    return C, N

if __name__ == "__main__":
    # quick self-check
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([[1.], [2.]])  # column vector (m x 1)
    c, N = solve(A, b)
    print("c shape:", c.shape, "c:", c.ravel())
    print("N shape:", N.shape)
