import numpy as np
from plu_decomposition import paq_lu
from logging_setup import get_logger

logger = get_logger(__name__)

def _forward_substitution(L, b):
    """Solve Ly = b for y (L lower-triangular, unit diagonal)."""
    m, n = L.shape
    y = np.zeros(m)
    for i in range(min(m, n)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def _back_substitution(U, y):
    """Solve Ux = y for x (U upper-triangular)."""
    n = U.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            x[i] = 0.0
        else:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def solve(A, b, tol=1e-12):
    """
    Solves Ax = b using complete-pivoting LU (PAQ = LU).
    Returns (c, N):
        c : particular solution (empty array if inconsistent)
        N : matrix whose columns form a basis of the nullspace of A
    """
    logger.info("Solving system of size A%s, b%s", A.shape, b.shape)

    b = b.reshape(-1)
    m, n = A.shape
    P, L, U, Q, r = paq_lu(A, tol)

    # Apply row permutation
    b_perm = P @ b
    y = _forward_substitution(L, b_perm)

    # Consistency check
    if r < m:
        tail = b_perm[r:] - L[r:, :r] @ y[:r]
        if np.any(np.abs(tail) > max(tol, 1e-10)):
            logger.warning("Inconsistent system detected.")
            return np.array([]), np.zeros((n, 0))

    # Robust least-squares solve for x1 (avoids round-off)
    x1, *_ = np.linalg.lstsq(U[:r, :r], y[:r], rcond=None)

    # Build particular solution and nullspace
    c_perm = np.zeros(n)
    c_perm[:r] = x1

    if r < n:
        N_perm = np.zeros((n, n - r))
        N_perm[:r, :] = -np.linalg.solve(U[:r, :r], U[:r, r:])
        N_perm[r:, :] = np.eye(n - r)
    else:
        N_perm = np.zeros((n, 0))

    c = Q @ c_perm
    N = Q @ N_perm

    # Iterative refinement for high precision
    rvec = b - A @ c
    if np.linalg.norm(rvec) > tol * np.linalg.norm(b):
        delta, *_ = np.linalg.lstsq(A, rvec, rcond=None)
        c += delta

    logger.info("Solver finished successfully.")
    return c, N

if __name__ == "__main__":
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([[1.], [2.]])
    c, N = solve(A, b)
    print("c =", c)
    print("N =", N)
