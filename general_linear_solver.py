import numpy as np
from plu_decomposition import paq_lu
from logging_setup import get_logger

logger = get_logger(__name__)

def _forward_substitution(L, b):
    """
    Solve Ly = b for y, where L is lower-triangular with unit diagonal.
    """
    m, n = L.shape
    y = np.zeros(m)
    for i in range(min(m, n)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def _back_substitution(U, y):
    """
    Solve Ux = y for x, where U is upper-triangular.
    """
    n = U.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            x[i] = 0.0
        else:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def solve(A, b, tol=1e-10):
    """
    Solve the linear system Ax = b using the PAQ = LU decomposition with complete pivoting.

    Returns:
        c : particular solution (None if system inconsistent)
        N : nullspace basis matrix (columns form basis for nullspace of A)
    """
    logger.info("Starting solver with matrix A shape %s and vector b shape %s", A.shape, b.shape)

    m, n = A.shape
    P, L, U, Q, r = paq_lu(A, tol=tol)

    # Apply row permutation to b
    b_perm = P @ b

    # Forward substitution on the first r equations
    y = _forward_substitution(L, b_perm)

    # --- Corrected consistency check (after elimination) ---
    if r < m:
        tail = b_perm[r:] - L[r:, :r] @ y
        if np.any(np.abs(tail) > max(tol, 1e-12)):
            logger.warning("System inconsistent: residual tail exceeds tolerance.")
            return None, None
    # --------------------------------------------------------

    # Back substitution to solve for the pivot variables
    x1 = _back_substitution(U[:r, :r], y[:r])

    # Construct particular solution and nullspace basis in permuted coordinates
    c_perm = np.zeros(n)
    c_perm[:r] = x1

    if r < n:
        N_perm = np.zeros((n, n - r))
        N_perm[:r, :] = -np.linalg.solve(U[:r, :r], U[:r, r:])
        N_perm[r:, :] = np.eye(n - r)
    else:
        N_perm = np.zeros((n, 0))

    # Map back to original variable ordering using Q
    c = Q @ c_perm
    N = Q @ N_perm

    logger.info("Solver completed successfully.")
    return c, N


if __name__ == "__main__":
    # Simple self-test for debugging
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([1., 2.])
    c, N = solve(A, b)
    print("Particular solution c:", c)
    print("Nullspace basis N:", N)
