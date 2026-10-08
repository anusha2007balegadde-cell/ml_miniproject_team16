"""
Reference ("true") solutions we compare the PINN against.
These replace a downloaded dataset: PINNs do not need one.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def consolidation_exact(x, t, n_terms=1000):
    """
    Problem 1, paper Eq. (4.5): analytical solution of the dimensionless
    1D consolidation problem
        p = sum over m = 1, 3, 5, ... of
            4/(m*pi) * sin(m*pi*x/2) * exp(-m^2 * pi^2 * t / 4)
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    p = np.zeros(np.broadcast(x, t).shape)
    for m in range(1, 2 * n_terms, 2):
        p += 4.0 / (m * np.pi) * np.sin(m * np.pi * x / 2.0) \
             * np.exp(-(m * np.pi) ** 2 * t / 4.0)
    return p


def heat_fd_reference(n=201):
    """
    Problem 2, paper Eq. (4.6)-(4.7):  T_xx + T_yy + 1 = 0 on [-1,1]^2,
    T = 0 on the boundary.

    The paper used a finite element solution from the deal.II library.
    We solve the same equation with a standard second-order finite-difference
    scheme on an n x n grid (5-point Laplacian), which is just as accurate
    for this smooth problem.

    returns x (1D grid), y (1D grid), T with T[j, i] = T(x_i, y_j)
    """
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    h = x[1] - x[0]
    m = n - 2  # number of interior points per direction

    D = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    I = sp.identity(m)
    A = (sp.kron(I, D) + sp.kron(D, I)) / h ** 2   # discrete -Laplacian
    b = np.ones(m * m)                              # right-hand side = 1

    T_inner = spla.spsolve(A.tocsc(), b)
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = T_inner.reshape(m, m)           # [y index, x index]
    return x, y, T


def spring_exact(t, m=1.0, c=0.4, k=4.0, u0=1.0, v0=1.0):
    """
    Problem 3, paper Eq. (4.9): free vibration with viscous damping
    (underdamped case). Used to generate the 'observations' and to check
    the PINN.
    """
    t = np.asarray(t, dtype=float)
    wn = np.sqrt(k / m)                 # natural frequency
    xi = c / (2.0 * m * wn)             # damping ratio
    wd = wn * np.sqrt(1.0 - xi ** 2)    # damped frequency
    return np.exp(-xi * wn * t) * (u0 * np.cos(wd * t)
                                   + (v0 + xi * wn * u0) / wd * np.sin(wd * t))
