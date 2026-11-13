import numpy as np
from numba import njit


def jacobi_solver(u_ss, v_ss, dx, dy, dt, rho, p_init=None, tol=1e-8, max_iter=100_000):
    """
    Step 3 of the fractional-step method: Poisson equation for the pressure.

        laplacian(p) = rho/dt * div(u**)

    Solver: Jacobi method (Course, §4.4), 2nd-order centred differences.
    Walls: dp/dn = 0, i.e. p_wall = p_first_interior_node (Course, eq. 4.28).

    Arrays of shape (Ny, Nx), index [j, i] <-> (y, x), wall nodes included.
    The pressure is defined up to a constant: mean(p) = 0 is imposed.
    """
    # Right-hand side on interior nodes
    div = ((u_ss[1:-1, 2:] - u_ss[1:-1, :-2]) / (2 * dx)
           + (v_ss[2:, 1:-1] - v_ss[:-2, 1:-1]) / (2 * dy))
    b = rho / dt * div
    b -= b.mean()   # compatibility condition with Neumann BCs

    p = np.zeros_like(u_ss) if p_init is None else p_init.copy()

    for _ in range(max_iter):
        p_old = p.copy()

        # Jacobi iteration on interior nodes
        p[1:-1, 1:-1] = (
            (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dx**2
            + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dy**2
            - b
        ) / (2 / dx**2 + 2 / dy**2)

        # Walls: dp/dn = 0
        p[:, 0], p[:, -1] = p[:, 1], p[:, -2]
        p[0, :], p[-1, :] = p[1, :], p[-2, :]

        p -= p.mean()

        if np.max(np.abs(p - p_old)) < tol:
            break

    return p


def sor_solver(u_ss, v_ss, dx, dy, dt, rho, p_init=None, tol=1e-8, max_iter=100_000):
    """
    Step 3 of the fractional-step method: Poisson equation for the pressure.

        laplacian(p) = rho/dt * div(u**)

    Solver: SOR, i.e. over-relaxed Gauss-Seidel (Course, §4.5.3, eq. 4.133),
    with the optimal coefficient omega = 2 / (1 + sin(pi/N)) (Course, eq. 4.137).
    Same discretisation, walls (dp/dn = 0) and mean(p) = 0 as jacobi_solver.
    """
    # Right-hand side on interior nodes (same as jacobi_solver)
    div = ((u_ss[1:-1, 2:] - u_ss[1:-1, :-2]) / (2 * dx)
           + (v_ss[2:, 1:-1] - v_ss[:-2, 1:-1]) / (2 * dy))
    b = rho / dt * div
    b -= b.mean()   # compatibility condition with Neumann BCs

    p = np.zeros_like(u_ss) if p_init is None else p_init.copy()

    N = max(u_ss.shape) - 1
    omega = 2.0 / (1.0 + np.sin(np.pi / N))

    return _sor_iterations(p, b, dx, dy, omega, tol, max_iter)


@njit(fastmath=True)
def _sor_iterations(p, b, dx, dy, omega, tol, max_iter):
    """SOR sweeps: new values p[j, i] are used as soon as they are computed."""
    Ny, Nx = p.shape
    dx2 = dx * dx
    dy2 = dy * dy
    diag = 2.0 / dx2 + 2.0 / dy2

    for _ in range(max_iter):
        max_change = 0.0

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                # Gauss-Seidel value (left and bottom neighbours already updated)
                p_gs = ((p[j, i + 1] + p[j, i - 1]) / dx2
                        + (p[j + 1, i] + p[j - 1, i]) / dy2
                        - b[j - 1, i - 1]) / diag
                # Over-relaxation
                p_new = (1.0 - omega) * p[j, i] + omega * p_gs
                max_change = max(max_change, abs(p_new - p[j, i]))
                p[j, i] = p_new

        # Walls: dp/dn = 0
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        p -= p.mean()

        if max_change < tol:
            break

    return p
