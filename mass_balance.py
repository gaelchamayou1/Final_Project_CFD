import numpy as np


def compute_pressure(u_ss, v_ss, dx, dy, dt, rho, p_init=None, tol=1e-8, max_iter=100_000):
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
