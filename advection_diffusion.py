import numpy as np
from numba import njit


@njit(fastmath=True)
def upwind_derivative(f, j, i, dj, di, c, h, order):
    """
    Upwind derivative of f at node (j, i) in the direction (dj, di) = (0, 1) for x or (1, 0) for y.
      c     : advection velocity in that direction (gives the upwind side)
      h     : grid spacing in that direction
      order : 1 -> 1st-order upwind
              2 -> 2nd-order upwind (Course, cheat sheet: 2nd-order backward/forward formulas)
    Next to a wall the node f[j - 2*dj, i - 2*di] (or + side) does not exist,
    so the 1st-order formula is used there.
    """
    Ny, Nx = f.shape
    f0 = f[j, i]

    if c > 0.0:   # information comes from the "minus" side
        jm2, im2 = j - 2 * dj, i - 2 * di
        if order == 2 and jm2 >= 0 and im2 >= 0:
            return (3.0 * f0 - 4.0 * f[j - dj, i - di] + f[jm2, im2]) / (2.0 * h)
        return (f0 - f[j - dj, i - di]) / h
    else:         # information comes from the "plus" side
        jp2, ip2 = j + 2 * dj, i + 2 * di
        if order == 2 and jp2 <= Ny - 1 and ip2 <= Nx - 1:
            return (-3.0 * f0 + 4.0 * f[j + dj, i + di] - f[jp2, ip2]) / (2.0 * h)
        return (f[j + dj, i + di] - f0) / h


@njit(fastmath=True)
def compute_rhs(u, v, dx, dy, nu, order):
    """
    Right-hand side of the prediction step (Steps 1 + 2) on interior nodes:
        rhs_u = -(u du/dx + v du/dy) + nu * laplacian(u)
        rhs_v = -(u dv/dx + v dv/dy) + nu * laplacian(v)
      - Advection: upwind, 1st or 2nd order (order = 1 or 2)
      - Diffusion: 2nd-order centred
    Wall nodes are left at 0 (boundary conditions are imposed elsewhere).
    Arrays of shape (Ny, Nx), index [j, i] <-> (y, x).
    """
    Ny, Nx = u.shape
    rhs_u = np.zeros_like(u)
    rhs_v = np.zeros_like(v)

    dx2 = dx * dx
    dy2 = dy * dy

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            u_c = u[j, i]
            v_c = v[j, i]

            # 1. Upwind advection
            du_dx = upwind_derivative(u, j, i, 0, 1, u_c, dx, order)
            du_dy = upwind_derivative(u, j, i, 1, 0, v_c, dy, order)
            dv_dx = upwind_derivative(v, j, i, 0, 1, u_c, dx, order)
            dv_dy = upwind_derivative(v, j, i, 1, 0, v_c, dy, order)

            adv_u = u_c * du_dx + v_c * du_dy
            adv_v = u_c * dv_dx + v_c * dv_dy

            # 2. Centred diffusion (2nd order)
            diff_u = (u[j, i + 1] - 2.0 * u_c + u[j, i - 1]) / dx2 + \
                     (u[j + 1, i] - 2.0 * u_c + u[j - 1, i]) / dy2

            diff_v = (v[j, i + 1] - 2.0 * v_c + v[j, i - 1]) / dx2 + \
                     (v[j + 1, i] - 2.0 * v_c + v[j - 1, i]) / dy2

            rhs_u[j, i] = -adv_u + nu * diff_u
            rhs_v[j, i] = -adv_v + nu * diff_v

    return rhs_u, rhs_v


@njit(fastmath=True)
def apply_boundary_conditions(u_ss, v_ss, U_lid):
    """
    Applies the boundary conditions to the predicted field (u**, v**).
    - Fixed walls: u = 0, v = 0 (bottom, left, right)
    - Moving top wall: u = U_lid(t), v = 0 (top)
    """
    # Bottom (y = 0)
    u_ss[0, :] = 0.0
    v_ss[0, :] = 0.0

    # Top (y = Ly)
    u_ss[-1, :] = U_lid
    v_ss[-1, :] = 0.0

    # Left (x = 0)
    u_ss[:, 0] = 0.0
    v_ss[:, 0] = 0.0

    # Right (x = Lx)
    u_ss[:, -1] = 0.0
    v_ss[:, -1] = 0.0

    return u_ss, v_ss


def lid_velocity(t, U_ref, L):
    """Lid ramp law: U_t(t) = U * (1 - exp(-t / tau)) with tau = L / U."""
    return U_ref * (1.0 - np.exp(-t / (L / U_ref)))


def step_prediction(u, v, dx, dy, dt, nu, t, U_ref=1.0, L=1.0,
                    advection="upwind1", time="euler"):
    """
    Steps 1 + 2 of the fractional-step method: predicted velocity (u**, v**)
    from the velocity at time t^n, with BCs at the new time t = t^{n+1}.
      advection : "upwind1" or "upwind2"
      time      : "euler" -> Forward Euler
                  "rk3"   -> RK3 Heun (Course, §3.9.4):
                     k1 = f(u^n)
                     k2 = f(u^n + dt/3 k1)
                     k3 = f(u^n + 2dt/3 k2)
                     u** = u^n + dt/4 (k1 + 3 k3)
    """
    order = 1 if advection == "upwind1" else 2
    t_n = t - dt

    if time == "euler":
        ku, kv = compute_rhs(u, v, dx, dy, nu, order)
        u_ss = u + dt * ku
        v_ss = v + dt * kv

    else:  # "rk3"
        k1u, k1v = compute_rhs(u, v, dx, dy, nu, order)

        u1, v1 = apply_boundary_conditions(u + dt / 3 * k1u, v + dt / 3 * k1v,
                                           lid_velocity(t_n + dt / 3, U_ref, L))
        k2u, k2v = compute_rhs(u1, v1, dx, dy, nu, order)

        u2, v2 = apply_boundary_conditions(u + 2 * dt / 3 * k2u, v + 2 * dt / 3 * k2v,
                                           lid_velocity(t_n + 2 * dt / 3, U_ref, L))
        k3u, k3v = compute_rhs(u2, v2, dx, dy, nu, order)

        u_ss = u + dt / 4 * (k1u + 3 * k3u)
        v_ss = v + dt / 4 * (k1v + 3 * k3v)

    # Boundary conditions at t^{n+1}
    u_ss, v_ss = apply_boundary_conditions(u_ss, v_ss, lid_velocity(t, U_ref, L))

    return u_ss, v_ss
