import numpy as np
from numba import njit

@njit(fastmath=True)
def compute_prediction(u, v, dx, dy, dt, nu):
    """
    Computes the predicted velocity (u**, v**) by combining:
      - Advection: 1st-order upwind scheme
      - Diffusion: 2nd-order centred FTCS scheme
    Inputs:
        u, v   : 2D arrays (Ny, Nx) of the velocities at time n
        dx, dy : Grid spacing (m)
        dt     : Time step (s)
        nu     : Kinematic viscosity (m²/s)
    Outputs:
        u_ss, v_ss : Predicted velocity fields (u**, v**)
    """
    Ny, Nx = u.shape
    u_ss = np.copy(u)
    v_ss = np.copy(v)

    dx2 = dx * dx
    dy2 = dy * dy

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            u_c = u[j, i]
            v_c = v[j, i]

            # 1. Upwind advection (1st order)
            # Derivatives of u
            du_dx = (u_c - u[j, i - 1]) / dx if u_c > 0.0 else (u[j, i + 1] - u_c) / dx
            du_dy = (u_c - u[j - 1, i]) / dy if v_c > 0.0 else (u[j + 1, i] - u_c) / dy

            # Derivatives of v
            dv_dx = (v_c - v[j, i - 1]) / dx if u_c > 0.0 else (v[j, i + 1] - v_c) / dx
            dv_dy = (v_c - v[j - 1, i]) / dy if v_c > 0.0 else (v[j + 1, i] - v_c) / dy

            adv_u = u_c * du_dx + v_c * du_dy
            adv_v = u_c * dv_dx + v_c * dv_dy

            # 2. Centred diffusion (2nd order)
            diff_u = (u[j, i + 1] - 2.0 * u_c + u[j, i - 1]) / dx2 + \
                     (u[j + 1, i] - 2.0 * u_c + u[j - 1, i]) / dy2

            diff_v = (v[j, i + 1] - 2.0 * v_c + v[j, i - 1]) / dx2 + \
                     (v[j + 1, i] - 2.0 * v_c + v[j - 1, i]) / dy2

            # 3. Explicit Euler step
            u_ss[j, i] = u_c + dt * (-adv_u + nu * diff_u)
            v_ss[j, i] = v_c + dt * (-adv_v + nu * diff_v)

    return u_ss, v_ss


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


def step_prediction(u, v, dx, dy, dt, nu, t, U_ref=1.0, L=1.0):
    """
    Interface called by the global time loop in main.py.
    """
    # Lid ramp law: U_t(t) = U * (1 - exp(-t / tau)) with tau = L / U
    tau_lid = L / U_ref
    U_lid = U_ref * (1.0 - np.exp(-t / tau_lid))

    # Steps 1 + 2: advection + diffusion
    u_ss, v_ss = compute_prediction(u, v, dx, dy, dt, nu)

    # Boundary conditions
    u_ss, v_ss = apply_boundary_conditions(u_ss, v_ss, U_lid)

    return u_ss, v_ss
