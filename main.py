"""
Lid-driven cavity solver: fractional-step method (Resolution_method.pdf, slide 9).

    Steps 1 + 2 : prediction u**      -> advection_diffusion.step_prediction
    Step 3      : Poisson for p       -> mass_balance.compute_pressure
    Step 4      : correction u^{n+1}  -> projection.correction_step
"""
import numpy as np

from advection_diffusion import step_prediction, apply_boundary_conditions
from mass_balance import compute_pressure
from projection import correction_step
from plot import plot_results


# ---------------- Physical parameters ----------------
L = 1.0          # cavity size (m)
U = 1.0          # lid velocity (m/s)
Re = 100.0       # Reynolds number
rho = 1.0        # density (kg/m^3)
nu = U * L / Re  # kinematic viscosity (m^2/s)

# ---------------- Numerical parameters ----------------
N = 41                         # nodes per direction (walls included)
dx = dy = L / (N - 1)
# Explicit stability: CFL < 1 (advection) and nu*dt/dx^2 < 1/4 (2D diffusion)
dt = 0.5 * min(dx / U, dx**2 / (4 * nu))
t_end = 20.0                   # final time (s), i.e. 20 tau_lid
steady_tol = 1e-7              # stop when max|u^{n+1} - u^n| / dt < steady_tol

# ---------------- Initial state: fluid at rest ----------------
u = np.zeros((N, N))   # index [j, i] <-> (y, x)
v = np.zeros((N, N))
p = np.zeros((N, N))

times, K_hist, P_hist = [], [], []
t = 0.0
n = 0

# ---------------- Time loop ----------------
while t < t_end:
    t += dt
    n += 1

    # Steps 1 + 2: prediction (BCs at the new time t^{n+1})
    u_ss, v_ss = step_prediction(u, v, dx, dy, dt, nu, t, U_ref=U, L=L)

    # Step 3: pressure (previous p used as initial guess)
    p = compute_pressure(u_ss, v_ss, dx, dy, dt, rho, p_init=p)

    # Step 4: correction
    u_new, v_new = correction_step(u_ss, v_ss, p, dx, dy, dt, rho)

    # Re-impose wall velocities (the correction only updates interior nodes)
    U_lid = U * (1.0 - np.exp(-t / (L / U)))
    u_new, v_new = apply_boundary_conditions(u_new, v_new, U_lid)

    # Diagnostics: kinetic energy K(t) and lid power P(t)
    K = 0.5 * rho * np.sum(u_new**2 + v_new**2) * dx * dy
    dudy_top = (u_new[-1, :] - u_new[-2, :]) / dy
    P = np.trapezoid(rho * nu * dudy_top * U_lid, dx=dx)

    times.append(t)
    K_hist.append(K)
    P_hist.append(P)

    # Steady-state check
    change = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v))) / dt
    u, v = u_new, v_new

    if n % 100 == 0:
        print(f"n={n:6d}  t={t:7.3f}  K={K:.5e}  P={P:.5e}  change={change:.2e}")

    if t > 5 * L / U and change < steady_tol:
        print(f"Steady state reached at t = {t:.3f} s")
        break

# ---------------- Plots ----------------
plot_results(u, v, L, Re, times, K_hist, P_hist)
