"""
Lid-driven cavity solver: fractional-step method (Resolution_method.pdf, slide 9).

    Steps 1 + 2 : prediction u**      -> advection_diffusion.step_prediction
    Step 3      : Poisson for p       -> mass_balance.jacobi_solver or sor_solver
    Step 4      : correction u^{n+1}  -> projection.correction_step
"""
import os
import numpy as np

from advection_diffusion import step_prediction, apply_boundary_conditions
from mass_balance import jacobi_solver, sor_solver
from projection import correction_step
from plot import plot_results, animate_results


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
snapshot_every = 25            # store (t, u, v) every 25 steps for the animation

# ---------------- Methods used for each step ----------------
METHODS = {
    "advection": "upwind1",     # Step 1: 1st-order upwind
    "diffusion": "FTCS",        # Step 2: explicit Euler + centred 2nd order
    "poisson":   "sor",         # Step 3: "jacobi" or "sor"
    "gradient":  "centred2",    # Step 4: 2nd-order centred gradient
}
run_name = "_".join(METHODS.values()) + f"_Re{Re:.0f}_N{N}"
os.makedirs("results", exist_ok=True)

# ---------------- Initial state: fluid at rest ----------------
u = np.zeros((N, N))   # index [j, i] <-> (y, x)
v = np.zeros((N, N))
p = np.zeros((N, N))

times, K_hist, P_hist = [], [], []
snapshots = []         # list of (t, u, v) used to animate the flow
t = 0.0
n = 0

# ---------------- Time loop ----------------
while t < t_end:
    t += dt
    n += 1

    # Steps 1 + 2: prediction (BCs at the new time t^{n+1})
    u_ss, v_ss = step_prediction(u, v, dx, dy, dt, nu, t, U_ref=U, L=L)

    # Step 3: pressure (previous p used as initial guess)
    if METHODS["poisson"] == "jacobi":
        p = jacobi_solver(u_ss, v_ss, dx, dy, dt, rho, p_init=p)
    else:
        p = sor_solver(u_ss, v_ss, dx, dy, dt, rho, p_init=p)

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

    if n % snapshot_every == 0:
        snapshots.append((t, u.copy(), v.copy()))

    if n % 100 == 0:
        print(f"n={n:6d}  t={t:7.3f}  K={K:.5e}  P={P:.5e}  change={change:.2e}")

    if t > 5 * L / U and change < steady_tol:
        print(f"Steady state reached at t = {t:.3f} s")
        break

# ---------------- Save results ----------------
# CSV: run description in the header, then the time history t, K(t), P(t)
header = "\n".join(
    [f"{step} = {method}" for step, method in METHODS.items()]
    + [f"Re = {Re}", f"N = {N}", f"dt = {dt}", f"t_final = {t}",
       f"K_final = {K_hist[-1]}", f"P_final = {P_hist[-1]}", "",
       "t,K,P"]
)
np.savetxt(f"results/{run_name}.csv", np.column_stack([times, K_hist, P_hist]),
           delimiter=",", header=header)

# PNG: figure annotated with the methods
title = " | ".join(f"{step}: {method}" for step, method in METHODS.items())
plot_results(u, v, L, Re, times, K_hist, P_hist, title=title,
             save_path=f"results/{run_name}.png")

# GIF: evolution of the velocity field versus time
animate_results(snapshots, L, title=title, save_path=f"results/{run_name}.gif")
