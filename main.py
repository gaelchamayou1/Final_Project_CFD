"""
Lid-driven cavity solver: fractional-step method (Resolution_method.pdf, slide 9).

    Steps 1 + 2 : prediction u**      -> advection_diffusion.step_prediction (Euler or RK3)
    Step 3      : Poisson for p       -> mass_balance.sor_solver
    Step 4      : correction u^{n+1}  -> projection.correction_step
"""
import os
import numpy as np

from advection_diffusion import step_prediction, apply_boundary_conditions
from mass_balance import sor_solver
from projection import correction_step
from plot import plot_results, animate_results


# ---------------- Physical parameters ----------------
L = 1.0          # cavity size (m)
U = 1.0          # lid velocity (m/s)
Re = 100.0       # Reynolds number
rho = 1.0        # density (kg/m^3)
nu = U * L / Re  # kinematic viscosity (m^2/s)

# ---------------- Numerical parameters ----------------
N = 201                         # nodes per direction (walls included)
dx = dy = L / (N - 1)
# Explicit stability: CFL < 1 (advection) and nu*dt/dx^2 < 1/4 (2D diffusion)
dt = 0.5 * min(dx / U, dx**2 / (4 * nu))
snapshot_every = 25            # store (t, u, v) every 25 steps for the animation

# ---------------- Stopping criterion ----------------
# Steady state: dimensionless residual R = max|u^{n+1} - u^n| * L / (U * dt) < steady_tol
steady_tol = 1e-6
# The lid still accelerates until U_lid >= 0.999 U, i.e. t > ln(1000) * L/U (~6.9 tau_lid):
# before that the flow cannot be steady, so the test is not checked.
t_ramp = np.log(1000) * L / U
# Safety only: a run that never becomes steady still ends
t_max = 1000 * L / U

# ---------------- Methods used for each step ----------------
METHODS = {
    "time":      "rk3",         # Steps 1 + 2 time scheme: "euler" or "rk3"
    "advection": "upwind2",     # Step 1: "upwind1" or "upwind2"
    "diffusion": "centred2",    # Step 2: 2nd-order centred
    "poisson":   "sor",         # Step 3: SOR
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
stop_reason = None
while stop_reason is None:
    t += dt
    n += 1

    # Steps 1 + 2: prediction (BCs at the new time t^{n+1})
    u_ss, v_ss = step_prediction(u, v, dx, dy, dt, nu, t, U_ref=U, L=L,
                                 advection=METHODS["advection"], time=METHODS["time"])

    # Step 3: pressure (previous p used as initial guess)
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

    # Dimensionless residual (rate of change of the velocity, in units of U per L/U)
    R = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v))) * L / (U * dt)
    u, v = u_new, v_new

    if n % snapshot_every == 0:
        snapshots.append((t, u.copy(), v.copy()))

    if n % 100 == 0:
        print(f"n={n:6d}  t={t:7.3f}  K={K:.5e}  P={P:.5e}  R={R:.2e}")

    # Stopping criterion
    if t > t_ramp and R < steady_tol:
        stop_reason = "steady"
        print(f"Steady state reached at t = {t:.3f} s (R = {R:.2e})")
    elif t >= t_max:
        stop_reason = "t_max (not converged)"
        print(f"WARNING: t_max = {t_max} s reached without steady state (R = {R:.2e})")

# ---------------- Save results ----------------
# CSV: run description in the header, then the time history t, K(t), P(t)
header = "\n".join(
    [f"{step} = {method}" for step, method in METHODS.items()]
    + [f"Re = {Re}", f"N = {N}", f"dt = {dt}", f"t_final = {t}",
       f"K_final = {K_hist[-1]}", f"P_final = {P_hist[-1]}",
       f"stop_reason = {stop_reason}", f"residual_final = {R}", f"steady_tol = {steady_tol}", "",
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
