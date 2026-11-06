import numpy as np
import time
from numba import njit

@njit(fastmath=True)
def compute_prediction(u, v, dx, dy, dt, nu):
    """
    Calcule la vitesse prédite (u**, v**) en combinant :
      - Advection : schéma Upwind 1er ordre
      - Diffusion : schéma FTCS centré 2nd ordre
    Entrées :
        u, v   : Tableaux 2D (Ny, Nx) des vitesses au temps n
        dx, dy : Pas d'espace (m)
        dt     : Pas de temps (s)
        nu     : Viscosité cinématique (m²/s)
    Sorties :
        u_ss, v_ss : Champs de vitesse prédits (u**, v**)
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

            # 1. Advection Upwind (1er ordre)
            # Dérivées de u
            du_dx = (u_c - u[j, i - 1]) / dx if u_c > 0.0 else (u[j, i + 1] - u_c) / dx
            du_dy = (u_c - u[j - 1, i]) / dy if v_c > 0.0 else (u[j + 1, i] - u_c) / dy

            # Dérivées de v
            dv_dx = (v_c - v[j, i - 1]) / dx if u_c > 0.0 else (v[j, i + 1] - v_c) / dx
            dv_dy = (v_c - v[j - 1, i]) / dy if v_c > 0.0 else (v[j + 1, i] - v_c) / dy

            adv_u = u_c * du_dx + v_c * du_dy
            adv_v = u_c * dv_dx + v_c * dv_dy

            # 2. Diffusion Centrée (2nd ordre)
            diff_u = (u[j, i + 1] - 2.0 * u_c + u[j, i - 1]) / dx2 + \
                     (u[j + 1, i] - 2.0 * u_c + u[j - 1, i]) / dy2

            diff_v = (v[j, i + 1] - 2.0 * v_c + v[j, i - 1]) / dx2 + \
                     (v[j + 1, i] - 2.0 * v_c + v[j - 1, i]) / dy2

            # 3. Étape d'Euler explicite
            u_ss[j, i] = u_c + dt * (-adv_u + nu * diff_u)
            v_ss[j, i] = v_c + dt * (-adv_v + nu * diff_v)

    return u_ss, v_ss


@njit(fastmath=True)
def apply_boundary_conditions(u_ss, v_ss, U_lid):
    """
    Applique les conditions aux limites sur le champ prédit (u**, v**).
    - Parois fixes : u = 0, v = 0 (Bas, Gauche, Droite)
    - Paroi supérieure mobile : u = U_lid(t), v = 0 (Haut)
    """
    # Bas (y = 0)
    u_ss[0, :] = 0.0
    v_ss[0, :] = 0.0

    # Haut (y = Ly)
    u_ss[-1, :] = U_lid
    v_ss[-1, :] = 0.0

    # Gauche (x = 0)
    u_ss[:, 0] = 0.0
    v_ss[:, 0] = 0.0

    # Droite (x = Lx)
    u_ss[:, -1] = 0.0
    v_ss[:, -1] = 0.0

    return u_ss, v_ss


def step_prediction(u, v, dx, dy, dt, nu, t, U_ref=1.0, L=1.0):
    """
    Interface appelée par la boucle temporelle globale dans main.py.
    """
    # Loi de rampe du couvercle : U_t(t) = U * (1 - exp(-t / tau)) avec tau = L / U
    tau_lid = L / U_ref
    U_lid = U_ref * (1.0 - np.exp(-t / tau_lid))

    # Étape 1 + 2 : Advection + Diffusion
    u_ss, v_ss = compute_prediction(u, v, dx, dy, dt, nu)

    # Conditions aux limites
    u_ss, v_ss = apply_boundary_conditions(u_ss, v_ss, U_lid)

    return u_ss, v_ss




# Paramètres physiques et numériques (Cavité Re = 100)
Re = 100.0
L = 1.0
U_ref = 1.0
nu = U_ref * L / Re

Nx, Ny = 65, 65
dx = L / (Nx - 1)
dy = L / (Ny - 1)

# Vérification du pas de temps sous critères CFL et Fourier
dt_cfl = 0.5 * min(dx, dy) / U_ref
dt_fo = 0.2 * min(dx**2, dy**2) / nu
dt = min(dt_cfl, dt_fo)

u = np.zeros((Ny, Nx), dtype=np.float64)
v = np.zeros((Ny, Nx), dtype=np.float64)

# Test d'exécution
t_start = time.time()
u_ss, v_ss = step_prediction(u, v, dx, dy, dt, nu, t=0.5, U_ref=U_ref, L=L)
t_elapsed = time.time() - t_start

print(f"Validation du module Prédiction (Étapes 1+2) réussie.")
print(f"Pas de temps dt retenu   : {dt:.6f} s")
print(f"Temps d'exécution        : {t_elapsed*1e3:.3f} ms")
print(f"Vitesse max u** obtenue  : {np.max(u_ss):.4f} m/s")