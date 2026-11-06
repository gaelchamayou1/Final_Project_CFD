import numpy as np
import matplotlib.pyplot as plt


def plot_results(u, v, L, Re, times, K_hist, P_hist):
    """
    Plots the final velocity field (magnitude + streamlines)
    and the time history of K(t) and P(t).
    Arrays u, v of shape (Ny, Nx), index [j, i] <-> (y, x).
    """
    Ny, Nx = u.shape
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    c = ax[0].contourf(x, y, np.sqrt(u**2 + v**2), levels=50, cmap="viridis")
    ax[0].streamplot(x, y, u, v, color="w", density=1.5, linewidth=0.6)
    fig.colorbar(c, ax=ax[0], label="|u| (m/s)")
    ax[0].set_title(f"Velocity, Re = {Re:.0f}")
    ax[0].set_aspect("equal")

    ax[1].plot(times, K_hist)
    ax[1].set_xlabel("t (s)")
    ax[1].set_title("Kinetic energy K(t)")

    ax[2].plot(times, P_hist)
    ax[2].set_xlabel("t (s)")
    ax[2].set_title("Lid power P(t)")

    plt.tight_layout()
    plt.show()
