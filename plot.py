import numpy as np
import matplotlib.pyplot as plt


def plot_results(u, v, L, Re, times, K_hist, P_hist, title="", save_path=None):
    """
    Plots the final velocity field (magnitude + streamlines)
    and the time history of K(t) and P(t).
    title     : figure title (e.g. the methods used for each step)
    save_path : if given, the figure is saved there as a PNG
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

    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()


def animate_results(snapshots, L, title="", save_path=None):
    """
    Animates the velocity field versus time: |u| in colour + velocity arrows.
    snapshots : list of (t, u, v) stored during the time loop
    save_path : if given, the animation is saved there as a GIF
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    t0, u0, v0 = snapshots[0]
    Ny, Nx = u0.shape
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    s = max(1, Nx // 20)   # draw one arrow every s nodes

    # Fixed colour scale: maximum speed over the whole run
    vmax = max(np.max(np.sqrt(u**2 + v**2)) for _, u, v in snapshots)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    mesh = ax.pcolormesh(x, y, np.sqrt(u0**2 + v0**2), cmap="viridis",
                         vmin=0, vmax=vmax, shading="gouraud")
    # Fixed arrow scale: the fastest arrow is as long as the spacing between arrows
    arrows = ax.quiver(x[::s], y[::s], u0[::s, ::s], v0[::s, ::s], color="w",
                       scale_units="xy", scale=vmax / (s * (x[1] - x[0])))
    fig.colorbar(mesh, ax=ax, label="|u| (m/s)")
    ax.set_aspect("equal")
    fig.suptitle(title, fontsize=8)

    def update(k):
        t, u, v = snapshots[k]
        mesh.set_array(np.sqrt(u**2 + v**2).ravel())
        arrows.set_UVC(u[::s, ::s], v[::s, ::s])
        ax.set_title(f"t = {t:.2f} s")
        return mesh, arrows

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=50)
    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=20))
    plt.show()
    return anim
