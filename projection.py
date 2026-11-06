import numpy as np

def correction_step(u_ss, v_ss, p, dx, dy, dt, rho):
    """
    Step 4 of the fractional-step method: pressure-gradient correction.

        u^{n+1} = u** - dt/rho * grad(p)

    Gradient: 2nd-order centred differences (Course, §2.2 and §7.3.2).

    Arrays of shape (Ny, Nx), index [j, i] <-> (y, x), wall nodes included.
    Boundary nodes are not modified (boundary conditions are imposed elsewhere).

    Warning (collocated grid): the centred gradient does not use p[j, i] itself,
    so a checkerboard pressure (+1, -1, +1, ...) has a zero gradient -> watch out.
    """
    u_new = u_ss.copy()
    v_new = v_ss.copy()

    # Centred dp/dx and dp/dy on interior nodes
    dpdx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    dpdy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)

    u_new[1:-1, 1:-1] = u_ss[1:-1, 1:-1] - dt / rho * dpdx
    v_new[1:-1, 1:-1] = v_ss[1:-1, 1:-1] - dt / rho * dpdy

    return u_new, v_new
