def correction_step(u_ss, v_ss, p, rho0, dt, dx, dy):
    """
    Etape 4 de la méthode à pas fractionné : correction par le gradient de pression.

        u^{n+1} = u** - dt/rho0 * grad(p)

    Gradient : différences centrées d'ordre 2 (Cours, §2.2 et §7.3.2).

    Tableaux de taille (Nx+1, Ny+1), indice [i, j] <-> (x, y), noeuds des parois inclus.
    Les noeuds de bord ne sont pas modifiés (conditions aux limites imposées ailleurs).

    Attention (grille collocalisée) : le gradient centré n'utilise pas p[i, j] lui-même,
    une pression en damier (+1, -1, +1, ...) a donc un gradient nul -> à surveiller.
    """
    u_new = u_ss.copy()
    v_new = v_ss.copy()

    # dp/dx et dp/dy centrés sur les noeuds intérieurs
    dpdx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
    dpdy = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy)

    u_new[1:-1, 1:-1] = u_ss[1:-1, 1:-1] - dt / rho0 * dpdx
    v_new[1:-1, 1:-1] = v_ss[1:-1, 1:-1] - dt / rho0 * dpdy

    return u_new, v_new
