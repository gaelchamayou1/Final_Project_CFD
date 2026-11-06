def diffusion_step(u_star, v_star, nu, dt, dx, dy):
    """
    Etape 2 de la méthode à pas fractionné : diffusion visqueuse.

        u** = u* + nu * dt * laplacien(u*)

    Schéma : Euler explicite en temps + différences centrées d'ordre 2 en espace
    (Cours, §5.6.1.2, éq. 5.147).

    Tableaux de taille (Nx+1, Ny+1), indice [i, j] <-> (x, y), noeuds des parois inclus.
    Les noeuds de bord ne sont pas modifiés (conditions aux limites imposées ailleurs).

    Stabilité : Fox + Foy <= 1/2.
    """
    # Nombres de Fourier
    Fox = nu * dt / dx**2
    Foy = nu * dt / dy**2

    def diffuse(phi):
        phi_new = phi.copy()
        phi_new[1:-1, 1:-1] = (
            phi[1:-1, 1:-1]
            + Fox * (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1])   # direction x
            + Foy * (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2])   # direction y
        )
        return phi_new

    u_ss = diffuse(u_star)
    v_ss = diffuse(v_star)
    return u_ss, v_ss
