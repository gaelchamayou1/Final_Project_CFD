"""Étape 3 du TP CFD : équation de Poisson, méthode de Jacobi.

Exécution de l'exemple : python poisson_jacobi.py
Dépendance : numpy. Ajouter --plot pour afficher des figures (matplotlib).

On résout Laplacien(p) = b dans [0, Lx] x [0, Ly], avec dp/dn = 0
sur les quatre côtés et moyenne(p) = 0. Aucune correction de vitesse ici.

CONVENTION À PARTAGER AVEC LES COLLÈGUES
Les pressions et b sont stockés AUX CENTRES des cellules d'une grille uniforme.
Un tableau a la forme (ny, nx) : p[j, i], j pour y et i pour x.
Les parois sont aux faces extérieures, pas aux centres des cellules extrêmes.
Le second membre futur sera b = (rho / dt) * divergence(u**, v**).
Son calcul dépend du placement des vitesses : il reste donc extérieur au solveur.
"""

import argparse
import numpy as np


def voisins_neumann(p):
    """Retourne les voisins est, ouest, nord et sud, avec Neumann nul."""
    # On ajoute une couche de cellules fictives, appelée cellules fantômes.
    # À gauche, par exemple, p_fantome = p_premiere_cellule :
    # (p_premiere_cellule - p_fantome) / dx = 0 à la paroi.
    # Le même principe s'applique aux trois autres côtés. Les coins fantômes
    # ne sont pas utilisés par le stencil à cinq points.
    g = np.pad(p, pad_width=1, mode="edge")
    return g[1:-1, 2:], g[1:-1, :-2], g[2:, 1:-1], g[:-2, 1:-1]


def laplacien_neumann(p, dx, dy):
    """Laplacien discret, avec le même traitement des bords que le solveur."""
    est, ouest, nord, sud = voisins_neumann(p)
    return (est - 2.0 * p + ouest) / dx**2 + (nord - 2.0 * p + sud) / dy**2


def poisson_jacobi(b, dx, dy, *, p_initial=None,
                   rtol=1e-8, atol=1e-10, max_iter=100_000):
    """Résout Poisson par itérations simultanées avec cellules fantômes.

    Paramètres
    ----------
    b : tableau (ny, nx)
        Second membre connu ; sa moyenne doit être nulle pour Neumann nul.
    dx, dy : nombres positifs
        Largeur et hauteur des cellules, par exemple dx = Lx / nx.
    p_initial : tableau optionnel de même forme que b
        Estimation initiale ; dans le TP, on pourra utiliser la pression du
        pas de temps précédent. Par défaut, on démarre avec une pression nulle.
    rtol, atol : tolérances relative et absolue du résidu
        Arrêt si ||b - Laplacien(p)||_inf <= atol + rtol * ||b||_inf.
        atol a les mêmes unités que b ; rtol est sans dimension.
    max_iter : entier positif
        Protection contre une boucle infinie. L'échec déclenche une exception.

    Retours
    -------
    p : pression calculée, de moyenne nulle
    infos : nombre d'itérations, résidus et seuil d'arrêt

    Note sur Jacobi aux bords
    ------------------------
    Toutes les valeurs de droite, cellules fantômes comprises, proviennent
    de l'ancienne itération. On conserve le dénominateur du stencil intérieur.
    C'est l'itération de Jacobi du stencil avec fantômes retardés ; aux bords,
    elle diffère de Jacobi appliqué à la matrice après élimination des fantômes.
    Cette précision importe : ce dernier peut présenter une oscillation en
    damier pour un problème de Neumann pur. Ici, les valeurs propres aux bords
    sont conservées dans la mise à jour, sans introduire de méthode SOR.
    """
    b = np.array(b, dtype=float, copy=True)
    if b.ndim != 2 or min(b.shape) < 2 or not np.all(np.isfinite(b)):
        raise ValueError("b doit être un tableau fini (ny, nx), avec nx, ny >= 2.")
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("dx et dy doivent être finis et strictement positifs.")
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("Les tolérances doivent être finies et positives ou nulles.")
    if rtol == 0 and atol == 0:
        raise ValueError("Au moins une tolérance doit être strictement positive.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter doit être un entier positif.")

    # CONDITION D'EXISTENCE : la somme du Laplacien discret est nulle.
    # Donc b doit avoir une moyenne nulle. Cette moyenne arithmétique convient ici parce que toutes les cellules ont le même volume.
    # On refuse une incompatibilité significative : la supprimer silencieusement pourrait cacher une erreur dans les vitesses ou leur divergence.
    moyenne_b = float(np.mean(b))
    norme_b = float(np.max(np.abs(b)))
    seuil_compatibilite = 100 * np.finfo(float).eps * norme_b
    if abs(moyenne_b) > seuil_compatibilite:
        raise ValueError(
            f"Second membre incompatible avec Neumann nul : moyenne(b)={moyenne_b:.3e}. "
            "Vérifier le calcul de divergence et les flux aux parois."
        )
    # Seule une erreur de moyenne de l'ordre de l'arrondi est retirée.
    b -= moyenne_b

    if p_initial is None:
        p = np.zeros_like(b)
    else:
        p = np.array(p_initial, dtype=float, copy=True)
        if p.shape != b.shape or not np.all(np.isfinite(p)):
            raise ValueError("p_initial doit être fini et de même forme que b.")
    # FIXER LA RÉFÉRENCE : p et p + constante satisfont la même équation.
    p -= np.mean(p)

    inv_dx2, inv_dy2 = 1.0 / dx**2, 1.0 / dy**2
    denominateur = 2.0 * (inv_dx2 + inv_dy2)
    seuil = atol + rtol * float(np.max(np.abs(b)))
    historique = []

    # k compte les itérations du solveur, PAS les pas de temps du fluide.
    # b reste inchangé pendant toute cette boucle.
    for k in range(max_iter + 1):
        # Le résidu mesure si l'équation est satisfaite, pas seulement si p varie
        # peu. On le calcule sur toutes les cellules, y compris celles des bords.
        residu = b - laplacien_neumann(p, dx, dy)
        norme_residu = float(np.max(np.abs(residu)))
        historique.append(norme_residu)
        if norme_residu <= seuil:
            return p, {
                "iterations": k,
                "residu_inf": norme_residu,
                "seuil": seuil,
                "historique_residu": np.array(historique),
                "moyenne_b_retiree": moyenne_b,
            }
        if k == max_iter:
            break

        # JACOBI : aucun élément de p n'est modifié pendant le calcul du nouveau tableau. Tous les voisins sont donc ceux de l'itération k.
        est, ouest, nord, sud = voisins_neumann(p)
        p_nouveau = (
            (est + ouest) * inv_dx2 + (nord + sud) * inv_dy2 - b
        ) / denominateur
        # Le signe devant b est négatif car on résout Laplacien(p) = +b.
        # Si dx = dy = h : p_nouveau = (somme_des_4_voisins - h*h*b) / 4.
        p_nouveau -= np.mean(p_nouveau)
        p = p_nouveau

    raise RuntimeError(
        f"Jacobi n'a pas convergé en {max_iter} itérations : "
        f"résidu={norme_residu:.3e}, seuil={seuil:.3e}."
    )


def exemple_analytique(nx=32, ny=32, afficher=False):
    """Vérification autonome, sans vitesses intermédiaires ni calcul CFD complet."""
    lx, ly = 1.0, 1.0
    dx, dy = lx / nx, ly / ny
    # Le premier centre est à dx/2 du mur, le dernier à Lx - dx/2.
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    # Solution choisie : p_exact = cos(pi*x/Lx) * cos(pi*y/Ly).
    # Sa dérivée normale est nulle sur tous les murs, car sin(0)=sin(pi)=0.
    # Son Laplacien continu vaut -[(pi/Lx)^2 + (pi/Ly)^2] * p_exact.
    p_exact = np.cos(np.pi * X / lx) * np.cos(np.pi * Y / ly)
    b = -((np.pi / lx)**2 + (np.pi / ly)**2) * p_exact
    p, infos = poisson_jacobi(b, dx, dy)
    erreur = float(np.max(np.abs(p - p_exact)))

    print(f"Maillage : {nx} x {ny} cellules")
    print(f"Itérations Jacobi : {infos['iterations']}")
    print(f"Résidu maximal : {infos['residu_inf']:.3e}")
    print(f"Seuil demandé : {infos['seuil']:.3e}")
    print(f"Moyenne de p : {np.mean(p):.3e}")
    print(f"Erreur maximale par rapport à la solution continue : {erreur:.3e}")
    # L'erreur à la solution continue ne devient pas nulle en itérant davantage :
    # il reste une erreur de discrétisation spatiale. Raffiner la grille doit
    # la réduire d'environ un facteur 4 quand dx et dy sont divisés par 2.

    if afficher:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, champ, titre in zip(
            axes[:2], (p, p - p_exact), ("Pression Jacobi", "Erreur spatiale")
        ):
            carte = ax.pcolormesh(X, Y, champ, shading="nearest", cmap="coolwarm")
            fig.colorbar(carte, ax=ax)
            ax.set(xlabel="x", ylabel="y", title=titre, aspect="equal")
        axes[2].semilogy(infos["historique_residu"], label="Résidu maximal")
        axes[2].axhline(infos["seuil"], linestyle="--", color="black", label="Seuil")
        axes[2].set(xlabel="Itération k", ylabel="Norme infinie", title="Convergence")
        axes[2].legend()
        plt.show()
    return erreur


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--plot", action="store_true", help="Afficher les figures.")
    args = parser.parse_args()
    if args.nx < 2 or args.ny < 2:
        parser.error("nx et ny doivent être supérieurs ou égaux à 2.")
    exemple_analytique(args.nx, args.ny, args.plot)
