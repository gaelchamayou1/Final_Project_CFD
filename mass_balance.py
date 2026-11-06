"""Calcul de pression de la cavité : étape 3 uniquement, avec Jacobi.

Dépendance : NumPy (python -m pip install numpy).
Ce module est à importer dans votre programme CFD ; il ne fabrique aucune donnée.
Il ne calcule ni l'advection, ni la diffusion, ni la correction des vitesses.

HYPOTHÈSE DE RACCORDEMENT : maillage cartésien uniforme décalé (MAC).
- pression p[j, i] au centre des cellules : forme (ny, nx) ;
- vitesse horizontale u_etoile2 aux faces verticales : (ny, nx + 1) ;
- vitesse verticale v_etoile2 aux faces horizontales : (ny + 1, nx).
Les tableaux fournis ne contiennent aucune couche fantôme supplémentaire.
L'indice j augmente avec y et l'indice i avec x.
Les faces normales des quatre parois doivent avoir une vitesse nulle.
Le couvercle glisse tangentiellement : sa vitesse verticale reste nulle.

Avec une autre disposition des vitesses, adapter le calcul de divergence,
NE PAS simplement redimensionner les tableaux pour obtenir les formes attendues.
Vous pouvez aussi appeler poisson_jacobi(b, dx, dy) si b est déjà calculé.

Unités : rho est la masse volumique constante ; dt, dx, dy et les vitesses
sont exprimés dans un système cohérent. En variables adimensionnées avec
p / (rho_ref * U_ref**2), utiliser rho=1.

Utilisation dans la boucle de temps commune, après les étapes 1 et 2 :
    from pression_jacobi import calculer_pression
    p, infos = calculer_pression(u_etoile2, v_etoile2, dx, dy, dt, rho)
Au pas suivant, on peut ajouter p_initial=p pour accélérer la convergence.
"""

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
    # Donc b doit avoir une moyenne nulle. Cette moyenne arithmétique convient
    # ici parce que toutes les cellules ont le même volume.
    # On refuse une incompatibilité significative : la supprimer silencieusement
    # pourrait cacher une erreur dans les vitesses ou leur divergence.
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

        # JACOBI : aucun élément de p n'est modifié pendant le calcul du nouveau
        # tableau. Tous les voisins sont donc ceux de l'itération k.
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


def calculer_pression(u_etoile2, v_etoile2, dx, dy, dt, rho,
                     *, p_initial=None, rtol=1e-8, atol=1e-10,
                     max_iter=100_000):
    """Construit b à partir des vitesses reçues, puis résout Laplacien(p) = b.

    u_etoile2 et v_etoile2 sont les résultats APRÈS advection et diffusion.
    Retour : p de forme (ny, nx) et dictionnaire infos de convergence.
    Les tableaux de vitesse ne sont pas modifiés.
    La pression est définie à une constante près : on impose moyenne(p)=0.
    Une erreur est signalée si le raccordement ou la convergence est incorrect.
    """
    for nom, valeur in (("dx", dx), ("dy", dy), ("dt", dt), ("rho", rho)):
        if not np.isscalar(valeur) or not np.isfinite(valeur) or valeur <= 0:
            raise ValueError(f"{nom} doit être un nombre fini strictement positif.")

    u = np.asarray(u_etoile2, dtype=float)
    v = np.asarray(v_etoile2, dtype=float)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("Les vitesses doivent être des tableaux 2D.")
    ny, nx = u.shape[0], u.shape[1] - 1
    if nx < 2 or ny < 2 or v.shape != (ny + 1, nx):
        raise ValueError(
            "Maillage MAC attendu : u de forme (ny, nx+1), "
            "v de forme (ny+1, nx), nx et ny >= 2."
        )
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Les vitesses contiennent des valeurs non finies.")

    # Le problème de pression prescrit impose dp/dn=0. Les vitesses
    # intermédiaires doivent donc déjà respecter l'imperméabilité des murs.
    # On refuse les entrées incorrectes plutôt que de les changer à votre insu.
    # Les valeurs normales prescrites à zéro doivent être affectées explicitement
    # par le code commun avant cet appel (ce ne sont pas les vitesses tangentielles).
    if (np.any(u[:, 0] != 0.0) or np.any(u[:, -1] != 0.0)
            or np.any(v[0, :] != 0.0) or np.any(v[-1, :] != 0.0)):
        raise ValueError(
            "Imposer u[:,0]=u[:,-1]=0 et v[0,:]=v[-1,:]=0 "
            "aux faces des murs avant le calcul de pression."
        )

    # DIVERGENCE AU CENTRE DE CHAQUE CELLULE :
    # dérivée en x = (u_face_droite - u_face_gauche) / dx ;
    # dérivée en y = (v_face_haute - v_face_basse) / dy.
    # Les différences de faces partagées se compensent dans la somme globale.
    # Avec des flux normaux nuls aux murs, la moyenne est donc nulle à l'arrondi près.
    divergence = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy

    # SECOND MEMBRE DU TP : b = rho/dt * divergence(u**, v**).
    # Cette fois, aucune solution analytique n'intervient dans les données.
    b = (rho / dt) * divergence

    # On délègue uniquement la résolution elliptique à Jacobi.
    # p_initial est une estimation de départ, jamais une pression imposée.
    p, infos = poisson_jacobi(
        b, dx, dy, p_initial=p_initial,
        rtol=rtol, atol=atol, max_iter=max_iter
    )
    infos["divergence_entree_inf"] = float(np.max(np.abs(divergence)))
    return p, infos


if __name__ == "__main__":
    # Sans les données de vos collègues, aucun calcul physique ne peut démarrer.
    print("Module prêt : importez calculer_pression dans votre programme CFD.")
    print("Fournissez u**, v**, dx, dy, dt et rho après les étapes 1 et 2.")
