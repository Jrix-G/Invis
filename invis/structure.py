"""Structure par intersection de rayons multi-vues.

Ce que ce module remplace
-------------------------
La premiere version trianguait par paires d'images consecutives, puis prenait
la mediane des resultats par point suivi. Deux defauts de fond:

1. *Les reperes ne sont pas les memes*. Chaque triangulation rendait un point
   dans le repere du drone a cet instant-la. Le drone bouge entre deux images:
   moyenner ces valeurs revient a melanger des coordonnees exprimees dans des
   reperes differents. L'erreur reste petite tant que le drone est lent, mais
   elle est systematique, pas aleatoire -- une mediane ne l'enleve pas.

2. *La base reste courte*. Deux images consecutives sont separees de quelques
   centimetres. Or l'incertitude de profondeur varie comme l'inverse de la
   base: repeter une mauvaise mesure sept fois vaut bien moins qu'une seule
   mesure faite sur une base sept fois plus longue.

La methode retenue
------------------
Chaque observation d'un point donne un rayon dans le repere *monde*: un
sommet (le centre optique) et une direction. Le point cherche est celui qui
passe au plus pres de tous ces rayons a la fois. Ecrit au moindre carre, cela
donne un systeme 3x3:

    A X = b,   A = somme (I - d d^T),   b = somme (I - d d^T) C

ou d est la direction unitaire du rayon et C le centre optique. La solution
est exacte et directe -- ni iteration, ni initialisation.

Pourquoi c'est aussi peu couteux
--------------------------------
A et b sont des *sommes*. Chaque nouvelle observation s'y ajoute en temps
constant, et l'historique des observations n'a pas besoin d'etre conserve: un
point suivi coute treize nombres, quel que soit le nombre d'images ou il
apparait. La resolution est un systeme 3x3, faite en une seule fois pour tous
les points de l'image.

L'incertitude vient avec la solution
------------------------------------
Les valeurs propres de A mesurent l'ouverture angulaire des visees. Quand
toutes les visees sont paralleles, A devient singuliere: le point peut se
trouver n'importe ou le long du rayon, et l'incertitude explose. C'est
exactement le comportement physique attendu, et il sort du calcul au lieu
d'etre impose par un seuil.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RayBundle:
    """Accumulation des visees par point suivi, resolution groupee.

    Le stockage est un tableau a lignes fixes plutot qu'un dictionnaire de
    tableaux: la mise a jour et la resolution deviennent des operations numpy
    sur des tranches, sans boucle Python sur les points.
    """

    def __init__(self, capacity: int = 4096, sigma_px: float = 0.5,
                 window: float = 12.0) -> None:
        self.capacity = int(capacity)
        self.sigma_px = float(sigma_px)
        # Oubli exponentiel des anciennes visees.
        #
        # Accumuler sans fin parait gratuit -- plus de visees, meilleure
        # solution -- et c'est faux pour deux raisons. Le suivi de points
        # glisse lentement sur une texture repetitive, donc les visees les plus
        # anciennes ne concernent plus tout a fait le meme detail. Et la pose
        # d'ou elles ont ete prises est entachee d'une derive qui, elle, ne
        # cesse de croitre. Une visee vieille de dix secondes n'est donc pas
        # une visee de plus: c'est une visee fausse.
        #
        # Le facteur decroit d'un facteur e sur `window` images. Le cout reste
        # celui d'une multiplication: aucune histoire n'est conservee.
        self.window = max(1.0, float(window))
        self._decay = float(np.exp(-1.0 / self.window))
        self._slot: dict = {}
        self._A = np.zeros((self.capacity, 3, 3), dtype=np.float64)
        self._b = np.zeros((self.capacity, 3), dtype=np.float64)
        self._n = np.zeros(self.capacity, dtype=np.int32)
        self._first_C = np.zeros((self.capacity, 3), dtype=np.float64)
        self._span = np.zeros(self.capacity, dtype=np.float64)
        # Correspondance inverse ligne -> identifiant. Sans elle, reprendre une
        # ligne laisserait l'ancienne entree du dictionnaire pointer dessus, et
        # deux points suivis partageraient la meme accumulation.
        self._owner = np.full(self.capacity, -1, dtype=np.int64)
        self._last_C = np.zeros(3, dtype=np.float64)
        self._next_row = 0

    def reset(self) -> None:
        self._slot.clear()
        self._next_row = 0
        self._n[:] = 0
        self._owner[:] = -1

    def __len__(self) -> int:
        return len(self._slot)

    # -- accumulation ------------------------------------------------------

    def _rows_for(self, ids: np.ndarray) -> np.ndarray:
        """Ligne de stockage de chaque identifiant, en creant les manquantes.

        Le tampon est circulaire: quand il est plein, les lignes les plus
        anciennes sont reprises. Un point suivi assez longtemps pour faire le
        tour du tampon a de toute facon deja ete resolu.
        """
        rows = np.empty(len(ids), dtype=np.int64)
        slot = self._slot
        for i, tid in enumerate(ids):
            key = int(tid)
            row = slot.get(key)
            if row is None:
                row = self._next_row
                self._next_row = (self._next_row + 1) % self.capacity
                # La ligne reprise appartenait peut-etre a un point encore
                # suivi: il faut retirer son entree, sinon deux identifiants
                # partageraient la meme accumulation.
                previous = int(self._owner[row])
                if previous >= 0:
                    slot.pop(previous, None)
                slot[key] = row
                self._owner[row] = key
                self._reset_row(row)
            rows[i] = row
        return rows

    def _reset_row(self, row: int) -> None:
        self._A[row] = 0.0
        self._b[row] = 0.0
        self._n[row] = 0
        self._span[row] = 0.0

    def observe(self, ids: np.ndarray, origin: np.ndarray,
                directions: np.ndarray) -> np.ndarray:
        """Ajoute une visee par identifiant. Renvoie les lignes touchees.

        `origin` est le centre optique dans le repere monde, commun a tous les
        points de l'image; `directions` sont les directions unitaires des
        rayons, une par point.
        """
        if len(ids) == 0:
            return np.zeros(0, dtype=np.int64)
        rows = self._rows_for(ids)

        d = np.asarray(directions, dtype=np.float64)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        np.divide(d, np.maximum(norm, 1e-12), out=d)

        # M = I - d d^T: projecteur sur le plan perpendiculaire au rayon. Le
        # residu M (X - C) est exactement la distance du point au rayon, ce qui
        # fait de la somme des M le systeme normal du probleme.
        M = -d[:, :, None] * d[:, None, :]
        M[:, 0, 0] += 1.0
        M[:, 1, 1] += 1.0
        M[:, 2, 2] += 1.0

        C = np.asarray(origin, dtype=np.float64).reshape(3)
        self._last_C = C.copy()
        first = self._n[rows] == 0
        if first.any():
            self._first_C[rows[first]] = C

        self._A[rows] = self._A[rows] * self._decay + M
        self._b[rows] = self._b[rows] * self._decay + M @ C
        self._n[rows] += 1
        # Base effective: distance a la premiere visee. C'est elle qui
        # conditionne la profondeur, pas le nombre d'images.
        self._span[rows] = np.linalg.norm(C - self._first_C[rows], axis=1)
        return rows

    # -- resolution --------------------------------------------------------

    def solve(self, rows: np.ndarray, min_views: int = 3,
              max_sigma_m: float = 1.0,
              focal_px: float = 300.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resout les points demandes.

        Renvoie (points monde, sigma metres, masque des lignes resolues). Le
        masque porte sur `rows`: une ligne mal conditionnee ou trop incertaine
        est simplement absente du resultat, elle n'est pas remplacee par une
        valeur de repli.
        """
        empty = (np.zeros((0, 3)), np.zeros(0), np.zeros(len(rows), dtype=bool))
        if len(rows) == 0:
            return empty

        ready = self._n[rows] >= min_views
        if not ready.any():
            return empty

        sel = rows[ready]
        A = self._A[sel]
        b = self._b[sel]

        # Les valeurs propres disent tout: leur plus petite mesure l'ouverture
        # angulaire des visees, donc la qualite du conditionnement, donc
        # l'incertitude le long de la direction la moins contrainte.
        eig = np.linalg.eigvalsh(A)
        lam_min = eig[:, 0]
        usable = lam_min > 1e-9
        if not usable.any():
            return empty

        idx = np.flatnonzero(usable)
        X = np.zeros((len(sel), 3))
        # Second membre en colonne: sur une pile de matrices, numpy traite un
        # tableau (M, 3) comme une matrice et non comme M vecteurs.
        X[idx] = np.linalg.solve(A[idx], b[idx][:, :, None])[:, :, 0]

        # Distance depuis la position *courante*, pas depuis la premiere visee:
        # c'est la distance a laquelle le point est observe maintenant qui fixe
        # l'erreur commise maintenant.
        depth = np.linalg.norm(X - self._last_C, axis=1)
        # Bruit angulaire d'une visee: un pixel d'erreur vu a la focale.
        # L'ecart perpendiculaire correspondant croit avec la distance, ce qui
        # est la raison de fond pour laquelle un point lointain est mal situe.
        sigma_perp = depth * self.sigma_px / max(1e-6, focal_px)
        # Trace de l'inverse de A: somme des inverses des valeurs propres.
        # C'est la variance totale de la position, a un facteur pres, et elle
        # est dominee par la direction la moins contrainte -- celle qui pose
        # probleme quand les visees sont presque paralleles.
        inv = np.where(eig > 1e-12, 1.0 / np.maximum(eig, 1e-12), np.inf)
        sigma = sigma_perp * np.sqrt(np.maximum(0.0, inv.sum(axis=1)))

        good = usable & np.isfinite(sigma) & (sigma < max_sigma_m)
        good &= np.isfinite(X).all(axis=1)

        mask = np.zeros(len(rows), dtype=bool)
        mask[np.flatnonzero(ready)[good]] = True
        return X[good], sigma[good], mask

    def views(self, rows: np.ndarray) -> np.ndarray:
        return self._n[rows]

    def baseline(self, rows: np.ndarray) -> np.ndarray:
        return self._span[rows]

    def shift(self, delta: np.ndarray) -> None:
        """Deplace toute la structure accumulee (correction de derive).

        Les visees sont enregistrees dans le repere monde. Si ce repere est
        recale -- fermeture de boucle -- les accumulations doivent suivre,
        sinon les visees d'avant et d'apres le recalage ne se couperaient plus
        au meme endroit.
        """
        d = np.asarray(delta, dtype=np.float64).reshape(3)
        if not np.any(d):
            return
        live = np.flatnonzero(self._n > 0)
        if len(live) == 0:
            return
        self._first_C[live] += d
        self._b[live] += self._A[live] @ d
