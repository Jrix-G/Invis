"""Carte d'elevation: la scene comme surface, et non comme poussiere de points.

Pourquoi une grille plutot que davantage de points
--------------------------------------------------
Le nuage de points repond a la question "ou y a-t-il quelque chose". Il repond
mal a "a quoi ressemble le terrain": un operateur ne reconnait pas un lieu dans
un nuage, il le reconnait dans une image. Or le sol *est* une surface, et une
surface se decrit par une hauteur et une couleur en chaque point du plan --
pas par une liste de points flottants.

Cette representation change trois choses d'un coup:

  - la memoire cesse de croitre. Repasser cent fois au meme endroit met a jour
    les memes cases au lieu d'empiler cent mille points;
  - les mesures repetees se moyennent naturellement, la ou un nuage se contente
    de les accumuler avec leur bruit;
  - la surface peut porter la couleur relevee dans l'image, ce qui donne une
    scene reconnaissable au lieu d'un semis colore.

Densification: le sol n'a pas besoin d'etre triangule
-----------------------------------------------------
Un point du sol ne demande aucune parallaxe, aucune seconde vue, aucun suivi:
son rayon perce un plan connu, et l'intersection est exacte. On peut donc
echantillonner l'image sur une grille reguliere et projeter tous ces pixels
d'un coup, au lieu de se limiter aux quelques centaines de points que le
suiveur veut bien fournir. Le calcul est le meme que pour un point suivi,
applique a plusieurs milliers de pixels en une operation vectorisee.

C'est la difference entre reconstruire ce qu'on suit et reconstruire ce qu'on
voit.

Cout et bornes
--------------
La grille fait un nombre fixe de cases et se recentre par blocs entiers quand
le drone s'approche du bord. Chaque image coute deux comptages vectorises sur
les cases touchees. Rien ne croit avec la duree du vol.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config


class ElevationGrid:
    """Hauteur, couleur et fiabilite du terrain, par case du plan horizontal."""

    def __init__(self, cells: int = config.GRID_CELLS,
                 resolution_m: float = config.GRID_RES_M) -> None:
        self.cells = int(cells)
        self.res = float(resolution_m)
        n = self.cells * self.cells
        self.height = np.zeros(n, dtype=np.float32)
        self.colour = np.zeros((n, 3), dtype=np.float32)
        self.count = np.zeros(n, dtype=np.uint16)
        self.stamp = np.zeros(n, dtype=np.float32)
        # Coin bas-gauche de la grille, en metres, repere monde.
        self.origin = np.zeros(2, dtype=np.float64)
        self._centred = False
        # Ombrage: garde en cache, et recalcule sur la seule emprise remplie.
        # Balayer les deux cent mille cases quand quelques milliers sont
        # renseignees coutait dix millisecondes par rendu.
        self._shade = np.full(n, config.GRID_AMBIENT, dtype=np.float32)
        self._shade_stale = True
        self._bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        self.height[:] = 0.0
        self.colour[:] = 0.0
        self.count[:] = 0
        self.stamp[:] = 0.0
        self.origin[:] = 0.0
        self._centred = False
        self._shade[:] = config.GRID_AMBIENT
        self._shade_stale = True
        self._bbox = None

    @property
    def span_m(self) -> float:
        return self.cells * self.res

    def __len__(self) -> int:
        return int(np.count_nonzero(self.count))

    # -- ancrage -----------------------------------------------------------

    def recentre(self, position_xy: np.ndarray) -> bool:
        """Recentre la grille sur le drone, par blocs entiers de cases.

        Le decalage se fait en nombre entier de cases, jamais en fraction: une
        case doit continuer de designer exactement le meme carre de terrain,
        sinon le contenu se brouillerait un peu plus a chaque recentrage.

        Le declenchement est hysteretique -- on ne recentre qu'a un quart de la
        grille du bord -- pour qu'un vol stationnaire pres d'une limite ne
        provoque pas un recentrage par image.
        """
        xy = np.asarray(position_xy, dtype=np.float64).reshape(2)
        half = self.span_m / 2.0
        target = xy - half

        if not self._centred:
            self.origin = np.round(target / self.res) * self.res
            self._centred = True
            return True

        offset = xy - (self.origin + half)
        if np.max(np.abs(offset)) < self.span_m * config.GRID_RECENTRE_FRACTION:
            return False

        shift_cells = np.round(offset / self.res).astype(int)
        if not np.any(shift_cells):
            return False

        # np.roll deplace le contenu; les colonnes qui entrent par un bord
        # decrivent du terrain jamais vu et doivent donc etre vidées, sans quoi
        # le terrain quitte par un bord et reapparait par l'autre.
        grid = self._as_2d()
        for array in grid:
            array[:] = np.roll(array, (-shift_cells[1], -shift_cells[0]), axis=(0, 1))
        self._clear_incoming(shift_cells)
        self.origin += shift_cells * self.res
        # L'emprise a ombrer suit le contenu; elle est simplement recalculee au
        # prochain rendu si le decalage l'a poussee hors grille.
        if self._bbox is not None:
            bx0, by0, bx1, by1 = self._bbox
            dx, dy = int(shift_cells[0]), int(shift_cells[1])
            bx0, bx1 = max(0, bx0 - dx), min(self.cells - 1, bx1 - dx)
            by0, by1 = max(0, by0 - dy), min(self.cells - 1, by1 - dy)
            self._bbox = (bx0, by0, bx1, by1) if bx0 <= bx1 and by0 <= by1 else None
        self._shade_stale = True
        return True

    def _as_2d(self):
        n = self.cells
        return (self.height.reshape(n, n), self.count.reshape(n, n),
                self.stamp.reshape(n, n), self._shade.reshape(n, n),
                self.colour.reshape(n, n, 3))

    def _clear_incoming(self, shift_cells: np.ndarray) -> None:
        n = self.cells
        height, count, stamp, shade, colour = self._as_2d()

        def wipe(rows, cols):
            height[rows, cols] = 0.0
            count[rows, cols] = 0
            stamp[rows, cols] = 0.0
            shade[rows, cols] = config.GRID_AMBIENT
            colour[rows, cols] = 0.0

        dx, dy = int(shift_cells[0]), int(shift_cells[1])
        if dx > 0:
            wipe(slice(None), slice(max(0, n - dx), n))
        elif dx < 0:
            wipe(slice(None), slice(0, min(n, -dx)))
        if dy > 0:
            wipe(slice(max(0, n - dy), n), slice(None))
        elif dy < 0:
            wipe(slice(0, min(n, -dy)), slice(None))

    # -- alimentation ------------------------------------------------------

    def add(self, world_xyz: np.ndarray, stamp: float,
            bgr: Optional[np.ndarray] = None) -> int:
        """Verse un lot de points mesures. Renvoie le nombre de cases touchees.

        Les points tombant plusieurs fois dans la meme case sont d'abord
        moyennes entre eux, puis fondus dans la valeur deja connue. Faire les
        deux en une passe donnerait un poids arbitraire aux cases les mieux
        echantillonnees de l'image courante -- typiquement celles du bas, les
        plus proches -- au detriment de tout l'historique.
        """
        if len(world_xyz) == 0:
            return 0
        pts = np.asarray(world_xyz, dtype=np.float64)
        rel = (pts[:, :2] - self.origin) / self.res
        ix = np.floor(rel[:, 0]).astype(np.int64)
        iy = np.floor(rel[:, 1]).astype(np.int64)
        inside = (ix >= 0) & (ix < self.cells) & (iy >= 0) & (iy < self.cells)
        inside &= np.isfinite(pts).all(axis=1)
        if not inside.any():
            return 0

        flat = iy[inside] * self.cells + ix[inside]
        z = pts[inside, 2]

        # Regroupement sur les seules cases touchees, jamais sur la grille
        # entiere. Un `bincount` de la taille de la grille reallouerait deux
        # cent mille cases quatre fois par image pour n'en remplir que
        # quelques centaines: mesure a cinq millisecondes par image, contre
        # une fraction de milliseconde ici.
        touched, inverse, hits = np.unique(flat, return_inverse=True, return_counts=True)
        frame_h = np.bincount(inverse, weights=z, minlength=len(touched)) / hits

        seen = self.count[touched]
        # Poids de la mesure nouvelle: 1/n tant que la case est peu vue, puis
        # un plancher. Le plancher compte: sans lui une case vue mille fois
        # cesserait d'ecouter, et un terrain qui change -- ou une pose qui
        # vient d'etre recalee -- ne serait jamais corrige.
        weight = np.maximum(config.GRID_MIN_WEIGHT,
                            1.0 / np.maximum(1, seen.astype(np.float64)))
        weight = np.where(seen == 0, 1.0, weight).astype(np.float32)

        self.height[touched] += weight * (frame_h.astype(np.float32) - self.height[touched])
        self.stamp[touched] = np.float32(stamp)
        total = seen.astype(np.int32) + hits.astype(np.int32)
        self.count[touched] = np.minimum(total, 65535).astype(np.uint16)

        if bgr is not None and len(bgr) == len(pts):
            cols = np.asarray(bgr, dtype=np.float64)[inside]
            frame_c = np.stack([
                np.bincount(inverse, weights=cols[:, c], minlength=len(touched))
                for c in range(3)
            ], axis=1) / hits[:, None]
            self.colour[touched] += weight[:, None] * (frame_c.astype(np.float32)
                                                       - self.colour[touched])

        self._mark(touched)
        return int(len(touched))

    def _mark(self, touched: np.ndarray) -> None:
        """Etend l'emprise a recalculer et invalide l'ombrage."""
        self._shade_stale = True
        cx = touched % self.cells
        cy = touched // self.cells
        x0, x1 = int(cx.min()), int(cx.max())
        y0, y1 = int(cy.min()), int(cy.max())
        if self._bbox is None:
            self._bbox = (x0, y0, x1, y1)
        else:
            bx0, by0, bx1, by1 = self._bbox
            self._bbox = (min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1))

    # -- lecture -----------------------------------------------------------

    def surface(self, min_count: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cases renseignees: centres 3D monde, couleurs, eclairement.

        L'eclairement vient de la pente locale de la surface. Sans lui, une
        surface coloree reste illisible en relief: l'oeil lit la forme dans les
        ombres, pas dans la teinte. Il se calcule par deux derivees de l'image
        de hauteur, soit deux convolutions 3x3.
        """
        valid = self.count >= min_count
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, np.zeros(0, dtype=np.float32)

        n = self.cells
        ix = (idx % n).astype(np.float32)
        iy = (idx // n).astype(np.float32)
        centres = np.empty((len(idx), 3), dtype=np.float32)
        centres[:, 0] = self.origin[0] + (ix + 0.5) * self.res
        centres[:, 1] = self.origin[1] + (iy + 0.5) * self.res
        centres[:, 2] = self.height[idx]

        shade = self.shading()[idx]
        return centres, self.colour[idx], shade

    def shading(self) -> np.ndarray:
        """Eclairement lambertien d'une lumiere fixe, oblique.

        Le calcul est restreint a l'emprise reellement renseignee et garde en
        cache. La grille couvre cinquante metres de cote alors qu'un vol en
        occupe quelques-uns: balayer tout coutait dix millisecondes par rendu
        pour un resultat identique sur les cases vides.
        """
        if not self._shade_stale or self._bbox is None:
            return self._shade
        x0, y0, x1, y1 = self._bbox
        n = self.cells
        # Une case de marge: le noyau de Sobel deborde d'un pixel, et sans
        # cette marge la pente serait fausse sur tout le pourtour de l'emprise.
        x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
        x1, y1 = min(n, x1 + 2), min(n, y1 + 2)

        h = self.height.reshape(n, n)[y0:y1, x0:x1]
        # Sobel plutot qu'une difference simple: il moyenne sur trois lignes,
        # ce qui evite qu'une case bruitee allume une raie entiere.
        gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * self.res)
        gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * self.res)
        # Normale non normalisee (-gx, -gy, 1), lumiere config.GRID_LIGHT.
        lx, ly, lz = config.GRID_LIGHT
        dot = (-gx * lx) + (-gy * ly) + lz
        norm = np.sqrt(gx * gx + gy * gy + 1.0)
        shade = dot / (norm * float(np.linalg.norm(config.GRID_LIGHT)))
        self._shade.reshape(n, n)[y0:y1, x0:x1] = np.clip(
            config.GRID_AMBIENT + (1.0 - config.GRID_AMBIENT) * shade, 0.0, 1.0)
        self._shade_stale = False
        return self._shade

    def height_at(self, world_xy: np.ndarray) -> np.ndarray:
        """Hauteur connue aux positions demandees, NaN si jamais observee."""
        xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
        rel = (xy - self.origin) / self.res
        ix = np.floor(rel[:, 0]).astype(np.int64)
        iy = np.floor(rel[:, 1]).astype(np.int64)
        inside = (ix >= 0) & (ix < self.cells) & (iy >= 0) & (iy < self.cells)
        out = np.full(len(xy), np.nan)
        if not inside.any():
            return out
        flat = iy[inside] * self.cells + ix[inside]
        known = self.count[flat] > 0
        sel = np.flatnonzero(inside)[known]
        out[sel] = self.height[flat[known]]
        return out
