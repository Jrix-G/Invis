"""Fermeture de boucle: reconnaitre un endroit deja survole.

Le probleme
-----------
Le cap n'est observe par rien, la position non plus. Elles s'integrent, donc
elles derivent, et rien dans les images prises *en avancant* ne peut les
recaler: chaque image ne dit que le deplacement depuis la precedente. Au bout
d'une minute de vol la position estimee peut avoir glisse de plusieurs metres,
et le nuage de points glisse avec elle -- deux passages au meme endroit
laissent alors deux copies decalees du meme mur.

Ce que ce module ajoute est la seule information capable d'arreter cela: la
reconnaissance d'un lieu deja vu. Elle ne vient d'aucun capteur nouveau, elle
vient de la memoire du vol.

La representation choisie
-------------------------
Une image brute ne se compare pas d'un passage a l'autre: le meme sol vu sous
un autre cap ou a une autre altitude donne une image differente. On redresse
donc chaque image en une *vignette metrique du sol*, vue de dessus, orientee
selon le nord de la reconstruction et centree sur le drone. Cette vignette a
une propriete decisive: elle ne depend plus ni du cap, ni de l'altitude, ni
de l'inclinaison de la camera. Le meme endroit donne la meme vignette, et
deux passages ne different plus que par une translation.

C'est pour cela que la comparaison est possible avec quelques centaines
d'operations au lieu d'un detecteur de points d'interet et d'un sac de mots.

Le cout
-------
Une redressement vers 64x64, un descripteur de 256 nombres, et un produit
matrice-vecteur contre les cles memorisees -- quelques dizaines de
microsecondes. Seule la meilleure candidate est ensuite verifiee finement,
une fois, par alignement direct.

Ce que le module ne fait pas
----------------------------
Il ne reoptimise pas la trajectoire passee. Une fermeture de boucle est
publiee comme une *mesure de position*, remise au filtre d'etat, qui la
pondere par l'incertitude accumulee et peut la refuser. Le nuage deja
construit n'est pas deplace: les points anciens gardent l'erreur qu'ils
avaient, les nouveaux sont poses au bon endroit. Retordre tout l'historique
demanderait une optimisation de graphe de poses, hors du budget de calcul
qu'on s'est fixe -- et la faire a moitie donnerait une carte pire, pas
meilleure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config, geometry
from .geometry import Intrinsics


@dataclass
class LoopMatch:
    """Un lieu reconnu, et la position que cela implique."""

    position_xy: np.ndarray      # position corrigee du drone, repere monde
    shift_m: float               # ampleur de la correction proposee
    yaw_error_deg: float         # ecart de cap constate
    similarity: float            # ressemblance de la vignette, entre -1 et 1
    age_s: float                 # anciennete du lieu reconnu
    index: int                   # rang de la cle reconnue


def ground_patch(gray: np.ndarray, K: Intrinsics, height_m: float,
                 tilt_deg: float, roll_deg: float, yaw_rad: float,
                 size_px: int = 64, span_m: float = 8.0) -> Optional[np.ndarray]:
    """Vue de dessus metrique du sol, orientee monde, centree sur le drone.

    Le redressement est une simple homographie: le sol est un plan, la camera
    le regarde, et la hauteur de vol donne l'echelle. Il n'y a donc rien a
    estimer ici -- la transformation se calcule, elle ne se cherche pas.

    Renvoie None si la geometrie ne permet pas de construire la vue.
    """
    if height_m <= 0.0 or size_px < 8 or span_m <= 0.0:
        return None

    res = span_m / float(size_px)          # metres par pixel de vignette
    centre = (size_px - 1) / 2.0

    # Vignette -> deplacement metrique dans le repere monde.
    A = np.array([[res, 0.0, -centre * res],
                  [0.0, res, -centre * res],
                  [0.0, 0.0, 1.0]])

    # Deplacement monde -> repere drone, sol a -h. C'est ici que le cap
    # disparait: la vignette est construite dans l'orientation du monde, donc
    # deux passages caps differents donnent la meme vignette.
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    M = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, -height_m]])

    R_bc = geometry.body_from_camera(tilt_deg, roll_deg)
    H = K.matrix @ R_bc.T @ M @ A

    patch = cv2.warpPerspective(gray, H, (size_px, size_px),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Devant la camera seulement. La troisieme ligne de l'homographie donne la
    # profondeur: negative, le point est derriere l'objectif et l'homographie
    # le ramene quand meme dans l'image, en miroir. Sans ce masque la vignette
    # contiendrait une copie retournee du sol.
    u, v = np.meshgrid(np.arange(size_px, dtype=np.float64),
                       np.arange(size_px, dtype=np.float64))
    depth = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    patch[depth <= 1e-6] = 0
    return patch


def descriptor(patch: np.ndarray, size_px: int = 16) -> Optional[np.ndarray]:
    """Signature compacte d'une vignette, centree et normee.

    Centrer supprime la luminosite moyenne, normer supprime le contraste: ce
    qui reste est la structure du sol, seule chose qu'on puisse esperer
    retrouver d'un passage a l'autre sous un autre eclairage et une autre
    exposition automatique.
    """
    valid = patch > 0
    if valid.mean() < config.LOOP_MIN_COVERAGE:
        return None
    small = cv2.resize(patch, (size_px, size_px), interpolation=cv2.INTER_AREA)
    d = small.astype(np.float32).ravel()
    d -= d.mean()
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return None
    return d / norm


class LoopCloser:
    """Memoire des lieux traverses, et reconnaissance parmi eux."""

    def __init__(self, capacity: int = config.LOOP_CAPACITY,
                 descriptor_px: int = config.LOOP_DESCRIPTOR_PX) -> None:
        self.capacity = int(capacity)
        self.descriptor_px = int(descriptor_px)
        dim = self.descriptor_px * self.descriptor_px
        self._desc = np.zeros((self.capacity, dim), dtype=np.float32)
        self._patch: list = [None] * self.capacity
        self._pose = np.zeros((self.capacity, 2), dtype=np.float64)
        self._yaw = np.zeros(self.capacity, dtype=np.float64)
        self._stamp = np.zeros(self.capacity, dtype=np.float64)
        self._count = 0
        self._write = 0
        self._last_key_xy: Optional[np.ndarray] = None
        self.matches = 0

    def reset(self) -> None:
        self._count = 0
        self._write = 0
        self._last_key_xy = None
        self.matches = 0

    def __len__(self) -> int:
        return self._count

    # -- memorisation ------------------------------------------------------

    def remember(self, patch: np.ndarray, desc: np.ndarray,
                 position_xy: np.ndarray, yaw_rad: float, stamp: float) -> bool:
        """Memorise ce lieu s'il est assez loin du dernier memorise.

        Espacer les cles est ce qui borne le cout: la memoire couvre une
        distance, pas une duree. Un vol stationnaire de dix minutes n'ajoute
        donc aucune cle, et une longue exploration en ajoute autant qu'il faut.
        """
        xy = np.asarray(position_xy, dtype=np.float64).reshape(2)
        if self._last_key_xy is not None:
            if float(np.linalg.norm(xy - self._last_key_xy)) < config.LOOP_KEY_SPACING_M:
                return False

        row = self._write
        self._desc[row] = desc
        self._patch[row] = patch.copy()
        self._pose[row] = xy
        self._yaw[row] = float(yaw_rad)
        self._stamp[row] = float(stamp)
        self._write = (self._write + 1) % self.capacity
        self._count = min(self.capacity, self._count + 1)
        self._last_key_xy = xy.copy()
        return True

    # -- reconnaissance ----------------------------------------------------

    def query(self, patch: np.ndarray, desc: np.ndarray, position_xy: np.ndarray,
              yaw_rad: float, stamp: float,
              position_sigma_m: float = 0.0) -> Optional[LoopMatch]:
        """Cherche un lieu deja vu compatible avec la position courante.

        Le rayon de recherche s'ouvre avec l'incertitude accumulee, et c'est
        necessaire: c'est precisement quand l'odometrie a beaucoup derive que
        le lieu reconnu se trouve loin de la position estimee. Un rayon fixe
        interdirait la fermeture juste quand elle sert le plus.
        """
        if self._count == 0:
            return None
        xy = np.asarray(position_xy, dtype=np.float64).reshape(2)

        pose = self._pose[:self._count]
        age = stamp - self._stamp[:self._count]
        radius = config.LOOP_SEARCH_RADIUS_M + 3.0 * max(0.0, position_sigma_m)
        near = np.linalg.norm(pose - xy, axis=1) <= radius
        # Une cle trop recente decrit le meme endroit vu il y a un instant:
        # elle confirmerait l'odometrie par elle-meme au lieu de la corriger.
        candidates = np.flatnonzero(near & (age >= config.LOOP_MIN_AGE_S))
        if len(candidates) == 0:
            return None

        # Ressemblance: un produit scalaire, les descripteurs etant centres et
        # normes. Tout l'ensemble candidat est traite en un produit matriciel.
        scores = self._desc[candidates] @ desc
        best = int(np.argmax(scores))
        similarity = float(scores[best])
        if similarity < config.LOOP_MIN_SIMILARITY:
            return None

        row = int(candidates[best])
        aligned = self._align(self._patch[row], patch)
        if aligned is None:
            return None
        delta_xy, yaw_error = aligned

        corrected = self._pose[row] + delta_xy
        shift = float(np.linalg.norm(corrected - xy))
        # Une correction absurde vient d'une fausse reconnaissance, pas d'une
        # grosse derive: au-dela de ce que le rayon de recherche autorisait,
        # elle ne peut pas etre juste.
        if shift > radius:
            return None

        self.matches += 1
        return LoopMatch(position_xy=corrected, shift_m=shift,
                         yaw_error_deg=float(math.degrees(yaw_error)),
                         similarity=similarity, age_s=float(age[best]), index=row)

    def _align(self, reference: Optional[np.ndarray],
               current: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Alignement fin des deux vignettes.

        La ressemblance des descripteurs dit "c'est probablement ici"; elle ne
        dit pas "de combien je me suis trompe". L'alignement direct des deux
        vignettes le dit, au pixel de vignette pres -- soit une douzaine de
        centimetres avec les reglages par defaut.

        La mesure est une correlation de phase, et rien d'autre. Ce choix a ete
        tranche par la mesure, contre l'intuition de depart.

        L'outil evident etait un alignement iteratif (`findTransformECC`), qui
        rend translation *et* rotation. Il s'est revele deux fois mauvais sur
        ce sol texture: partant de l'identite il ne convergeait qu'une fois
        sur deux, et quand il convergeait il inventait jusqu'a cinq degres de
        rotation sur des decalages de translation pure, en degradant au passage
        la translation de vingt pour cent. C'est le defaut d'une descente de
        gradient sur une surface d'erreur pleine de minima locaux -- ce qu'est
        exactement une texture aleatoire.

        La correlation de phase n'a pas ce probleme: elle cherche un pic dans
        le domaine de Fourier, donc globalement, sans point de depart. Mesuree
        sur des decalages connus allant jusqu'a trois metres, son erreur reste
        sous trois centimetres, et la nettete du pic donne en prime un
        indicateur de confiance.

        Le prix a payer est qu'elle ne mesure pas la rotation. Ce n'est pas
        genant ici: la vignette est *deja* orientee selon le monde a partir du
        cap estime. Une erreur de cap residuelle ne se traduit donc pas par une
        rotation a mesurer, mais par une degradation du pic -- que le seuil de
        reponse rejette.

        Convention de signe, verifiee sur des decalages connus: la translation
        rendue vaut (position courante - position memorisee).
        """
        if reference is None:
            return None
        ref = reference.astype(np.float64)
        cur = current.astype(np.float64)

        # Fenetre de Hann: sans elle, les bords francs de la vignette dominent
        # le spectre et la correlation repond sur le cadre, pas sur le sol.
        window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_64F)
        (dx, dy), response = cv2.phaseCorrelate(ref, cur, window)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return None
        if response < config.LOOP_MIN_PHASE_RESPONSE:
            return None

        res = config.LOOP_PATCH_SPAN_M / float(config.LOOP_PATCH_PX)
        return np.array([-dx, -dy], dtype=np.float64) * res, 0.0
