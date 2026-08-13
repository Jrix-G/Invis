"""Geometrie camera: rayons, intersection avec le sol, incertitude sur la hauteur.

Conventions
-----------
Repere camera (OpenCV): x a droite, y vers le bas de l'image, z vers l'avant
optique.
Repere drone: X vers l'avant, Y vers la gauche, Z vers le haut. Origine a la
camera.

La camera est piquee de `tilt` degres sous l'horizontale (45 sur ce drone).

Pourquoi une distance metrique est possible
-------------------------------------------
Une image seule ne donne aucune echelle: c'est une propriete des images, pas
une limite de calcul. Mais ici l'echelle vient d'ailleurs -- la camera est a
une hauteur h au-dessus d'un plan. Un rayon qui perce ce plan donne une
distance unique. Le calcul est exact; toute l'erreur vient de h et de
l'assiette.

Effet d'une hauteur fausse
--------------------------
D est *proportionnel* a h. Une erreur de 20 % sur h donne 20 % sur toutes les
distances, sans deformer la scene: les rapports entre distances, l'ordre
"lequel est le plus proche", et le temps avant collision restent exacts. C'est
pourquoi ce module renvoie toujours un intervalle, et pourquoi le mode
"unites de h" (h = 1) reste juste meme sans altimetre.

L'assiette, elle, n'est pas une simple echelle: 1 degre d'erreur de tangage
deplace le point d'impact de facon non lineaire, surtout en haut d'image. Elle
est donc estimee sur l'image (voir `attitude_from_plane_normal`) plutot que
supposee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config


def orient_frame(bgr: np.ndarray, flip_h: bool = config.CAMERA_FLIP_H,
                 flip_v: bool = config.CAMERA_FLIP_V) -> np.ndarray:
    """Remet l'image dans le sens reel de la scene.

    A appliquer avant *tout* le reste, et pas seulement pour le confort de
    lecture. Deux raisons de fond:

    1. La ligne d'image porte la distance. Le modele suppose le haut de
       l'image loin et le bas pres. Une image retournee verticalement inverse
       cette relation: les distances annoncees seraient fausses, pas
       seulement l'affichage.

    2. Un miroir simple inverse le sens direct du repere (determinant -1).
       La decomposition d'homographie en rotation et translation suppose un
       repere direct: avec un seul miroir non corrige, le sens du lacet
       s'inverse et la reconstruction part de travers. Deux miroirs (rotation
       de 180 degres) preservent le sens direct, un seul non.

    Redresser l'image restitue la geometrie du sten pinhole d'origine, et tout
    le calcul en aval redevient valable sans changement.
    """
    if flip_h and flip_v:
        return cv2.flip(bgr, -1)      # rotation de 180 degres
    if flip_h:
        return cv2.flip(bgr, 1)       # miroir gauche/droite
    if flip_v:
        return cv2.flip(bgr, 0)       # miroir haut/bas
    return bgr


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_fov(width: int, height: int,
                 hfov_deg: float = config.HFOV_DEG,
                 vfov_deg: float = config.VFOV_DEG) -> "Intrinsics":
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
        return Intrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                          width=width, height=height)

    def scaled(self, factor: float) -> "Intrinsics":
        """Meme optique vue a une autre resolution de travail."""
        return Intrinsics(fx=self.fx * factor, fy=self.fy * factor,
                          cx=self.cx * factor, cy=self.cy * factor,
                          width=int(round(self.width * factor)),
                          height=int(round(self.height * factor)))

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def body_from_camera(tilt_deg: float = config.CAMERA_TILT_DEG,
                     roll_deg: float = 0.0,
                     pitch_offset_deg: float = 0.0) -> np.ndarray:
    """Rotation repere camera -> repere drone.

    `tilt_deg` est negatif quand la camera pique vers le sol. `pitch_offset_deg`
    ajoute l'assiette instantanee du drone, `roll_deg` son roulis.
    """
    theta = math.radians(-tilt_deg + pitch_offset_deg)
    ct, st = math.cos(theta), math.sin(theta)

    # Colonnes: axes camera exprimes dans le repere drone.
    x_c = np.array([0.0, -1.0, 0.0])          # droite image = droite drone (-Y)
    z_c = np.array([ct, 0.0, -st])            # axe optique, pique de theta
    y_c = np.cross(z_c, x_c)                  # bas image
    R = np.column_stack([x_c, y_c, z_c])

    if roll_deg:
        phi = math.radians(roll_deg)
        cr, sr = math.cos(phi), math.sin(phi)
        # Roulis autour de l'axe avant du drone.
        R_roll = np.array([[1.0, 0.0, 0.0],
                           [0.0, cr, -sr],
                           [0.0, sr, cr]])
        R = R_roll @ R
    return R


def rays_body(uv: np.ndarray, K: Intrinsics, R_bc: np.ndarray) -> np.ndarray:
    """Directions (non normalisees) des rayons pixel, dans le repere drone."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    d_cam = np.empty((len(uv), 3), dtype=np.float64)
    d_cam[:, 0] = (uv[:, 0] - K.cx) / K.fx
    d_cam[:, 1] = (uv[:, 1] - K.cy) / K.fy
    d_cam[:, 2] = 1.0
    return d_cam @ R_bc.T


def ground_points(uv: np.ndarray, K: Intrinsics, height_m: float,
                  tilt_deg: float = config.CAMERA_TILT_DEG,
                  roll_deg: float = 0.0,
                  pitch_offset_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Intersection des rayons avec le sol.

    Renvoie (points, valides): points de forme (N, 3) dans le repere drone
    (X avant, Y gauche, Z bas = -height). Un rayon qui monte ou qui frole
    l'horizontale ne coupe pas le sol devant: il est marque invalide plutot
    que de produire une distance absurde.
    """
    R_bc = body_from_camera(tilt_deg, roll_deg, pitch_offset_deg)
    d = rays_body(uv, K, R_bc)

    dz = d[:, 2]
    # Seuil: sous cette pente le point d'impact part vers l'infini et la
    # moindre erreur d'assiette le fait exploser. Mieux vaut ne rien dire.
    valid = dz < -config.MIN_RAY_SLOPE

    t = np.zeros(len(d))
    np.divide(-height_m, dz, out=t, where=valid)
    pts = d * t[:, None]
    pts[~valid] = np.nan
    return pts, valid


def ground_range(uv: np.ndarray, K: Intrinsics, height_m: float,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Distance horizontale au sol pour chaque pixel, et validite."""
    pts, valid = ground_points(uv, K, height_m, **kwargs)
    rng = np.hypot(pts[:, 0], pts[:, 1])
    rng[~valid] = np.nan
    return rng, valid


def range_band(range_m: float, height_m: float, sigma_h_m: float) -> Tuple[float, float]:
    """Intervalle de distance induit par l'incertitude sur la hauteur.

    D est proportionnel a h, donc l'incertitude est purement multiplicative:
    aucune deformation, juste une echelle mal connue. Afficher l'intervalle
    evite d'annoncer des centimetres qu'on n'a pas.
    """
    if height_m <= 0:
        return (float("nan"), float("nan"))
    ratio = max(0.0, min(0.95, sigma_h_m / height_m))
    return (range_m * (1.0 - ratio), range_m * (1.0 + ratio))


def attitude_from_plane_normal(normal_cam: np.ndarray) -> Tuple[float, float]:
    """Tangage et roulis deduits de la normale du plan sol mesuree sur l'image.

    L'homographie du sol donne la direction de sa normale dans le repere
    camera. Cette direction est observable sans aucune echelle -- c'est la
    partie de la geometrie que l'image *peut* mesurer. On s'en sert pour
    corriger l'inclinaison reelle au lieu de faire confiance aux 45 degres
    nominaux, parce que 1 degre d'erreur coute ~10 % de distance en haut
    d'image.

    Renvoie (tilt_deg, roll_deg), tilt negatif = camera piquee vers le sol.
    """
    n = np.asarray(normal_cam, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return (config.CAMERA_TILT_DEG, 0.0)
    n = n / norm
    # La normale du sol pointe vers le haut: dans le repere camera son
    # composante y (vers le bas de l'image) doit etre negative.
    if n[1] > 0:
        n = -n

    # Avec la convention de body_from_camera, la normale s'ecrit
    #   n = (-sin(roll), -cos(tilt)cos(roll), -sin(tilt)cos(roll)).
    # Le rapport n[2]/n[1] elimine cos(roll) et donne le tangage; le roulis se
    # lit sur n[0] compare a la norme des deux autres, sans quoi il ressort
    # divise par cos(tilt).
    tilt = -math.degrees(math.atan2(-n[2], -n[1]))
    roll = math.degrees(math.atan2(-n[0], math.hypot(n[1], n[2])))
    return (tilt, roll)


def expected_ground_normal(tilt_deg: float = config.CAMERA_TILT_DEG,
                           roll_deg: float = 0.0) -> np.ndarray:
    """Normale du sol attendue, exprimee dans le repere camera."""
    return body_from_camera(tilt_deg, roll_deg).T @ np.array([0.0, 0.0, 1.0])


def decompose_plane(H: np.ndarray, K_matrix: np.ndarray,
                    expected_normal: np.ndarray):
    """Decompose une homographie de plan et retient la solution la plus plausible.

    Renvoie (R, t, n, score) ou None. `score` est le produit scalaire entre la
    normale trouvee et celle attendue pour le sol: il vaut 1 quand le plan
    trouve est bien le sol, et chute des qu'il s'agit d'autre chose.

    Convention de translation: pour une homographie qui envoie l'image
    precedente sur l'image courante, OpenCV rend une rotation conforme a
    X2 = R X1 mais une translation de signe oppose. Verifie contre la pose
    exacte du simulateur, avec et sans lacet. La correction est appliquee ici,
    une seule fois.
    """
    try:
        retval, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K_matrix)
    except cv2.error:
        return None
    if not retval:
        return None

    best = None
    best_score = -2.0
    for R, t, n in zip(Rs, Ts, Ns):
        n = np.asarray(n).reshape(3)
        t = np.asarray(t).reshape(3)
        if n[1] > 0:
            n, t = -n, -t
        score = float(np.dot(n, expected_normal))
        if score > best_score:
            best_score = score
            best = (np.asarray(R), -t, n, score)
    return best


def horizon_row(K: Intrinsics, tilt_deg: float = config.CAMERA_TILT_DEG,
                pitch_offset_deg: float = 0.0) -> Optional[float]:
    """Ligne d'horizon en pixels, ou None si elle sort de l'image."""
    theta = math.radians(-tilt_deg + pitch_offset_deg)
    if abs(math.cos(theta)) < 1e-9:
        return None
    row = K.cy - K.fy * math.tan(theta)
    if row < 0 or row > K.height:
        return None
    return row


def row_for_range(K: Intrinsics, range_m: float, height_m: float,
                  tilt_deg: float = config.CAMERA_TILT_DEG,
                  pitch_offset_deg: float = 0.0) -> Optional[float]:
    """Ligne d'image ou le sol se trouve a la distance demandee.

    Inverse de `ground_range` le long de l'axe median. Sert a tracer des
    reperes de distance directement sur l'image: le pilote lit alors la
    distance sans conversion mentale.
    """
    if range_m <= 1e-6 or height_m <= 0:
        return None
    theta = math.radians(-tilt_deg + pitch_offset_deg)
    alpha = math.atan2(height_m, range_m)          # depression du rayon
    row = K.cy + K.fy * math.tan(alpha - theta)
    if row < 0 or row > K.height:
        return None
    return row


def coverage(K: Intrinsics, height_m: float,
             tilt_deg: float = config.CAMERA_TILT_DEG) -> Tuple[float, float]:
    """Distances couvertes par le bas et le haut de l'image.

    Sert a dire franchement jusqu'ou la camera voit: a 45 degres de piquage,
    la portee plafonne autour de 2,2 fois la hauteur de vol. Ce n'est pas un
    defaut de l'algorithme, c'est le montage.
    """
    uv = np.array([[K.cx, K.height - 1.0], [K.cx, 0.0]])
    rng, valid = ground_range(uv, K, height_m, tilt_deg=tilt_deg)
    near = float(rng[0]) if valid[0] else float("nan")
    far = float(rng[1]) if valid[1] else float("inf")
    return near, far
