"""Reconstruction 3D metrique a partir du flux monoculaire.

Ce que ce module etablit, image apres image:
  - l'assiette reelle de la camera, mesuree sur le sol lui-meme;
  - la distance metrique des points du sol (intersection rayon / plan);
  - le deplacement de la camera, mis a l'echelle par la hauteur de vol;
  - la position 3D des points qui ne sont pas sur le sol, par triangulation
    entre deux vues suffisamment ecartees.

Choix de conception: la derive est limitee a trois degres de liberte
------------------------------------------------------------------
Une odometrie monoculaire classique derive sur six axes. Ici le sol est
visible en permanence et il porte deux informations *absolues*:

  - sa normale donne le tangage et le roulis, sans integration donc sans
    derive;
  - la hauteur de vol donne l'altitude et l'echelle.

Il ne reste a integrer que l'avance, le deplacement lateral et le cap. Le cap
derive (rien ne l'observe sans magnetometre), les deux autres derivent
lentement. C'est la raison pour laquelle la scene reconstruite reste plate et
a la bonne echelle meme apres plusieurs dizaines de secondes, alors qu'une
odometrie libre l'aurait deja fait basculer.

Une hauteur fausse ne deforme rien: elle multiplie toute la scene, distances
et trajectoire comprises. Les rapports, l'ordre des obstacles et le temps
avant collision restent exacts.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import config, geometry
from .detector import STATE_NO_FLOW, DetectionResult
from .fusion import ConstantVelocity, HeadingFilter, wrap_angle
from .geometry import Intrinsics
from .structure import RayBundle

KIND_GROUND = 0
KIND_OBSTACLE = 1


@dataclass
class NearestObstacle:
    range_m: float
    band: Tuple[float, float]
    forward_m: float
    lateral_m: float
    height_m: float
    n_points: int


@dataclass
class MapFrame:
    """Sortie d'une image: geometrie mesuree et points produits."""

    ok: bool = False
    note: str = ""
    tilt_deg: float = config.CAMERA_TILT_DEG
    roll_deg: float = 0.0
    attitude_locked: bool = False
    height_m: float = config.DEFAULT_HEIGHT_M
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_deg: float = 0.0
    speed_mps: float = 0.0
    coverage: Tuple[float, float] = (0.0, 0.0)
    # Incertitude de position accumulee par l'odometrie. Elle croit tant que
    # rien ne la recale: c'est la description honnete d'une odometrie libre.
    position_sigma_m: float = 0.0

    # Points de l'image courante, repere drone (X avant, Y gauche, Z haut).
    ground_body: Optional[np.ndarray] = None
    ground_uv: Optional[np.ndarray] = None
    obstacle_body: Optional[np.ndarray] = None
    obstacle_uv: Optional[np.ndarray] = None
    obstacle_range: Optional[np.ndarray] = None
    obstacle_sigma: Optional[np.ndarray] = None

    nearest: Optional[NearestObstacle] = None
    contact: Optional[NearestObstacle] = None
    contact_uv: Optional[np.ndarray] = None
    contact_row: Optional[float] = None
    n_cloud: int = 0
    n_new_points: int = 0


class PointCloud:
    """Tampon circulaire de points 3D. Taille bornee, ecriture vectorisee.

    Chaque point porte quatre choses en plus de sa position: sa nature (sol ou
    relief), sa date, son incertitude et sa couleur relevee dans l'image. Les
    trois dernieres ne sont pas du decor:

      - la date permet d'oublier ce qui est trop ancien pour etre encore vrai;
      - l'incertitude permet de ponderer au lieu de seuiller, et de montrer a
        l'operateur ce que la reconstruction sait mal;
      - la couleur transforme un nuage de points en une scene reconnaissable,
        seul moyen pour un humain de verifier d'un coup d'oeil que la
        reconstruction correspond a ce qu'il voit.
    """

    def __init__(self, capacity: int = config.CLOUD_CAPACITY) -> None:
        self.capacity = capacity
        self.xyz = np.zeros((capacity, 3), dtype=np.float32)
        self.kind = np.zeros(capacity, dtype=np.uint8)
        self.stamp = np.zeros(capacity, dtype=np.float32)
        self.sigma = np.zeros(capacity, dtype=np.float32)
        self.bgr = np.zeros((capacity, 3), dtype=np.uint8)
        self._write = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(self, pts: np.ndarray, kind: int, stamp: float,
            sigma: Optional[np.ndarray] = None,
            bgr: Optional[np.ndarray] = None) -> None:
        n = len(pts)
        if n == 0:
            return
        if n >= self.capacity:
            # Garder la fin: ce sont les points les plus recents de l'apport.
            keep = slice(n - self.capacity, n)
            pts = pts[keep]
            sigma = sigma[keep] if sigma is not None else None
            bgr = bgr[keep] if bgr is not None else None
            n = len(pts)

        end = self._write + n
        if end <= self.capacity:
            parts = [(slice(self._write, end), slice(0, n))]
        else:
            first = self.capacity - self._write
            parts = [(slice(self._write, self.capacity), slice(0, first)),
                     (slice(0, n - first), slice(first, n))]

        for dst, src in parts:
            self.xyz[dst] = pts[src]
            self.kind[dst] = kind
            self.stamp[dst] = stamp
            self.sigma[dst] = 0.0 if sigma is None else sigma[src]
            self.bgr[dst] = 160 if bgr is None else bgr[src]

        self._write = end % self.capacity
        self._count = min(self.capacity, self._count + n)

    def view(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.xyz[:self._count], self.kind[:self._count], self.stamp[:self._count])

    def view_full(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
        c = self._count
        return (self.xyz[:c], self.kind[:c], self.stamp[:c],
                self.sigma[:c], self.bgr[:c])

    def shift(self, delta: np.ndarray) -> None:
        """Deplace tout le nuage (correction de derive apres recalage)."""
        d = np.asarray(delta, dtype=np.float32).reshape(3)
        if np.any(d):
            self.xyz[:self._count] += d

    def clear(self) -> None:
        self._write = 0
        self._count = 0


class Mapper:
    def __init__(self, height_m: float = config.DEFAULT_HEIGHT_M,
                 sigma_h_m: float = config.DEFAULT_SIGMA_H_M) -> None:
        self.height_m = height_m
        self.sigma_h_m = sigma_h_m
        self.calibrate_attitude = True

        self.cloud = PointCloud()
        self.trajectory: List[np.ndarray] = []

        self._K: Optional[Intrinsics] = None
        self._tilt = config.CAMERA_TILT_DEG
        self._roll = 0.0
        self._attitude_seen = False
        self._tilt_samples: deque = deque(maxlen=config.ATTITUDE_WINDOW)
        self._roll_samples: deque = deque(maxlen=config.ATTITUDE_WINDOW)
        self._yaw = 0.0
        self._pos = np.zeros(3, dtype=np.float64)
        self._last_stamp: Optional[float] = None
        self._speed = 0.0

        # Accumulation des visees par point suivi: la structure 3D en sort par
        # intersection de rayons, sans conserver l'historique des images.
        self._bundle = RayBundle(capacity=config.STRUCTURE_CAPACITY,
                                 sigma_px=config.STRUCTURE_SIGMA_PX,
                                 window=config.STRUCTURE_WINDOW)

        # Filtrage de la pose. L'odometrie mesure une vitesse par image, tres
        # bruitee sur une base aussi courte; le filtre la combine a un modele
        # de mouvement au lieu de la recopier telle quelle.
        self._pose_kf = ConstantVelocity(
            dim=2, accel_sigma=config.KALMAN_ACCEL_SIGMA,
            meas_sigma=config.KALMAN_VELOCITY_SIGMA,
            gate_sigma=config.KALMAN_GATE_SIGMA)
        self._heading_kf = HeadingFilter(
            rate_sigma=config.KALMAN_YAW_ACCEL_DPS2,
            meas_sigma=config.KALMAN_YAW_SIGMA_DPS,
            gate_sigma=config.KALMAN_GATE_SIGMA)
        self._rejected_motions = 0

    # -- reglages ----------------------------------------------------------

    def reset(self) -> None:
        self.cloud.clear()
        self.trajectory.clear()
        self._bundle.reset()
        self._yaw = 0.0
        self._pos = np.zeros(3, dtype=np.float64)
        self._tilt = config.CAMERA_TILT_DEG
        self._roll = 0.0
        self._attitude_seen = False
        self._tilt_samples.clear()
        self._roll_samples.clear()
        self._last_stamp = None
        self._speed = 0.0
        self._pose_kf.reset()
        self._heading_kf.reset()
        self._rejected_motions = 0

    def set_height(self, height_m: float, sigma_h_m: Optional[float] = None) -> None:
        self.height_m = max(0.05, float(height_m))
        if sigma_h_m is not None:
            self.sigma_h_m = max(0.0, float(sigma_h_m))

    # -- boucle ------------------------------------------------------------

    def update(self, result: DetectionResult,
               bgr: Optional[np.ndarray] = None) -> MapFrame:
        """Une image: assiette, pose, points du sol, structure, obstacle.

        `bgr` est facultatif et ne sert qu'a colorer les points produits. La
        geometrie n'en depend pas: sans image, la reconstruction est identique,
        seul le rendu perd ses couleurs.
        """
        w, h = result.work_size
        if w == 0 or h == 0 or result.pts_cur is None:
            return MapFrame(note="pas de points exploitables", height_m=self.height_m,
                            tilt_deg=self._tilt, roll_deg=self._roll,
                            position=self._pos.copy(), yaw_deg=math.degrees(self._yaw),
                            n_cloud=len(self.cloud))

        if self._K is None or self._K.width != w or self._K.height != h:
            self._K = Intrinsics.from_fov(w, h)

        frame = MapFrame(height_m=self.height_m)
        self._pos[2] = self.height_m

        # 1) Assiette: mesuree sur le sol quand c'est possible.
        self._update_attitude(result)
        frame.tilt_deg = self._tilt
        frame.roll_deg = self._roll
        frame.attitude_locked = self._attitude_seen

        # 2) Deplacement de la camera, a l'echelle de la hauteur de vol.
        self._update_pose(result)

        frame.position = self._pos.copy()
        frame.yaw_deg = math.degrees(self._yaw)
        frame.speed_mps = self._speed
        frame.position_sigma_m = self._pose_kf.position_sigma
        frame.coverage = geometry.coverage(self._K, self.height_m, self._tilt)

        # 3) Points du sol: distance directe, aucune integration, aucun besoin
        #    de mouvement. C'est ce qui fait que des distances restent
        #    affichees meme en vol stationnaire.
        self._project_ground(result, frame, bgr)

        # 4) Points hors sol: intersection des visees accumulees.
        #
        # Aucune condition de mouvement ici, contrairement a la triangulation
        # par paires qu'elle remplace: une visee supplementaire prise a
        # l'arret ne fausse rien, elle reduit le bruit de pointage sans
        # ajouter de base. C'est le conditionnement du systeme qui decide
        # quand le point est resolvable, et il le decide mieux qu'un test de
        # deplacement.
        self._structure(result, frame, bgr)

        # 5) Point de contact au sol: distance sans parallaxe, des la premiere
        #    image ou l'obstacle est visible.
        self._ground_contact(result, frame)

        frame.nearest = self._nearest_obstacle(frame, result.timestamp)
        frame.n_cloud = len(self.cloud)
        frame.ok = True
        if result.state == STATE_NO_FLOW:
            frame.note = "immobile: sol mesure, relief en attente de parallaxe"
        return frame

    # -- assiette ----------------------------------------------------------

    def _update_attitude(self, result: DetectionResult) -> None:
        if not self.calibrate_attitude or result.homography is None or self._K is None:
            return

        # L'assiette ne se mesure que sur une image ou le sol est *seul*.
        #
        # Ce critere a ete durci sur mesure. L'ancien seuil laissait passer
        # toute image ou le sol restait majoritaire; or un mur encore lointain
        # se plie a l'homographie du sol a un pixel pres -- sa parallaxe est
        # trop faible pour l'en distinguer -- tout en tirant la normale
        # ajustee vers lui. Le tangage estime derivait alors de quatre degres
        # et demi pendant l'approche, sans qu'aucun indicateur ne franchisse
        # son seuil. Quatre degres et demi, ce sont environ quatre-vingts
        # centimetres d'erreur sur un obstacle a deux metres et demi.
        #
        # On exige donc les deux temoins a la fois, et haut: presque tous les
        # points compatibles avec le plan, et presque aucun point signale hors
        # sol. Renoncer a mesurer l'assiette ne coute rien -- elle est
        # mecanique, la derniere valeur reste valable -- alors que la mesurer
        # sur une image contaminee fausse toute la reconstruction.
        if result.plane_inlier_ratio < config.ATTITUDE_MIN_INLIER_RATIO:
            return
        off = result.off_plane
        if off is not None and len(off) and float(off.mean()) > config.ATTITUDE_MAX_OFF_PLANE:
            return
        decomposition = self._decompose(result.homography)
        if decomposition is None:
            return
        _R, _t, normal = decomposition

        tilt, roll = geometry.attitude_from_plane_normal(normal)
        if abs(tilt - config.CAMERA_TILT_DEG) > config.ATTITUDE_MAX_DEVIATION_DEG:
            # Le plan trouve n'est pas le sol (mur, objet plat dominant).
            # Garder l'inclinaison nominale vaut mieux qu'un faux sol.
            return

        self._tilt_samples.append(tilt)
        self._roll_samples.append(roll)
        if len(self._tilt_samples) < config.ATTITUDE_MIN_SAMPLES:
            return

        # Mediane: une poignee d'images ou l'obstacle fausse le plan ne
        # deplace pas l'estimation, alors qu'une moyenne l'emporterait avec
        # elle.
        tilt_med = float(np.median(self._tilt_samples))
        roll_med = float(np.median(self._roll_samples))

        k = config.ATTITUDE_SMOOTHING
        if not self._attitude_seen:
            self._tilt, self._roll = tilt_med, roll_med
            self._attitude_seen = True
        else:
            self._tilt += k * (tilt_med - self._tilt)
            self._roll += k * (roll_med - self._roll)

    def _decompose(self, H: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Decompose l'homographie et retient la solution compatible avec le sol.

        La decomposition rend jusqu'a quatre solutions. On garde celle dont la
        normale ressemble le plus a celle attendue pour le sol -- c'est le seul
        critere qui ne demande pas d'information supplementaire.

        Convention de translation
        -------------------------
        Pour une homographie qui envoie l'image precedente sur l'image
        courante, `decomposeHomographyMat` rend une rotation qui verifie bien
        X2 = R X1, mais une translation de signe oppose a cette relation. Ce
        n'est pas une supposition: le simulateur fournit la pose exacte, et la
        comparaison donne t_renvoye = -R_wc2^T (p1 - p2) au chiffre pres, avec
        et sans lacet. On corrige donc ici, une fois, plutot que de propager
        une convention ambigue dans tout le module.
        """
        assert self._K is not None
        expected = geometry.expected_ground_normal(self._tilt, self._roll)
        decomposition = geometry.decompose_plane(H, self._K.matrix, expected)
        if decomposition is None:
            return None
        R, t, n, score = decomposition
        if score < config.GROUND_NORMAL_MIN_SCORE:
            return None
        return R, t, n

    # -- pose --------------------------------------------------------------

    def _update_pose(self, result: DetectionResult) -> bool:
        """Met a jour la pose. Renvoie True si la camera a bouge.

        Le deplacement mesure n'est plus integre directement: il entre dans un
        filtre a vitesse constante (voir `fusion.py`). Trois consequences
        concretes, toutes verifiees par les tests:

          - une image sans homographie exploitable ne fige plus la trajectoire,
            le modele la prolonge;
          - une mesure incompatible avec l'etat est ecartee sur un critere
            statistique, la ou seul un plafond de vitesse en dur filtrait;
          - un stationnaire est traite comme une *mesure* de vitesse nulle et
            non comme une absence de mesure, ce qui arrete net la derive.
        """
        dt = result.dt if result.dt > 0 else 0.0
        if self._last_stamp is not None and result.timestamp <= self._last_stamp:
            return False
        self._last_stamp = result.timestamp

        use_kf = config.KALMAN_ENABLED
        if use_kf and dt > 0.0:
            self._pose_kf.predict(dt)
            self._heading_kf.predict(dt)

        if result.state == STATE_NO_FLOW or result.homography is None:
            if use_kf:
                if result.state == STATE_NO_FLOW and dt > 0.0:
                    # Scene immobile: information, pas silence.
                    self._pose_kf.update_velocity(np.zeros(2), sigma=config.KALMAN_ZUPT_SIGMA)
                    self._heading_kf.update_delta(0.0, dt,
                                                  sigma_dps=config.KALMAN_ZUPT_YAW_DPS)
                self._sync_from_filters()
            self._speed = 0.0 if result.state == STATE_NO_FLOW else self._speed
            return False

        decomposition = self._decompose(result.homography)
        if decomposition is None:
            if use_kf:
                self._sync_from_filters()
            return False
        R_c1c2, t_over_d, _n = decomposition

        # La decomposition rend la translation divisee par la distance au plan.
        # Cette distance, c'est la hauteur de vol: c'est la seule grandeur
        # metrique de tout le systeme, et c'est elle qui donne l'echelle.
        t_metric = t_over_d * self.height_m

        R_wc_before = self._R_wc()
        R_wc_after = R_wc_before @ R_c1c2.T
        delta = -(R_wc_after @ t_metric)

        step = float(np.hypot(delta[0], delta[1]))
        speed = step / dt if dt > 0 else 0.0
        # Garde-fou physique, applique avant le filtre: ce drone ne depasse pas
        # quelques metres par seconde. Une vitesse au-dessus vient d'une
        # homographie fausse -- typiquement un obstacle qui remplit l'image et
        # se fait passer pour le sol.
        if not np.isfinite(step) or speed > config.MAX_SPEED_MPS:
            if use_kf:
                self._sync_from_filters()
            self._rejected_motions += 1
            return False

        R_wb_after = R_wc_after @ geometry.body_from_camera(self._tilt, self._roll).T
        yaw_measured = math.atan2(R_wb_after[1, 0], R_wb_after[0, 0])

        if use_kf and dt > 0.0:
            accepted = self._pose_kf.update_velocity(delta[:2] / dt)
            self._heading_kf.update_delta(wrap_angle(yaw_measured - self._yaw), dt)
            self._sync_from_filters()
            if not accepted:
                # Mesure jugee aberrante: elle ne doit pas non plus servir de
                # base a la triangulation, qui l'utiliserait telle quelle.
                self._rejected_motions += 1
                return False
        else:
            self._pos[0] += delta[0]
            self._pos[1] += delta[1]
            self._yaw = yaw_measured
            self._speed = speed

        self._pos[2] = self.height_m
        self._push_trajectory()
        return step > 1e-4

    def _sync_from_filters(self) -> None:
        """Recopie l'etat filtre dans la pose exposee au reste du module."""
        self._pos[:2] = self._pose_kf.position
        self._pos[2] = self.height_m
        self._yaw = self._heading_kf.yaw_rad
        self._speed = self._pose_kf.speed

    def _push_trajectory(self) -> None:
        """Ajoute la pose courante au trace, sans empiler les doublons.

        En stationnaire la pose ne change pas: enregistrer chaque image
        remplirait le tampon de points identiques et effacerait le debut du
        vol, qui lui porte de l'information.
        """
        if self.trajectory and float(np.hypot(*(self._pos[:2] - self.trajectory[-1][:2]))) < 1e-3:
            return
        self.trajectory.append(self._pos.copy())
        if len(self.trajectory) > config.TRAJECTORY_CAPACITY:
            del self.trajectory[:len(self.trajectory) - config.TRAJECTORY_CAPACITY]

    def _R_wb(self) -> np.ndarray:
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _R_wc(self) -> np.ndarray:
        return self._R_wb() @ geometry.body_from_camera(self._tilt, self._roll)

    # -- points sol --------------------------------------------------------

    def _project_ground(self, result: DetectionResult, frame: MapFrame,
                        bgr: Optional[np.ndarray] = None) -> None:
        assert self._K is not None
        uv = result.pts_cur
        if uv is None or len(uv) == 0:
            return

        on_plane = ~result.off_plane if result.off_plane is not None else np.ones(len(uv), bool)
        pts, valid = geometry.ground_points(uv, self._K, self.height_m,
                                            tilt_deg=self._tilt, roll_deg=self._roll)

        keep = valid & on_plane & np.isfinite(pts).all(axis=1)
        keep &= np.hypot(pts[:, 0], pts[:, 1]) < config.MAX_POINT_RANGE_M
        if not keep.any():
            return

        body = pts[keep]
        frame.ground_body = body.astype(np.float32)
        frame.ground_uv = uv[keep]

        world = self._to_world(body)
        # L'incertitude d'un point du sol vient de la hauteur de vol, pas de la
        # geometrie: elle est proportionnelle a la distance, comme la distance
        # elle-meme est proportionnelle a h.
        sigma = (np.hypot(body[:, 0], body[:, 1])
                 * (self.sigma_h_m / max(1e-6, self.height_m))).astype(np.float32)
        self.cloud.add(world.astype(np.float32), KIND_GROUND, result.timestamp,
                       sigma=sigma,
                       bgr=self._sample_colours(bgr, frame.ground_uv, result.scale))
        frame.n_new_points += len(world)

    def _to_world(self, body: np.ndarray) -> np.ndarray:
        return (body @ self._R_wb().T) + self._pos

    def _to_body(self, world: np.ndarray) -> np.ndarray:
        return (world - self._pos) @ self._R_wb()

    # -- contact au sol ----------------------------------------------------

    def _ground_contact(self, result: DetectionResult, frame: MapFrame) -> None:
        """Distance a l'obstacle par son point d'appui sur le sol.

        Un objet pose au sol touche le plan quelque part. Ce point-la, lui,
        appartient au sol: son rayon perce le plan, donc sa distance est
        directement metrique -- sans deuxieme vue, sans parallaxe, donc des la
        premiere image ou l'obstacle apparait. C'est la mesure la plus tot
        disponible et la plus stable des trois.

        Prendre simplement le point hors-plan le plus bas donne une distance
        systematiquement trop grande, et pour une raison de fond: pres du pied
        de l'obstacle, l'ecart au sol tend vers zero, donc ces points-la
        ressemblent au sol et ne sont jamais signales. Le plus bas point
        *detecte* est toujours au-dessus du vrai pied, et son rayon retombe
        derriere l'obstacle.

        On corrige en extrapolant: l'ecart au plan croit avec la hauteur
        au-dessus du sol, donc en regressant l'ecart sur la ligne d'image on
        trouve la ligne ou il s'annule. C'est la ligne de contact.
        """
        assert self._K is not None
        if result.off_plane is None or result.pts_cur is None or result.residuals is None:
            return
        sel = np.flatnonzero(result.off_plane)
        if len(sel) < config.MIN_CONTACT_POINTS:
            return

        uv = result.pts_cur[sel]
        res = result.residuals[sel]

        # Un seul amas a la fois: on garde la bande de colonnes autour du gros
        # de l'obstacle, sinon deux objets distincts se melangeraient.
        u_med = float(np.median(uv[:, 0]))
        band = np.abs(uv[:, 0] - u_med) < config.CONTACT_COLUMN_BAND * result.work_size[0]
        if band.sum() < config.MIN_CONTACT_POINTS:
            return
        uv, res = uv[band], res[band]

        v_low = float(uv[:, 1].max())
        v_base = v_low
        if len(uv) >= config.MIN_CONTACT_POINTS and np.ptp(uv[:, 1]) > 2.0:
            slope, intercept = np.polyfit(uv[:, 1], res, 1)
            if slope < -1e-6:
                candidate = -intercept / slope
                # L'extrapolation reste bornee: au-dela, l'obstacle est trop
                # proche pour que son pied soit encore dans le champ, et la
                # droite ajustee ne dit plus rien de fiable.
                limit = result.work_size[1] * config.CONTACT_EXTRAPOLATION_LIMIT
                v_base = float(np.clip(candidate, v_low, limit))

        probe = np.array([[u_med, v_base]])
        pts, valid = geometry.ground_points(probe, self._K, self.height_m,
                                            tilt_deg=self._tilt, roll_deg=self._roll)
        if not valid[0] or not np.isfinite(pts[0]).all():
            return
        r = float(np.hypot(pts[0, 0], pts[0, 1]))
        if not np.isfinite(r) or r > config.MAX_POINT_RANGE_M:
            return

        frame.contact_uv = uv
        frame.contact_row = v_base
        frame.contact = NearestObstacle(
            range_m=r,
            band=geometry.range_band(r, self.height_m, self.sigma_h_m),
            forward_m=float(pts[0, 0]),
            lateral_m=float(pts[0, 1]),
            height_m=0.0,
            n_points=int(len(uv)),
        )

    # -- triangulation -----------------------------------------------------

    def _structure(self, result: DetectionResult, frame: MapFrame,
                   bgr: Optional[np.ndarray] = None) -> None:
        """Position 3D des points hors sol, par intersection de visees.

        Un obstacle qui ne touche pas le sol (branche, cable, mur vu de face)
        n'a pas de point de contact: l'intersection avec le plan ne dit rien de
        lui. Plusieurs visees le situent.

        Chaque image ajoute un rayon par point suivi. Le point cherche est
        celui qui passe au plus pres de tous ces rayons; le systeme est resolu
        d'un coup pour tous les points de l'image (voir `structure.py`). Deux
        differences avec une triangulation par paires:

          - la base utilisee est celle de toute la fenetre d'observation, pas
            celle de deux images consecutives. Or l'incertitude varie comme
            l'inverse de la base: c'est la ou se joue la precision.
          - l'incertitude de chaque point sort du calcul lui-meme, ce qui
            permet d'ecarter un point sur ce qu'il vaut reellement plutot que
            sur un angle de parallaxe suppose representatif.
        """
        assert self._K is not None
        if (result.off_plane is None or result.track_ids is None
                or result.pts_cur is None):
            return

        sel = np.flatnonzero(result.off_plane)
        if len(sel) == 0:
            return

        uv = result.pts_cur[sel]
        ids = result.track_ids[sel]

        R_bc = geometry.body_from_camera(self._tilt, self._roll)
        rays_world = geometry.rays_body(uv, self._K, R_bc) @ self._R_wb().T

        rows = self._bundle.observe(ids, self._pos, rays_world)
        world, sigma, solved = self._bundle.solve(
            rows,
            min_views=config.STRUCTURE_MIN_VIEWS,
            max_sigma_m=config.STRUCTURE_MAX_SIGMA_M,
            focal_px=self._K.fx,
        )
        if len(world) == 0:
            return

        body = self._to_body(world)
        keep = self._plausible(body, sigma)
        if not keep.any():
            return

        body = body[keep]
        world = world[keep]
        sigma = sigma[keep].astype(np.float32)
        kept_uv = uv[solved][keep]

        frame.obstacle_body = body.astype(np.float32)
        frame.obstacle_range = np.hypot(body[:, 0], body[:, 1]).astype(np.float32)
        frame.obstacle_uv = kept_uv
        frame.obstacle_sigma = sigma

        colours = self._sample_colours(bgr, kept_uv, result.scale)
        self.cloud.add(world.astype(np.float32), KIND_OBSTACLE, result.timestamp,
                       sigma=sigma, bgr=colours)
        frame.n_new_points += len(world)

    def _plausible(self, body: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Ne garde que des points geometriquement defendables.

        Le critere decisif est le dernier: un obstacle *depasse du sol*. Sans
        lui, tout point mal ajuste par l'homographie devenait un obstacle, y
        compris un point du sol situe juste devant l'objet. Ces points-la sont
        plus proches que l'obstacle lui-meme et tiraient la distance annoncee
        vers le bas -- l'erreur mesuree atteignait 60 cm sur un mur a 3 m.

        Le seuil n'est pas fixe: il vaut au moins la hauteur en dessous de
        laquelle un relief n'interesse personne, et au moins l'incertitude
        propre du point. Un point dont la hauteur ne depasse pas sa propre
        barre d'erreur n'a pas prouve qu'il n'etait pas au sol.
        """
        ok = np.isfinite(body).all(axis=1)
        ok &= body[:, 0] > 0.05                          # devant le drone
        ok &= np.hypot(body[:, 0], body[:, 1]) < config.MAX_POINT_RANGE_M

        above_ground = body[:, 2] + self.height_m
        # Sous le sol: impossible, donc erreur de reconstruction.
        ok &= above_ground > -0.5
        ok &= above_ground > np.maximum(config.OBSTACLE_MIN_HEIGHT_M,
                                        config.OBSTACLE_HEIGHT_SIGMA_K * sigma)
        return ok

    def _sample_colours(self, bgr: Optional[np.ndarray], uv: np.ndarray,
                        scale: float) -> Optional[np.ndarray]:
        """Couleur de l'image sous chaque point, a la resolution d'affichage.

        Les coordonnees sont donnees a la resolution de travail; l'image
        affichee est plus grande. Le facteur est celui que le detecteur a deja
        calcule, il ne se redecouvre pas ici.
        """
        if bgr is None or len(uv) == 0:
            return None
        h, w = bgr.shape[:2]
        x = np.clip((uv[:, 0] * scale).astype(np.int32), 0, w - 1)
        y = np.clip((uv[:, 1] * scale).astype(np.int32), 0, h - 1)
        return bgr[y, x]

    # -- lecture -----------------------------------------------------------

    def _nearest_obstacle(self, frame: MapFrame, now: float) -> Optional[NearestObstacle]:
        """Obstacle le plus proche, a partir des points recents accumules.

        Se limiter aux points tries dans l'image courante rendrait la distance
        sautillante: quelques images sans triangulation valide et l'obstacle
        "disparaitrait". On interroge donc les dernieres secondes du nuage.

        La distance retenue est un quantile bas, pas le minimum: le minimum
        d'un nuage bruite est un point aberrant par construction.

        Les points mal situes sont ecartes avant le quantile plutot que
        moyennes avec les autres. Chaque point connait desormais sa propre
        incertitude (voir `structure.py`); un point dont l'incertitude depasse
        sa contribution utile n'ajoute pas de l'information bruitee, il ajoute
        du bruit tout court.
        """
        xyz, kind, stamp, sigma, _bgr = self.cloud.view_full()
        if len(xyz) == 0:
            return None
        recent = (kind == KIND_OBSTACLE) & (stamp > now - config.OBSTACLE_MEMORY_S)
        if recent.sum() < config.MIN_OBSTACLE_POINTS:
            return None

        body = self._to_body(xyz[recent].astype(np.float64))
        conf = sigma[recent]
        ahead = (body[:, 0] > 0.1) & (np.abs(body[:, 1]) < config.OBSTACLE_CORRIDOR_M)
        trusted = ahead & (conf <= config.OBSTACLE_MAX_SIGMA_M)
        # Si le tri par incertitude ne laisse pas assez de monde, on retombe
        # sur l'ensemble: mieux vaut une distance imprecise et signalee comme
        # telle que pas de distance du tout devant un obstacle.
        if trusted.sum() >= config.MIN_OBSTACLE_POINTS:
            ahead = trusted
        elif ahead.sum() < config.MIN_OBSTACLE_POINTS:
            return None

        sub = body[ahead]
        rng = np.hypot(sub[:, 0], sub[:, 1])
        r = float(np.quantile(rng, config.OBSTACLE_RANGE_QUANTILE))
        near = sub[rng <= max(r, rng.min() + 1e-6)]
        centre = near.mean(axis=0)

        # L'intervalle affiche combine les deux sources d'erreur: l'echelle,
        # mal connue par la hauteur de vol, et la dispersion propre des points
        # reconstruits. Elles sont independantes, donc elles s'additionnent en
        # quadrature et non l'une a l'autre.
        lo, hi = geometry.range_band(r, self.height_m, self.sigma_h_m)
        spread = float(np.median(conf[ahead])) if conf[ahead].size else 0.0
        half = math.hypot((hi - lo) / 2.0, spread)
        return NearestObstacle(
            range_m=r,
            band=(max(0.0, r - half), r + half),
            forward_m=float(centre[0]),
            lateral_m=float(centre[1]),
            height_m=float(centre[2]),
            n_points=int(ahead.sum()),
        )
