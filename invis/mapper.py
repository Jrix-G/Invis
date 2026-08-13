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

import cv2
import numpy as np

from . import config, geometry
from .detector import STATE_NO_FLOW, DetectionResult
from .geometry import Intrinsics

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

    # Points de l'image courante, repere drone (X avant, Y gauche, Z haut).
    ground_body: Optional[np.ndarray] = None
    ground_uv: Optional[np.ndarray] = None
    obstacle_body: Optional[np.ndarray] = None
    obstacle_uv: Optional[np.ndarray] = None
    obstacle_range: Optional[np.ndarray] = None

    nearest: Optional[NearestObstacle] = None
    contact: Optional[NearestObstacle] = None
    contact_uv: Optional[np.ndarray] = None
    contact_row: Optional[float] = None
    n_cloud: int = 0
    n_new_points: int = 0


class PointCloud:
    """Tampon circulaire de points 3D. Taille bornee, ecriture vectorisee."""

    def __init__(self, capacity: int = config.CLOUD_CAPACITY) -> None:
        self.capacity = capacity
        self.xyz = np.zeros((capacity, 3), dtype=np.float32)
        self.kind = np.zeros(capacity, dtype=np.uint8)
        self.stamp = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def add(self, pts: np.ndarray, kind: int, stamp: float) -> None:
        n = len(pts)
        if n == 0:
            return
        if n >= self.capacity:
            pts = pts[-self.capacity:]
            n = len(pts)

        end = self._write + n
        if end <= self.capacity:
            sl = slice(self._write, end)
            self.xyz[sl] = pts
            self.kind[sl] = kind
            self.stamp[sl] = stamp
        else:
            first = self.capacity - self._write
            self.xyz[self._write:] = pts[:first]
            self.kind[self._write:] = kind
            self.stamp[self._write:] = stamp
            self.xyz[:n - first] = pts[first:]
            self.kind[:n - first] = kind
            self.stamp[:n - first] = stamp

        self._write = end % self.capacity
        self._count = min(self.capacity, self._count + n)

    def view(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.xyz[:self._count], self.kind[:self._count], self.stamp[:self._count])

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

        # Mesures de profondeur par point suivi, pour la mediane glissante.
        self._depth_hist: dict = {}
        # Derniere pose relative mesuree (rotation, translation metrique).
        self._last_motion: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -- reglages ----------------------------------------------------------

    def reset(self) -> None:
        self.cloud.clear()
        self.trajectory.clear()
        self._depth_hist.clear()
        self._last_motion = None
        self._yaw = 0.0
        self._pos = np.zeros(3, dtype=np.float64)
        self._tilt = config.CAMERA_TILT_DEG
        self._roll = 0.0
        self._attitude_seen = False
        self._tilt_samples.clear()
        self._roll_samples.clear()
        self._last_stamp = None
        self._speed = 0.0

    def set_height(self, height_m: float, sigma_h_m: Optional[float] = None) -> None:
        self.height_m = max(0.05, float(height_m))
        if sigma_h_m is not None:
            self.sigma_h_m = max(0.0, float(sigma_h_m))

    # -- boucle ------------------------------------------------------------

    def update(self, result: DetectionResult) -> MapFrame:
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
        moved = self._update_pose(result)

        frame.position = self._pos.copy()
        frame.yaw_deg = math.degrees(self._yaw)
        frame.speed_mps = self._speed
        frame.coverage = geometry.coverage(self._K, self.height_m, self._tilt)

        # 3) Points du sol: distance directe, aucune integration, aucun besoin
        #    de mouvement. C'est ce qui fait que des distances restent
        #    affichees meme en vol stationnaire.
        self._project_ground(result, frame)

        # 4) Points hors sol: triangulation sur la paire d'images courante.
        if moved and self._last_motion is not None:
            self._triangulate(result, frame, *self._last_motion)

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
        if result.plane_inlier_ratio < config.PLANE_INLIER_MIN_RATIO:
            # Le plan trouve n'est pas majoritaire: c'est probablement un mur
            # ou un objet plat qui remplit l'image, pas le sol. Recalibrer
            # l'assiette dessus ferait basculer toute la reconstruction.
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
        """Integre le deplacement. Renvoie True si la camera a bouge."""
        dt = result.dt if result.dt > 0 else 0.0
        if self._last_stamp is not None and result.timestamp <= self._last_stamp:
            return False
        self._last_stamp = result.timestamp

        if result.state == STATE_NO_FLOW or result.homography is None:
            self._speed = 0.0
            return False

        decomposition = self._decompose(result.homography)
        if decomposition is None:
            self._speed = 0.0
            return False
        R_c1c2, t_over_d, _n = decomposition
        self._last_motion = (R_c1c2, t_over_d * self.height_m)

        # La decomposition rend la translation divisee par la distance au plan.
        # Cette distance, c'est la hauteur de vol: c'est la seule grandeur
        # metrique de tout le systeme, et c'est elle qui donne l'echelle.
        t_metric = t_over_d * self.height_m

        R_wc_before = self._R_wc()
        R_wc_after = R_wc_before @ R_c1c2.T
        delta = -(R_wc_after @ t_metric)

        step = float(np.hypot(delta[0], delta[1]))
        speed = step / dt if dt > 0 else 0.0
        # Garde-fou: ce drone ne depasse pas quelques metres par seconde. Une
        # vitesse au-dessus vient d'une homographie fausse -- typiquement un
        # obstacle qui remplit l'image et se fait passer pour le sol. Mieux
        # vaut declarer l'estimation indisponible que la propager dans la
        # trajectoire.
        if not np.isfinite(step) or speed > config.MAX_SPEED_MPS:
            self._speed = 0.0
            return False

        self._pos[0] += delta[0]
        self._pos[1] += delta[1]
        self._pos[2] = self.height_m

        # Seul le cap s'integre: tangage et roulis viennent du sol a chaque
        # image, donc ils ne derivent pas.
        R_wb_after = R_wc_after @ geometry.body_from_camera(self._tilt, self._roll).T
        self._yaw = math.atan2(R_wb_after[1, 0], R_wb_after[0, 0])

        self._speed = speed
        self.trajectory.append(self._pos.copy())
        if len(self.trajectory) > config.TRAJECTORY_CAPACITY:
            del self.trajectory[:len(self.trajectory) - config.TRAJECTORY_CAPACITY]
        return step > 1e-4

    def _R_wb(self) -> np.ndarray:
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _R_wc(self) -> np.ndarray:
        return self._R_wb() @ geometry.body_from_camera(self._tilt, self._roll)

    # -- points sol --------------------------------------------------------

    def _project_ground(self, result: DetectionResult, frame: MapFrame) -> None:
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
        self.cloud.add(world.astype(np.float32), KIND_GROUND, result.timestamp)
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

    def _triangulate(self, result: DetectionResult, frame: MapFrame,
                     R_rel: np.ndarray, t_rel: np.ndarray) -> None:
        """Position 3D des points hors sol, sur la paire d'images courante.

        Un obstacle qui ne touche pas le sol (branche, cable, mur vu de face)
        n'a pas de point de contact: l'intersection avec le plan ne dit rien de
        lui. Deux visees le situent.

        La triangulation utilise le *deplacement relatif* entre les deux
        dernieres images, pas la position integree depuis le debut. La
        difference est decisive: la position integree derive, et elle derive le
        plus quand l'obstacle occupe l'image, c'est-a-dire au moment precis ou
        la mesure compte. Le deplacement relatif est remesure a chaque image et
        ne cumule rien.

        La contrepartie est une base courte, donc une profondeur bruitee. Elle
        est compensee par une mediane glissante par point suivi: le meme detail
        est mesure plusieurs images de suite, et la mediane de ces mesures vaut
        bien mieux que chacune d'elles.
        """
        assert self._K is not None
        if (result.off_plane is None or result.track_ids is None
                or result.pts_prev is None or result.pts_cur is None):
            return

        sel = np.flatnonzero(result.off_plane)
        if len(sel) == 0:
            return

        K = self._K.matrix
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R_rel, t_rel.reshape(3, 1)])

        uv0 = result.pts_prev[sel].T.astype(np.float64)
        uv1 = result.pts_cur[sel].T.astype(np.float64)
        homog = cv2.triangulatePoints(P1, P2, uv0, uv1)
        w = homog[3]
        finite = np.abs(w) > 1e-9
        X1 = np.full((3, len(sel)), np.nan)
        X1[:, finite] = homog[:3, finite] / w[finite]
        X1 = X1.T

        X2 = X1 @ R_rel.T + t_rel                    # repere camera courante
        body = X2 @ geometry.body_from_camera(self._tilt, self._roll).T

        good = self._filter_points(X1, X2, body, t_rel)
        if not good.any():
            return

        ids = result.track_ids[sel][good]
        measured = body[good]
        kept_uv = result.pts_cur[sel][good]

        stable: List[np.ndarray] = []
        stable_uv: List[np.ndarray] = []
        for tid, pt, uv in zip(ids, measured, kept_uv):
            hist = self._depth_hist.setdefault(int(tid), deque(maxlen=config.DEPTH_HISTORY))
            hist.append(pt)
            if len(hist) >= config.DEPTH_MIN_SAMPLES:
                stable.append(np.median(np.asarray(hist), axis=0))
                stable_uv.append(uv)

        self._prune_depth_history(result.track_ids)
        if not stable:
            return

        pts_body = np.asarray(stable)
        frame.obstacle_body = pts_body.astype(np.float32)
        frame.obstacle_range = np.hypot(pts_body[:, 0], pts_body[:, 1]).astype(np.float32)
        frame.obstacle_uv = np.asarray(stable_uv)

        world = self._to_world(pts_body)
        self.cloud.add(world.astype(np.float32), KIND_OBSTACLE, result.timestamp)
        frame.n_new_points += len(world)

    def _prune_depth_history(self, alive_ids: np.ndarray) -> None:
        if len(self._depth_hist) <= 4 * config.MAX_FEATURES:
            return
        alive = set(int(i) for i in alive_ids)
        self._depth_hist = {k: v for k, v in self._depth_hist.items() if k in alive}

    def _filter_points(self, X1: np.ndarray, X2: np.ndarray, body: np.ndarray,
                       t_rel: np.ndarray) -> np.ndarray:
        """Ne garde que des points geometriquement defendables."""
        ok = np.isfinite(X1).all(axis=1) & np.isfinite(X2).all(axis=1)
        ok &= X1[:, 2] > 0.05           # devant la camera dans les deux vues
        ok &= X2[:, 2] > 0.05
        rng = np.hypot(body[:, 0], body[:, 1])
        ok &= rng < config.MAX_POINT_RANGE_M
        # Sous le sol: impossible, donc erreur de triangulation.
        ok &= body[:, 2] > -self.height_m - 0.5

        # Angle entre les deux visees, vu depuis le point. Deux rayons presque
        # paralleles se coupent n'importe ou: la profondeur sortirait du bruit
        # et non de la geometrie.
        centre2 = -t_rel                # centre de la 2e camera, repere 1
        v1 = X1
        v2 = X1 - centre2
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        safe = (n1 > 1e-6) & (n2 > 1e-6)
        cos = np.ones(len(X1))
        np.divide((v1 * v2).sum(axis=1), n1 * n2, out=cos, where=safe)
        parallax = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        ok &= parallax >= config.MIN_PARALLAX_DEG
        return ok

    # -- lecture -----------------------------------------------------------

    def _nearest_obstacle(self, frame: MapFrame, now: float) -> Optional[NearestObstacle]:
        """Obstacle le plus proche, a partir des points recents accumules.

        Se limiter aux points tries dans l'image courante rendrait la distance
        sautillante: quelques images sans triangulation valide et l'obstacle
        "disparaitrait". On interroge donc les dernieres secondes du nuage.

        La distance retenue est un quantile bas, pas le minimum: le minimum
        d'un nuage bruite est un point aberrant par construction.
        """
        xyz, kind, stamp = self.cloud.view()
        if len(xyz) == 0:
            return None
        recent = (kind == KIND_OBSTACLE) & (stamp > now - config.OBSTACLE_MEMORY_S)
        if recent.sum() < config.MIN_OBSTACLE_POINTS:
            return None

        body = self._to_body(xyz[recent].astype(np.float64))
        ahead = (body[:, 0] > 0.1) & (np.abs(body[:, 1]) < config.OBSTACLE_CORRIDOR_M)
        if ahead.sum() < config.MIN_OBSTACLE_POINTS:
            return None

        sub = body[ahead]
        rng = np.hypot(sub[:, 0], sub[:, 1])
        r = float(np.quantile(rng, config.OBSTACLE_RANGE_QUANTILE))
        near = sub[rng <= max(r, rng.min() + 1e-6)]
        centre = near.mean(axis=0)
        return NearestObstacle(
            range_m=r,
            band=geometry.range_band(r, self.height_m, self.sigma_h_m),
            forward_m=float(centre[0]),
            lateral_m=float(centre[1]),
            height_m=float(centre[2]),
            n_points=int(ahead.sum()),
        )
