"""Simulateur de vol: images synthetiques avec verite terrain.

Sans reference connue, impossible de dire si une distance annoncee est juste.
Ce module fabrique donc des images par projection exacte d'un monde connu:
un sol texture et des obstacles verticaux places a des positions choisies. La
distance vraie est disponible a chaque image, ce qui permet de mesurer
l'erreur du reconstructeur au lieu de la supposer.

Il sert aussi de source video pour l'interface, pour travailler sans drone.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config, geometry
from .geometry import Intrinsics
from .mjpeg_client import Frame, LinkStats


@dataclass
class Wall:
    """Panneau vertical: x avant, y lateral, largeur, hauteur."""

    x_m: float
    y_m: float = 0.0
    width_m: float = 1.0
    height_m: float = 1.2
    texture: Optional[np.ndarray] = None


@dataclass
class WorldTruth:
    time_s: float
    camera_x: float
    camera_y: float
    height_m: float
    nearest_wall_range: Optional[float]


class FlightSimulator:
    """Camera piquee de 45 degres avancant en ligne droite au-dessus d'un sol."""

    def __init__(self, width: int = 320, height: int = 240,
                 height_m: float = 2.0, speed_mps: float = 0.8,
                 tilt_deg: float = config.CAMERA_TILT_DEG,
                 walls: Optional[List[Wall]] = None,
                 yaw_rate_dps: float = 0.0,
                 jpeg_quality: int = 52, seed: int = 7) -> None:
        self.K = Intrinsics.from_fov(width, height)
        self.width, self.height = width, height
        self.height_m = height_m
        self.speed_mps = speed_mps
        self.tilt_deg = tilt_deg
        self.yaw_rate_dps = yaw_rate_dps
        self.jpeg_quality = jpeg_quality
        self.walls = walls if walls is not None else [Wall(x_m=5.0)]

        rng = np.random.default_rng(seed)
        # Sol: 1 cm par pixel, de -3 m a +32 m devant, +/- 8 m lateralement.
        self.ground_ppm = 100.0
        self.ground_x0, self.ground_y0 = -3.0, -8.0
        gw = int((32.0 - self.ground_x0) * self.ground_ppm)
        gh = int((8.0 - self.ground_y0) * self.ground_ppm)
        tex = rng.integers(60, 200, size=(gh // 4, gw // 4), dtype=np.uint8)
        tex = cv2.resize(tex, (gw, gh), interpolation=cv2.INTER_LINEAR)
        tex = cv2.GaussianBlur(tex, (7, 7), 0)
        # Quelques marques contrastees: un sol trop uniforme ne donnerait pas
        # de points a suivre, et le test mesurerait la texture, pas le calcul.
        for _ in range(600):
            cx, cy = rng.integers(0, gw), rng.integers(0, gh)
            cv2.circle(tex, (int(cx), int(cy)), int(rng.integers(4, 14)),
                       int(rng.integers(20, 240)), -1)
        self.ground_tex = tex

        for wall in self.walls:
            if wall.texture is None:
                wt = rng.integers(0, 255, size=(120, 120), dtype=np.uint8)
                wall.texture = cv2.GaussianBlur(wt, (5, 5), 0)

    # -- geometrie ---------------------------------------------------------

    def yaw_rad(self, t: float) -> float:
        return math.radians(self.yaw_rate_dps) * t

    def camera_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Position monde et rotation monde<-camera a l'instant t.

        Le drone avance selon son cap; un lacet non nul decrit donc un arc.
        """
        if abs(self.yaw_rate_dps) < 1e-6:
            pos = np.array([self.speed_mps * t, 0.0, self.height_m])
        else:
            omega = math.radians(self.yaw_rate_dps)
            radius = self.speed_mps / omega
            psi = omega * t
            pos = np.array([radius * math.sin(psi),
                            radius * (1.0 - math.cos(psi)),
                            self.height_m])
        psi = self.yaw_rad(t)
        c, s = math.cos(psi), math.sin(psi)
        R_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return pos, R_wb @ geometry.body_from_camera(self.tilt_deg)

    def _plane_homography(self, origin: np.ndarray, axis_u: np.ndarray,
                          axis_v: np.ndarray, t: float) -> np.ndarray:
        """Homographie coordonnees-plan -> pixels."""
        pos, R_wc = self.camera_pose(t)
        R_cw = R_wc.T
        trans = -R_cw @ pos
        cols = np.column_stack([R_cw @ axis_u, R_cw @ axis_v, R_cw @ origin + trans])
        return self.K.matrix @ cols

    # -- rendu -------------------------------------------------------------

    def render(self, t: float) -> np.ndarray:
        H_ground = self._plane_homography(
            origin=np.array([0.0, 0.0, 0.0]),
            axis_u=np.array([1.0, 0.0, 0.0]),
            axis_v=np.array([0.0, 1.0, 0.0]),
            t=t,
        )
        # Texture -> coordonnees monde, puis monde -> pixels.
        A = np.array([[1.0 / self.ground_ppm, 0.0, self.ground_x0],
                      [0.0, 1.0 / self.ground_ppm, self.ground_y0],
                      [0.0, 0.0, 1.0]])
        H_tex = H_ground @ A

        img = cv2.warpPerspective(self.ground_tex, H_tex, (self.width, self.height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        for wall in self.walls:
            self._draw_wall(img, wall, t)

        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else bgr

    def _draw_wall(self, img: np.ndarray, wall: Wall, t: float) -> None:
        pos, R_wc = self.camera_pose(t)
        origin = np.array([wall.x_m, wall.y_m - wall.width_m / 2.0, 0.0])

        # Le panneau derriere la camera ne doit rien dessiner.
        #
        # Une homographie de plan ne sait pas dire "derriere": elle projette
        # quand meme, et les points de profondeur negative reviennent dans
        # l'image sous forme de surface etiree qui recouvre tout. On obtenait
        # ainsi, apres depassement du mur, des images uniformes -- que le
        # controle qualite signalait a juste titre comme inexploitables, mais
        # qui n'avaient aucune raison d'exister. Le test de profondeur se fait
        # donc sur les quatre coins, avant tout dessin.
        corners_world = np.array([
            origin,
            origin + np.array([0.0, wall.width_m, 0.0]),
            origin + np.array([0.0, wall.width_m, wall.height_m]),
            origin + np.array([0.0, 0.0, wall.height_m]),
        ])
        depth = (corners_world - pos) @ R_wc[:, 2]
        if np.any(depth <= 0.05):
            return

        H = self._plane_homography(origin,
                                   axis_u=np.array([0.0, 1.0, 0.0]),
                                   axis_v=np.array([0.0, 0.0, 1.0]),
                                   t=t)
        assert wall.texture is not None
        th, tw = wall.texture.shape[:2]
        A = np.array([[wall.width_m / tw, 0.0, 0.0],
                      [0.0, wall.height_m / th, 0.0],
                      [0.0, 0.0, 1.0]])
        H_tex = H @ A

        corners = np.array([[0.0, 0.0], [tw, 0.0], [tw, th], [0.0, th]], dtype=np.float64)
        proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_tex).reshape(-1, 2)
        if not np.isfinite(proj).all():
            return
        # Derriere la camera ou hors champ: rien a dessiner.
        if proj[:, 0].max() < 0 or proj[:, 0].min() > self.width:
            return
        if proj[:, 1].max() < 0 or proj[:, 1].min() > self.height:
            return

        warped = cv2.warpPerspective(wall.texture, H_tex, (self.width, self.height),
                                     flags=cv2.INTER_LINEAR)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, proj.astype(np.int32), 255)
        np.copyto(img, warped, where=mask.astype(bool))

    # -- defauts du lien ---------------------------------------------------

    @staticmethod
    def damage(img: np.ndarray, rng: np.random.Generator,
               blue_blocks: bool = True, blur: bool = True) -> np.ndarray:
        """Reproduit les defauts observes sur la vraie carte.

        Deux defauts distincts, aux consequences opposees pour l'analyse:
        des pave bleus, qui touchent surtout la couleur, et un flou franc, qui
        detruit les points suivis. Les simuler permet de mesurer la robustesse
        au lieu de la supposer.
        """
        out = img.copy()
        h, w = out.shape[:2]

        if blue_blocks:
            for _ in range(int(rng.integers(3, 9))):
                bw = int(rng.integers(12, 48))
                bh = int(rng.integers(6, 26))
                x = int(rng.integers(0, max(1, w - bw)))
                y = int(rng.integers(0, max(1, h - bh)))
                out[y:y + bh, x:x + bw, 0] = 255                    # canal bleu sature
                out[y:y + bh, x:x + bw, 1] //= 2
                out[y:y + bh, x:x + bw, 2] //= 2

        if blur:
            out = cv2.GaussianBlur(out, (9, 9), 3.0)

        return out

    # -- verite terrain ----------------------------------------------------

    def truth(self, t: float) -> WorldTruth:
        pos, _ = self.camera_pose(t)
        ranges = [w.x_m - pos[0] for w in self.walls if w.x_m - pos[0] > 0]
        return WorldTruth(
            time_s=t,
            camera_x=float(pos[0]),
            camera_y=float(pos[1]),
            height_m=self.height_m,
            nearest_wall_range=min(ranges) if ranges else None,
        )


class SimulatedLink:
    """Source video simulee, interface identique a VideoLink."""

    def __init__(self, simulator: Optional[FlightSimulator] = None,
                 fps: float = 7.0, on_log=None, **_ignored) -> None:
        self.sim = simulator or FlightSimulator()
        self.fps = fps
        self._on_log = on_log or (lambda msg: None)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._counter = 0
        self.stats = LinkStats()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats = LinkStats()
        self._thread = threading.Thread(target=self._run, name="sim-link", daemon=True)
        self._thread.start()
        self._on_log(f"simulation: {self.fps:.0f} img/s, mur a "
                     f"{self.sim.walls[0].x_m:.1f} m, vol a {self.sim.height_m:.1f} m")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        self.stats.connected = False

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def take_latest(self) -> Optional[Frame]:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def _run(self) -> None:
        self.stats.connected = True
        period = 1.0 / self.fps
        t_sim = 0.0
        next_at = time.time()
        while not self._stop.is_set():
            frame = self.sim.render(t_sim)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.sim.jpeg_quality])
            if ok:
                now = time.time()
                self._counter += 1
                with self._lock:
                    if self._latest is not None:
                        self.stats.dropped += 1
                    self._latest = Frame(jpeg=buf.tobytes(), recv_time=now, index=self._counter)
                self.stats.frames += 1
                self.stats.fps = self.fps
                self.stats.kbps = len(buf) * self.fps / 1024.0
            t_sim += period
            next_at += period
            self._stop.wait(max(0.0, next_at - time.time()))
        self.stats.connected = False
