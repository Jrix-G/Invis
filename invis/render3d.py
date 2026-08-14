"""Rendu du nuage de points 3D, en numpy pur.

Aucune bibliotheque 3D: la scene tient en quelques dizaines de milliers de
points, et une projection vectorisee suivie d'une ecriture directe dans le
tampon image coute moins d'une milliseconde. Passer par un moteur de rendu
couterait plus en installation et en latence qu'il ne rapporterait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .mapper import KIND_GROUND, KIND_OBSTACLE

COLOR_BG = (22, 22, 26)
COLOR_GRID = (52, 52, 58)
COLOR_GRID_MAIN = (78, 78, 86)
COLOR_GROUND_PT = (150, 130, 95)
COLOR_OBSTACLE_PT = (70, 90, 245)
COLOR_TRAJ = (120, 220, 120)
COLOR_DRONE = (250, 250, 250)
COLOR_TEXT = (215, 215, 215)
COLOR_DIM = (130, 130, 138)

# Modes de couleur. Chacun repond a une question differente: a quoi ressemble
# le terrain, quel est son relief, et jusqu'ou peut-on croire ce qu'on voit.
MODE_REAL = "reelle"
MODE_HEIGHT = "hauteur"
MODE_CONFIDENCE = "fiabilite"
MODE_PLAIN = "uniforme"
COLOUR_MODES = (MODE_REAL, MODE_HEIGHT, MODE_CONFIDENCE, MODE_PLAIN)


def _ramp(t: np.ndarray) -> np.ndarray:
    """Degrade bleu -> vert -> jaune, en BGR.

    Choisi pour rester lisible en luminance croissante: le bleu sombre se lit
    comme "bas" ou "douteux" et le jaune clair comme "haut" ou "sur", y compris
    pour un oeil qui distingue mal le rouge du vert.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)[:, None]
    cold = np.array([140.0, 60.0, 30.0], dtype=np.float32)
    mid = np.array([90.0, 190.0, 80.0], dtype=np.float32)
    hot = np.array([70.0, 235.0, 245.0], dtype=np.float32)
    lower = cold + (mid - cold) * np.clip(t * 2.0, 0.0, 1.0)
    upper = mid + (hot - mid) * np.clip(t * 2.0 - 1.0, 0.0, 1.0)
    return np.where(t < 0.5, lower, upper)


@dataclass
class OrbitCamera:
    """Camera d'observation en coordonnees spheriques autour d'une cible."""

    yaw_deg: float = config.VIEW3D_DEFAULT_YAW_DEG
    pitch_deg: float = config.VIEW3D_DEFAULT_PITCH_DEG
    range_m: float = config.VIEW3D_DEFAULT_RANGE_M
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_deg: float = 55.0

    def orbit(self, dyaw: float, dpitch: float) -> None:
        """Tourne autour de la cible.

        Le tangage descend sous l'horizon: regarder la scene par en dessous
        sert a juger la hauteur des points, ce qu'une vue plongeante rend
        justement difficile. Il reste borne juste avant la verticale, ou le
        repere de la camera devient indetermine.
        """
        self.yaw_deg = (self.yaw_deg + dyaw) % 360.0
        self.pitch_deg = float(np.clip(self.pitch_deg + dpitch, -88.0, 88.0))

    def pan(self, dx_screen: float, dy_screen: float) -> None:
        """Deplace la cible dans le plan de l'ecran.

        Sans cela, la vue reste rivee au drone et une zone du nuage exploree
        plus tot devient inatteignable.
        """
        yaw = math.radians(self.yaw_deg)
        # Vecteurs droite et avant de la vue, projetes au sol.
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        forward = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
        scale = self.range_m / 400.0
        target = np.asarray(self.target, dtype=np.float64)
        target = target + right * (dx_screen * scale) + forward * (dy_screen * scale)
        self.target = (float(target[0]), float(target[1]), float(target[2]))

    def zoom(self, factor: float) -> None:
        self.range_m = float(np.clip(self.range_m * factor, 0.4, 200.0))

    def matrices(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Renvoie (rotation monde->vue, position de l'oeil, focale pixels)."""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        offset = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]) * self.range_m
        eye = np.asarray(self.target, dtype=np.float64) + offset

        forward = np.asarray(self.target, dtype=np.float64) - eye
        forward /= max(1e-9, np.linalg.norm(forward))
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.vstack([right, -up, forward])   # lignes: x ecran, y ecran, profondeur
        f = (height / 2.0) / math.tan(math.radians(self.fov_deg / 2.0))
        return R, eye, f


class Renderer3D:
    """Dessine le nuage, la trajectoire et le drone dans un tampon reutilise."""

    def __init__(self, size: Tuple[int, int] = config.VIEW3D_SIZE) -> None:
        self.width, self.height = size
        self.camera = OrbitCamera()
        self._canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.auto_follow = True
        self.spin_dps = 0.0
        self.show_surface = True
        self.colour_mode = MODE_REAL

    def resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width, self.height = max(80, width), max(60, height)
        self._canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)

    # -- projection --------------------------------------------------------

    def _project(self, pts: np.ndarray, R: np.ndarray, eye: np.ndarray, f: float,
                 clip_to_view: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projette des points monde. Renvoie (u, v, masque utilisable).

        `clip_to_view` sert aux points, qu'on ecrit directement dans le tampon
        et qui doivent donc tomber dedans. Pour les segments il faut le
        desactiver: une ligne dont une extremite sort du cadre reste visible
        entre les deux, et l'exiger dans le cadre effacait toute la grille.
        """
        if len(pts) == 0:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty, np.zeros(0, dtype=bool)
        cam = (pts - eye) @ R.T
        z = cam[:, 2]
        usable = z > 0.05
        u = np.zeros(len(pts))
        v = np.zeros(len(pts))
        np.divide(cam[:, 0] * f, z, out=u, where=usable)
        np.divide(cam[:, 1] * f, z, out=v, where=usable)
        u = u + self.width / 2.0
        v = v + self.height / 2.0
        if clip_to_view:
            usable &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        else:
            # cv2.line coupe tout seul, mais deraille sur des coordonnees
            # enormes: on borne largement autour du cadre.
            limit = 10 * max(self.width, self.height)
            np.clip(u, -limit, limit, out=u)
            np.clip(v, -limit, limit, out=v)
        return u.astype(np.int32), v.astype(np.int32), usable

    # -- rendu -------------------------------------------------------------

    def render(self, mapper, frame=None, dt: float = 0.0) -> np.ndarray:
        img = self._canvas
        img[:] = COLOR_BG

        if self.spin_dps and dt:
            self.camera.orbit(self.spin_dps * dt, 0.0)

        pos = np.asarray(frame.position, dtype=np.float64) if frame is not None else np.zeros(3)
        if self.auto_follow:
            # Suivre le drone en gardant le sol comme reference verticale.
            self.camera.target = (float(pos[0]), float(pos[1]), 0.0)

        R, eye, f = self.camera.matrices(self.width, self.height)

        self._draw_grid(img, R, eye, f, centre=pos)
        if self.show_surface and getattr(mapper, "grid", None) is not None:
            self._draw_surface(img, mapper.grid, R, eye, f)
        self._draw_cloud(img, mapper, R, eye, f)
        self._draw_trajectory(img, mapper, R, eye, f)
        if frame is not None:
            self._draw_drone(img, frame, R, eye, f)
        self._draw_legend(img, mapper, frame)
        return img

    def _draw_grid(self, img: np.ndarray, R: np.ndarray, eye: np.ndarray,
                   f: float, centre: np.ndarray, step: float = 1.0) -> None:
        half = max(4.0, min(20.0, self.camera.range_m))
        cx = math.floor(centre[0] / step) * step
        cy = math.floor(centre[1] / step) * step
        ticks = np.arange(-half, half + step, step)

        starts = []
        ends = []
        for t in ticks:
            starts.append([cx + t, cy - half, 0.0])
            ends.append([cx + t, cy + half, 0.0])
            starts.append([cx - half, cy + t, 0.0])
            ends.append([cx + half, cy + t, 0.0])
        if not starts:
            return

        s = np.asarray(starts)
        e = np.asarray(ends)
        us, vs, oks = self._project(s, R, eye, f, clip_to_view=False)
        ue, ve, oke = self._project(e, R, eye, f, clip_to_view=False)
        both = oks & oke
        for i in np.flatnonzero(both):
            main = (abs(s[i, 0] % 5.0) < 1e-6) or (abs(s[i, 1] % 5.0) < 1e-6)
            cv2.line(img, (us[i], vs[i]), (ue[i], ve[i]),
                     COLOR_GRID_MAIN if main else COLOR_GRID, 1, cv2.LINE_AA)

    def _draw_surface(self, img: np.ndarray, grid, R: np.ndarray, eye: np.ndarray,
                      f: float) -> None:
        """Dessine la carte d'elevation comme une surface, non comme des points.

        Chaque case est etalee sur autant de pixels qu'elle en couvre
        reellement -- une case de dix centimetres vue a huit metres en occupe
        environ quatre. Sans cet etalement la surface se lirait comme un semis
        troue des qu'on s'approche, et comme une bouillie des qu'on s'eloigne.

        L'ordre de dessin resout l'occultation sans tampon de profondeur. La
        taille d'une case a l'ecran ne depend que de sa distance: dessiner par
        taille croissante, c'est dessiner du plus lointain au plus proche. Le
        recouvrement se fait donc dans le bon sens, gratuitement, la ou un vrai
        tampon de profondeur couterait une comparaison par pixel.
        """
        centres, colours, shade = grid.surface(min_count=config.SURFACE_MIN_COUNT)
        if len(centres) == 0:
            return

        cam = (centres.astype(np.float64) - eye) @ R.T
        z = cam[:, 2]
        visible = z > 0.05
        if not visible.any():
            return

        u = np.zeros(len(centres))
        v = np.zeros(len(centres))
        np.divide(cam[:, 0] * f, z, out=u, where=visible)
        np.divide(cam[:, 1] * f, z, out=v, where=visible)
        ui = (u + self.width / 2.0).astype(np.int32)
        vi = (v + self.height / 2.0).astype(np.int32)
        visible &= (ui >= 0) & (ui < self.width) & (vi >= 0) & (vi < self.height)
        if not visible.any():
            return

        rgb = self._surface_colour(centres, colours, shade)

        # Rayon arrondi *par exces*. Deux cases voisines sont separees a
        # l'ecran de f*res/z pixels; un rayon tronque laisse alors une ligne
        # vide sur deux, et la surface se couvre d'un damier qui n'existe pas.
        # Arrondir au-dessus garantit que les etalements se touchent.
        tmp = np.zeros(len(centres))
        np.divide(f * grid.res, 2.0 * np.maximum(z, 1e-6), out=tmp, where=visible)
        np.clip(np.ceil(tmp), 0, config.SURFACE_MAX_SPLAT_PX, out=tmp)
        half = np.zeros(len(centres), dtype=np.int32)
        half[visible] = tmp[visible].astype(np.int32)

        for radius in np.unique(half[visible]):
            sel = np.flatnonzero(visible & (half == radius))
            if len(sel) == 0:
                continue
            colour = rgb[sel]
            su, sv = ui[sel], vi[sel]
            for dv in range(-int(radius), int(radius) + 1):
                rows = np.clip(sv + dv, 0, self.height - 1)
                for du in range(-int(radius), int(radius) + 1):
                    img[rows, np.clip(su + du, 0, self.width - 1)] = colour

    def _surface_colour(self, centres: np.ndarray, colours: np.ndarray,
                        shade: np.ndarray) -> np.ndarray:
        """Couleur finale des cases, selon le mode d'affichage choisi.

        Le mode n'est pas un ornement. La couleur reelle sert a reconnaitre le
        terrain, la hauteur a lire le relief la ou la texture est uniforme, et
        la fiabilite a savoir ce qu'on regarde -- une surface bien mesuree ou
        une extrapolation. Aucun de ces trois besoins ne se satisfait des deux
        autres.
        """
        if self.colour_mode == MODE_HEIGHT:
            # Echelle fixe, jamais ajustee sur le contenu. Une normalisation
            # entre le minimum et le maximum vus donnerait une couleur qui
            # change de signification a chaque image, et peindrait un sol plat
            # comme s'il etait accidente. Ici une couleur vaut toujours la meme
            # hauteur, et un terrain plat se voit comme plat.
            t = centres[:, 2] / config.SURFACE_HEIGHT_SPAN_M
            base = _ramp(t)
        elif self.colour_mode == MODE_PLAIN:
            base = np.tile(np.array(COLOR_GROUND_PT, dtype=np.float32), (len(centres), 1))
        else:
            base = colours
        out = base * shade[:, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _draw_cloud(self, img: np.ndarray, mapper, R: np.ndarray, eye: np.ndarray,
                    f: float) -> None:
        xyz, kind, _stamp, sigma, bgr = mapper.cloud.view_full()
        if len(xyz) == 0:
            return
        u, v, ok = self._project(xyz.astype(np.float64), R, eye, f)
        if not ok.any():
            return

        ground = ok & (kind == KIND_GROUND)
        obstacle = ok & (kind == KIND_OBSTACLE)

        # Le sol du nuage disparait quand la surface est affichee: elle dit la
        # meme chose en mieux, et les deux superposes se brouillent.
        if self.show_surface:
            ground[:] = False

        # Ecriture directe: une passe numpy par categorie, sans boucle Python.
        if ground.any():
            img[v[ground], u[ground]] = COLOR_GROUND_PT
        if obstacle.any():
            # Les obstacles comptent plus que le sol: on les epaissit pour
            # qu'ils restent lisibles quand le nuage de sol est dense.
            uu, vv = u[obstacle], v[obstacle]
            colour = self._obstacle_colour(sigma[obstacle], bgr[obstacle])
            img[vv, uu] = colour
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nu = np.clip(uu + du, 0, self.width - 1)
                nv = np.clip(vv + dv, 0, self.height - 1)
                img[nv, nu] = colour

    def _obstacle_colour(self, sigma: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        """Couleur des points de relief.

        En mode fiabilite, elle repond a une question que le nuage ne posait
        pas: ce point-la, le croit-on? Un obstacle reconstruit avec vingt
        centimetres d'incertitude et un autre avec deux se ressemblaient
        exactement a l'ecran, alors qu'on n'agit pas de la meme facon sur les
        deux.
        """
        if self.colour_mode == MODE_CONFIDENCE:
            t = np.clip(sigma / max(1e-6, config.STRUCTURE_MAX_SIGMA_M), 0.0, 1.0)
            return _ramp(1.0 - t).astype(np.uint8)
        if self.colour_mode == MODE_REAL:
            return bgr
        return np.array(COLOR_OBSTACLE_PT, dtype=np.uint8)

    def _draw_trajectory(self, img: np.ndarray, mapper, R: np.ndarray,
                         eye: np.ndarray, f: float) -> None:
        traj = mapper.trajectory
        if len(traj) < 2:
            return
        pts = np.asarray(traj[-config.TRAJECTORY_CAPACITY:], dtype=np.float64)
        u, v, ok = self._project(pts, R, eye, f, clip_to_view=False)
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            return
        poly = np.stack([u[idx], v[idx]], axis=1).astype(np.int32)
        cv2.polylines(img, [poly], False, COLOR_TRAJ, 1, cv2.LINE_AA)

    def _draw_drone(self, img: np.ndarray, frame, R: np.ndarray, eye: np.ndarray,
                    f: float) -> None:
        pos = np.asarray(frame.position, dtype=np.float64)
        yaw = math.radians(frame.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        nose = pos + np.array([c, s, 0.0]) * 0.6
        left = pos + np.array([-0.25 * c - 0.25 * -s, -0.25 * s - 0.25 * c, 0.0])
        right = pos + np.array([-0.25 * c + 0.25 * -s, -0.25 * s + 0.25 * c, 0.0])
        ground = np.array([pos[0], pos[1], 0.0])

        pts = np.vstack([pos, nose, left, right, ground])
        u, v, ok = self._project(pts, R, eye, f)
        if not ok[0]:
            return
        if ok[1]:
            cv2.line(img, (u[0], v[0]), (u[1], v[1]), COLOR_DRONE, 2, cv2.LINE_AA)
        if ok[2] and ok[3]:
            cv2.line(img, (u[2], v[2]), (u[3], v[3]), COLOR_DRONE, 1, cv2.LINE_AA)
        if ok[4]:
            # Verticale jusqu'au sol: sans elle, l'altitude est illisible.
            cv2.line(img, (u[0], v[0]), (u[4], v[4]), COLOR_DIM, 1, cv2.LINE_AA)
        cv2.circle(img, (u[0], v[0]), 3, COLOR_DRONE, -1, cv2.LINE_AA)

    def _draw_legend(self, img: np.ndarray, mapper, frame) -> None:
        cells = len(mapper.grid) if getattr(mapper, "grid", None) is not None else 0
        head = (f"surface {cells} cases  |  relief {len(mapper.cloud)} pts"
                if self.show_surface else f"nuage {len(mapper.cloud)} pts")
        cv2.putText(img, head, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"couleur: {self.colour_mode}", (6, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "sol", (6, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, COLOR_GROUND_PT, 1, cv2.LINE_AA)
        cv2.putText(img, "obstacle", (44, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, COLOR_OBSTACLE_PT, 1, cv2.LINE_AA)
        cv2.putText(img, "trajet", (118, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, COLOR_TRAJ, 1, cv2.LINE_AA)
        if frame is not None:
            txt = (f"vue {self.camera.range_m:.0f} m  "
                   f"cap {frame.yaw_deg:+.0f} deg  z {frame.position[2]:.2f} m")
            cv2.putText(img, txt, (6, self.height - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, COLOR_DIM, 1, cv2.LINE_AA)
