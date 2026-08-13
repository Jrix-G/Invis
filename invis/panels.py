"""Composition des quatre panneaux en une seule image.

Un seul tampon est envoye a l'interface plutot que quatre. Tkinter recree une
image a chaque affichage: une composition unique divise ce cout par quatre et
supprime tout risque de panneaux desynchronises entre eux.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .detector import STATE_CLEAR, STATE_NO_FLOW, STATE_OBSTACLE

COLOR_BG = (18, 18, 20)
COLOR_PANEL = (26, 26, 30)
COLOR_SEP = (60, 60, 68)
COLOR_TITLE = (185, 185, 195)
COLOR_TEXT = (225, 225, 230)
COLOR_DIM = (135, 135, 145)
COLOR_GOOD = (110, 210, 110)
COLOR_WARN = (60, 165, 245)
COLOR_ALERT = (70, 70, 245)
COLOR_RADAR = (70, 78, 90)
COLOR_POINT = (70, 90, 245)
COLOR_BAND = (95, 105, 120)

FONT = cv2.FONT_HERSHEY_SIMPLEX


_CANVAS_CACHE: dict = {}


def compose(camera_view: Optional[np.ndarray], measures: np.ndarray,
            view3d: np.ndarray, spare: np.ndarray,
            cell: Tuple[int, int]) -> np.ndarray:
    """Assemble les quatre panneaux dans une grille 2x2.

    Le tampon est conserve d'un appel a l'autre: reallouer et repeindre deux
    megaoctets a chaque image coutait plus cher que tout le reste du rendu.
    """
    cw, ch = cell
    shape = (ch * 2 + 3, cw * 2 + 3, 3)
    canvas = _CANVAS_CACHE.get(shape)
    if canvas is None:
        canvas = np.empty(shape, dtype=np.uint8)
        _CANVAS_CACHE.clear()
        _CANVAS_CACHE[shape] = canvas
        canvas[:] = COLOR_BG

    tiles = [
        (camera_view, 0, 0, "1. camera + champ de vecteurs"),
        (measures, 0, 1, "2. mesures en direct"),
        (view3d, 1, 0, "3. reconstruction 3D"),
        (spare, 1, 1, "4. libre"),
    ]
    for tile, row, col, title in tiles:
        y0 = row * (ch + 2) + 1
        x0 = col * (cw + 2) + 1
        block = canvas[y0:y0 + ch, x0:x0 + cw]
        if tile is None:
            block[:] = COLOR_PANEL
        else:
            fitted = _fit(tile, cw, ch - 16)
            fh, fw = fitted.shape[:2]
            oy = 16 + (ch - 16 - fh) // 2
            ox = (cw - fw) // 2
            # Ne repeindre que ce que la tuile ne couvre pas: bandeau de titre
            # et marges eventuelles.
            block[:oy] = COLOR_PANEL
            if oy + fh < ch:
                block[oy + fh:] = COLOR_PANEL
            if ox > 0:
                block[oy:oy + fh, :ox] = COLOR_PANEL
            if ox + fw < cw:
                block[oy:oy + fh, ox + fw:] = COLOR_PANEL
            block[oy:oy + fh, ox:ox + fw] = fitted
        cv2.putText(block, title, (6, 11), FONT, 0.36, COLOR_TITLE, 1, cv2.LINE_AA)

    cv2.line(canvas, (0, ch + 1), (canvas.shape[1], ch + 1), COLOR_SEP, 1)
    cv2.line(canvas, (cw + 1, 0), (cw + 1, canvas.shape[0]), COLOR_SEP, 1)
    return canvas


def _fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Redimensionne en preservant le rapport, sans jamais deformer."""
    h, w = img.shape[:2]
    if w <= 0 or h <= 0 or max_w <= 0 or max_h <= 0:
        return img
    scale = min(max_w / w, max_h / h)
    if abs(scale - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=interp)


# ---------------------------------------------------------------------------
# Panneau 2: mesures
# ---------------------------------------------------------------------------

def draw_measures(size: Tuple[int, int], result, mapframe,
                  link_fps: float = 0.0, analysis_fps: float = 0.0,
                  sigma_h: float = config.DEFAULT_SIGMA_H_M, gate=None) -> np.ndarray:
    """Chiffres et vue de dessus des points mesures."""
    w, h = size
    img = np.full((h, w, 3), COLOR_PANEL, dtype=np.uint8)
    split = int(w * 0.52)

    _draw_readouts(img[:, :split], result, mapframe, link_fps, analysis_fps, sigma_h, gate)
    _draw_radar(img[:, split:], mapframe)
    cv2.line(img, (split, 4), (split, h - 4), COLOR_SEP, 1)
    return img


def _state_colour(state: str):
    if state == STATE_OBSTACLE:
        return COLOR_ALERT
    if state == STATE_NO_FLOW:
        return COLOR_WARN
    if state == STATE_CLEAR:
        return COLOR_GOOD
    return COLOR_DIM


def _draw_readouts(panel: np.ndarray, result, mapframe, link_fps: float,
                   analysis_fps: float, sigma_h: float, gate=None) -> None:
    h, w = panel.shape[:2]
    y = 20

    state = result.state if result is not None else STATE_NO_FLOW
    cv2.putText(panel, state, (8, y), FONT, 0.6, _state_colour(state), 2, cv2.LINE_AA)
    y += 16
    if result is not None and result.reason:
        cv2.putText(panel, result.reason[:38], (8, y), FONT, 0.34, COLOR_DIM, 1, cv2.LINE_AA)
    y += 18

    # Distance: la mesure la plus tot disponible et la plus precise, cote a
    # cote. Elles ne viennent pas du meme calcul et ne se remplacent pas.
    contact = getattr(mapframe, "contact", None) if mapframe else None
    nearest = getattr(mapframe, "nearest", None) if mapframe else None

    y = _distance_block(panel, y, "contact sol", contact, COLOR_WARN)
    y = _distance_block(panel, y, "triangulation", nearest, COLOR_ALERT)

    ttc = result.global_ttc if result is not None else None
    rows = [
        ("temps avant contact", f"{ttc:.2f} s" if ttc else "--"),
        ("vitesse estimee", f"{mapframe.speed_mps:.2f} m/s" if mapframe else "--"),
        ("hauteur retenue", f"{mapframe.height_m:.2f} +/- {sigma_h:.2f} m" if mapframe else "--"),
        ("assiette mesuree", f"{mapframe.tilt_deg:+.1f} / {mapframe.roll_deg:+.1f} deg"
         if mapframe else "--"),
        ("portee du champ", f"{mapframe.coverage[0]:.1f} - {mapframe.coverage[1]:.1f} m"
         if mapframe else "--"),
        ("points suivis", str(result.n_tracked) if result is not None else "--"),
        ("dont hors sol", str(int(result.off_plane.sum()))
         if result is not None and result.off_plane is not None else "0"),
        ("nuage 3D", f"{mapframe.n_cloud} pts" if mapframe else "0"),
        ("cadence reseau", f"{link_fps:.1f} img/s"),
        ("cadence analyse", f"{analysis_fps:.1f} img/s"),
        ("images ecartees", f"{gate.rejected}/{gate.total}" if gate and gate.total else "0"),
    ]
    for label, value in rows:
        if y > h - 8:
            break
        cv2.putText(panel, label, (8, y), FONT, 0.34, COLOR_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, value, (w - 8 - _text_width(value, 0.36), y), FONT, 0.36,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        y += 14


def _distance_block(panel: np.ndarray, y: int, label: str, measure,
                    colour) -> int:
    w = panel.shape[1]
    cv2.putText(panel, label, (8, y), FONT, 0.34, COLOR_DIM, 1, cv2.LINE_AA)
    if measure is None:
        cv2.putText(panel, "--", (w - 30, y), FONT, 0.42, COLOR_DIM, 1, cv2.LINE_AA)
        return y + 20

    value = f"{measure.range_m:.2f} m"
    cv2.putText(panel, value, (w - 8 - _text_width(value, 0.52), y), FONT, 0.52,
                colour, 2, cv2.LINE_AA)
    y += 13
    # L'intervalle vient entierement de l'incertitude sur la hauteur: l'afficher
    # evite de faire passer une echelle mal connue pour une precision.
    band = f"[{measure.band[0]:.2f} - {measure.band[1]:.2f}]"
    cv2.putText(panel, band, (w - 8 - _text_width(band, 0.32), y), FONT, 0.32,
                COLOR_BAND, 1, cv2.LINE_AA)
    return y + 14


def _text_width(text: str, scale: float) -> int:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    return tw


def _draw_radar(panel: np.ndarray, mapframe) -> None:
    """Vue de dessus: le drone en bas, ce qu'il mesure devant lui."""
    h, w = panel.shape[:2]
    cv2.putText(panel, "vue de dessus", (8, 14), FONT, 0.34, COLOR_DIM, 1, cv2.LINE_AA)

    origin = (w // 2, h - 14)
    span_m = config.RADAR_SPAN_M
    scale = (h - 34) / span_m

    for ring in range(1, int(span_m) + 1):
        r = int(ring * scale)
        cv2.circle(panel, origin, r, COLOR_RADAR, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{ring}", (origin[0] + 3, origin[1] - r + 10), FONT,
                    0.28, COLOR_RADAR, 1, cv2.LINE_AA)
    cv2.line(panel, origin, (origin[0], 20), COLOR_RADAR, 1)

    if mapframe is None:
        return

    def plot(points, colour, radius):
        if points is None or len(points) == 0:
            return
        pts = np.asarray(points, dtype=np.float64)
        px = origin[0] - (pts[:, 1] * scale)      # Y drone = gauche
        py = origin[1] - (pts[:, 0] * scale)      # X drone = avant
        inside = (px >= 0) & (px < w) & (py >= 20) & (py < h)
        for x, y in zip(px[inside].astype(int), py[inside].astype(int)):
            cv2.circle(panel, (x, y), radius, colour, -1, cv2.LINE_AA)

    plot(mapframe.ground_body, (90, 82, 62), 1)
    plot(mapframe.obstacle_body, COLOR_POINT, 2)

    measure = mapframe.nearest or mapframe.contact
    if measure is not None:
        y = origin[1] - int(measure.range_m * scale)
        if 20 <= y < h:
            cv2.line(panel, (8, y), (w - 8, y), COLOR_ALERT, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{measure.range_m:.2f} m", (10, max(28, y - 4)), FONT,
                        0.34, COLOR_ALERT, 1, cv2.LINE_AA)

    cv2.circle(panel, origin, 3, COLOR_TEXT, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Panneau 4: reserve
# ---------------------------------------------------------------------------

def draw_spare(size: Tuple[int, int], lines=None) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), COLOR_PANEL, dtype=np.uint8)
    text = lines or ["emplacement libre"]
    y = h // 2 - 6 * len(text)
    for line in text:
        cv2.putText(img, line, ((w - _text_width(line, 0.38)) // 2, y), FONT, 0.38,
                    COLOR_DIM, 1, cv2.LINE_AA)
        y += 16
    return img
