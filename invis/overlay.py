"""Rendu du calque de debogage sur l'image affichee."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config, geometry
from .detector import STATE_CLEAR, STATE_NO_FLOW, STATE_OBSTACLE, DetectionResult
from .geometry import Intrinsics

COLOR_INLIER = (90, 200, 90)
COLOR_OUTLIER = (60, 60, 235)
COLOR_GRID = (70, 70, 70)
COLOR_HIT = (0, 165, 255)
COLOR_CONFIRMED = (0, 0, 255)
COLOR_HORIZON = (200, 200, 60)
COLOR_TEXT = (245, 245, 245)
COLOR_RANGE = (185, 165, 90)
COLOR_CONTACT = (60, 200, 255)


def draw(frame: np.ndarray, result: DetectionResult, show_flow: bool = True,
         mapframe=None, show_ranges: bool = True) -> np.ndarray:
    """Retourne une copie annotee de l'image."""
    out = frame.copy()
    h, w = out.shape[:2]
    s = result.scale

    _draw_grid(out, w, h)

    if show_ranges and mapframe is not None:
        _draw_range_ticks(out, mapframe, w, h)

    if result.horizon_row is not None:
        y = int(result.horizon_row)
        cv2.line(out, (0, y), (w, y), COLOR_HORIZON, 1)
        cv2.putText(out, "horizon", (4, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, COLOR_HORIZON, 1, cv2.LINE_AA)

    if show_flow:
        # Gain adaptatif. Un facteur fixe marchait a basse vitesse et noyait
        # l'image des que le deplacement grandissait: a 15 px de flux, des
        # vecteurs multiplies par quatre couvrent tout. On vise une longueur
        # lisible constante, quelle que soit la vitesse.
        median_flow = max(0.2, result.median_flow_px)
        gain = float(np.clip(config.FLOW_TARGET_PX / median_flow, 1.0, 6.0))
        for (p0, p1, is_outlier) in result.flow:
            a = (int(p0[0] * s), int(p0[1] * s))
            b = (int(p1[0] * s), int(p1[1] * s))
            color = COLOR_OUTLIER if is_outlier else COLOR_INLIER
            tip = (int(a[0] + (b[0] - a[0]) * gain), int(a[1] + (b[1] - a[1]) * gain))
            cv2.line(out, a, tip, color, 1, cv2.LINE_AA)
            cv2.circle(out, a, 1, color, -1)

    _draw_cells(out, result, w, h)
    if mapframe is not None:
        _draw_contact(out, mapframe, w, h)
    _draw_hud(out, result, w, h, mapframe)
    return out


def _draw_range_ticks(img: np.ndarray, mapframe, w: int, h: int) -> None:
    """Reperes de distance au sol, traces directement sur l'image."""
    K = Intrinsics.from_fov(w, h)
    for range_m in config.RANGE_TICKS_M:
        row = geometry.row_for_range(K, range_m, mapframe.height_m,
                                     tilt_deg=mapframe.tilt_deg)
        if row is None:
            continue
        y = int(row)
        cv2.line(img, (0, y), (18, y), COLOR_RANGE, 1, cv2.LINE_AA)
        cv2.line(img, (w - 18, y), (w, y), COLOR_RANGE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{range_m:g}m", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, COLOR_RANGE, 1, cv2.LINE_AA)


def _draw_contact(img: np.ndarray, mapframe, w: int, h: int) -> None:
    """Ligne de contact estimee et points hors sol qui l'ont produite."""
    if mapframe.contact_uv is not None:
        for (x, y) in np.asarray(mapframe.contact_uv, dtype=np.int32):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (int(x), int(y)), 2, COLOR_CONTACT, -1, cv2.LINE_AA)

    if mapframe.contact_row is None or mapframe.contact is None:
        return
    y = int(mapframe.contact_row)
    if not (0 <= y < h):
        return
    cv2.line(img, (0, y), (w, y), COLOR_CONTACT, 1, cv2.LINE_AA)
    label = f"contact {mapframe.contact.range_m:.2f} m"
    cv2.putText(img, label, (w - 118, max(11, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, COLOR_CONTACT, 1, cv2.LINE_AA)


def _draw_grid(img: np.ndarray, w: int, h: int) -> None:
    for c in range(1, config.GRID_COLS):
        x = int(w * c / config.GRID_COLS)
        cv2.line(img, (x, 0), (x, h), COLOR_GRID, 1)
    for r in range(1, config.GRID_ROWS):
        y = int(h * r / config.GRID_ROWS)
        cv2.line(img, (0, y), (w, y), COLOR_GRID, 1)


def _draw_cells(img: np.ndarray, result: DetectionResult, w: int, h: int) -> None:
    cw = w / config.GRID_COLS
    ch = h / config.GRID_ROWS
    for cell in result.cells:
        if not (cell.raw_hit or cell.confirmed):
            continue
        x0, y0 = int(cell.col * cw), int(cell.row * ch)
        x1, y1 = int((cell.col + 1) * cw), int((cell.row + 1) * ch)
        color = COLOR_CONFIRMED if cell.confirmed else COLOR_HIT
        thickness = 2 if cell.confirmed else 1
        cv2.rectangle(img, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), color, thickness)
        label = cell.name
        if cell.ttc is not None:
            label += f" {cell.ttc:.1f}s"
        cv2.putText(img, label, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1, cv2.LINE_AA)


def _draw_hud(img: np.ndarray, result: DetectionResult, w: int, h: int,
              mapframe=None) -> None:
    if result.state == STATE_OBSTACLE:
        banner, color = f"OBSTACLE  {result.reason}", COLOR_CONFIRMED
    elif result.state == STATE_NO_FLOW:
        banner, color = "PAS DE FLUX  (scene immobile)", COLOR_HORIZON
    elif result.state == STATE_CLEAR:
        banner, color = "LIBRE", COLOR_INLIER
    else:
        banner, color = result.reason or "...", COLOR_TEXT

    cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(img, banner, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    ttc = f"{result.global_ttc:.2f}s" if result.global_ttc else "--"
    line = (f"pts {result.n_tracked}  flux {result.median_flow_px:.2f}px  "
            f"ttc {ttc}  plan {'oui' if result.plane_found else 'non'}")
    if mapframe is not None:
        line += f"  h {mapframe.height_m:.1f}m  tilt {mapframe.tilt_deg:+.0f}" 
    cv2.rectangle(img, (0, h - 16), (w, h), (0, 0, 0), -1)
    cv2.putText(img, line, (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)


def to_ppm(bgr: np.ndarray) -> Optional[bytes]:
    """Encode en PPM binaire, format que Tk sait afficher sans dependance."""
    ok, buf = cv2.imencode(".ppm", bgr)
    if not ok:
        return None
    return buf.tobytes()
