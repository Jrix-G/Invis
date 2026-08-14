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
COLOR_TRACK = (120, 220, 120)

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
        (spare, 1, 1, "4. carte du vol"),
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

    # La distance droit devant vient en premier: c'est la seule qui reponde a
    # la question du pilote sans qu'il ait a interpreter. Les deux suivantes
    # disent d'ou elle vient.
    y = _forward_block(panel, y, mapframe)
    y = _distance_block(panel, y, "contact sol", contact, COLOR_WARN)
    y = _distance_block(panel, y, "triangulation", nearest, COLOR_ALERT)

    ttc = result.global_ttc if result is not None else None
    clusters = getattr(mapframe, "clusters", []) if mapframe else []
    rows = [
        ("objets distincts", str(len(clusters)) if mapframe else "--"),
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


def _forward_block(panel: np.ndarray, y: int, mapframe) -> int:
    """Distance droit devant, et couleur du niveau d'alerte.

    Une valeur toujours presente, jamais "--": faute d'obstacle, c'est le sol
    qui est vise, et sa distance est connue exactement. Un tiret laisserait
    croire que la mesure a echoue alors que la voie est simplement libre.
    """
    w = panel.shape[1]
    forward = getattr(mapframe, "forward_m", float("nan")) if mapframe else float("nan")
    level = getattr(mapframe, "alert_level", 0) if mapframe else 0
    colour = COLOR_ALERT if level >= 2 else (COLOR_WARN if level == 1 else COLOR_GOOD)

    cv2.putText(panel, "droit devant", (8, y), FONT, 0.34, COLOR_DIM, 1, cv2.LINE_AA)
    value = f"{forward:.2f} m" if math.isfinite(forward) else "--"
    cv2.putText(panel, value, (w - 8 - _text_width(value, 0.62), y), FONT, 0.62,
                colour, 2, cv2.LINE_AA)
    y += 14
    if mapframe is not None:
        kind = ("obstacle" if getattr(mapframe, "forward_is_obstacle", False)
                else "voie libre, portee du champ")
        note = mapframe.alert_reason or kind
        cv2.putText(panel, note[:30], (w - 8 - _text_width(note[:30], 0.32), y), FONT,
                    0.32, colour if level else COLOR_DIM, 1, cv2.LINE_AA)
    return y + 16


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

    # Couloir de vol: sans lui, "l'obstacle est a gauche" ne dit pas s'il
    # genera. Le tracer transforme une position en decision.
    corridor = int(config.OBSTACLE_CORRIDOR_M * scale)
    cv2.line(panel, (origin[0] - corridor, origin[1]), (origin[0] - corridor, 20),
             COLOR_RADAR, 1, cv2.LINE_AA)
    cv2.line(panel, (origin[0] + corridor, origin[1]), (origin[0] + corridor, 20),
             COLOR_RADAR, 1, cv2.LINE_AA)

    plot(mapframe.ground_body, (90, 82, 62), 1)
    plot(mapframe.obstacle_body, COLOR_POINT, 2)

    # Un rectangle par objet distinct, plutot qu'une ligne unique en travers
    # de tout le radar: deux objets separes se lisent comme deux objets.
    for rank, cluster in enumerate(getattr(mapframe, "clusters", [])):
        cx = origin[0] - int(cluster.lateral_m * scale)
        cy = origin[1] - int(cluster.forward_m * scale)
        if not (0 <= cx < w and 20 <= cy < h):
            continue
        half = max(2, int(cluster.width_m * scale / 2))
        colour = COLOR_ALERT if cluster.in_corridor else COLOR_BAND
        cv2.rectangle(panel, (cx - half, cy - 3), (cx + half, cy + 3), colour, 1)
        if rank < 2:
            cv2.putText(panel, f"{cluster.range_m:.1f}", (cx + half + 3, cy + 3),
                        FONT, 0.30, colour, 1, cv2.LINE_AA)

    cv2.circle(panel, origin, 3, COLOR_TEXT, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Panneau 4: carte du vol
# ---------------------------------------------------------------------------

def draw_map(size: Tuple[int, int], mapper, mapframe) -> np.ndarray:
    """Carte de dessus du terrain parcouru, en coordonnees monde.

    La vue de dessus du panneau 2 est centree sur le drone et ne montre que
    l'instant: elle repond a "qu'y a-t-il devant moi maintenant". Elle ne dit
    rien de ce qui a ete explore, ni de la forme du trajet, ni si l'on
    s'apprete a repasser au meme endroit.

    Cette carte-ci est fixe dans le monde. C'est la seule vue ou une derive
    d'odometrie se *voit*: un aller-retour qui ne se referme pas y saute aux
    yeux, alors qu'aucun chiffre ne l'aurait signale.

    Le fond est la carte d'elevation elle-meme, lue directement comme une
    image -- elle est deja une grille reguliere, il n'y a donc rien a
    reprojeter, seulement a recadrer.
    """
    w, h = size
    img = np.full((h, w, 3), COLOR_PANEL, dtype=np.uint8)
    grid = getattr(mapper, "grid", None)
    cv2.putText(img, "carte du vol (repere monde)", (8, 14), FONT, 0.34,
                COLOR_DIM, 1, cv2.LINE_AA)
    if grid is None or len(grid) == 0:
        return draw_spare(size, ["carte du vol",
                                 "le terrain apparait a mesure du survol"])

    view = _map_extent(mapper, mapframe, w, h)
    if view is None:
        return img
    x0, y0, span = view
    scale = min(w, h - 20) / span

    def to_px(xy: np.ndarray) -> np.ndarray:
        px = (xy[:, 0] - x0) * scale
        # L'axe Y du monde monte vers la gauche; a l'ecran il descend. Le
        # renversement se fait ici, une seule fois.
        py = (h - 6) - (xy[:, 1] - y0) * scale
        return np.column_stack([px, py])

    _blit_terrain(img, grid, x0, y0, span, scale, w, h)
    _draw_track(img, mapper, to_px, w, h)
    _draw_map_obstacles(img, mapper, mapframe, to_px, w, h)
    _draw_map_drone(img, mapframe, to_px, w, h)

    cv2.putText(img, f"{span:.0f} m de cote  |  {len(grid)} cases",
                (8, h - 6), FONT, 0.32, COLOR_DIM, 1, cv2.LINE_AA)
    return img


def _map_extent(mapper, mapframe, w: int, h: int):
    """Cadre a afficher: tout le terrain connu, avec une marge."""
    grid = mapper.grid
    filled = np.flatnonzero(grid.count > 0)
    if len(filled) == 0:
        return None
    cx = filled % grid.cells
    cy = filled // grid.cells
    lo = np.array([grid.origin[0] + cx.min() * grid.res,
                   grid.origin[1] + cy.min() * grid.res])
    hi = np.array([grid.origin[0] + (cx.max() + 1) * grid.res,
                   grid.origin[1] + (cy.max() + 1) * grid.res])
    centre = (lo + hi) / 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.15
    span = max(span, config.MAP_MIN_SPAN_M)
    return centre[0] - span / 2.0, centre[1] - span / 2.0, span


def _blit_terrain(img, grid, x0, y0, span, scale, w, h) -> None:
    """Recadre la carte d'elevation dans le panneau, sans reprojection."""
    n = grid.cells
    i0 = int(np.floor((x0 - grid.origin[0]) / grid.res))
    j0 = int(np.floor((y0 - grid.origin[1]) / grid.res))
    side = int(np.ceil(span / grid.res))
    i1, j1 = min(n, i0 + side), min(n, j0 + side)
    i0, j0 = max(0, i0), max(0, j0)
    if i1 <= i0 or j1 <= j0:
        return

    colour = grid.colour.reshape(n, n, 3)[j0:j1, i0:i1]
    lit = grid.shading().reshape(n, n)[j0:j1, i0:i1]
    seen = grid.count.reshape(n, n)[j0:j1, i0:i1] > 0

    tile = np.clip(colour * lit[:, :, None], 0, 255).astype(np.uint8)
    tile[~seen] = COLOR_PANEL
    # L'axe Y monte vers le haut dans le monde, vers le bas a l'ecran.
    tile = np.flipud(tile)

    out_w = max(1, int((i1 - i0) * grid.res * scale))
    out_h = max(1, int((j1 - j0) * grid.res * scale))
    tile = cv2.resize(tile, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    px = int((grid.origin[0] + i0 * grid.res - x0) * scale)
    py = (h - 6) - int((grid.origin[1] + j1 * grid.res - y0) * scale)
    sx0, sy0 = max(0, px), max(0, py)
    sx1, sy1 = min(w, px + out_w), min(h, py + out_h)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    img[sy0:sy1, sx0:sx1] = tile[sy0 - py:sy1 - py, sx0 - px:sx1 - px]


def _draw_track(img, mapper, to_px, w, h) -> None:
    if len(mapper.trajectory) < 2:
        return
    pts = to_px(np.asarray(mapper.trajectory, dtype=np.float64)[:, :2])
    poly = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], False, COLOR_TRACK, 1, cv2.LINE_AA)


def _draw_map_obstacles(img, mapper, mapframe, to_px, w, h) -> None:
    """Reliefs releves, en coordonnees monde, les recents mis en avant."""
    xyz, kind, stamp, _sigma, _bgr = mapper.cloud.view_full()
    sel = kind == 1
    if not sel.any():
        return
    pts = to_px(xyz[sel][:, :2].astype(np.float64))
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] < w)
              & (pts[:, 1] >= 20) & (pts[:, 1] < h))
    if not inside.any():
        return
    px = pts[inside].astype(np.int32)
    img[px[:, 1], px[:, 0]] = COLOR_POINT


def _draw_map_drone(img, mapframe, to_px, w, h) -> None:
    if mapframe is None:
        return
    pos = np.asarray(mapframe.position, dtype=np.float64)[None, :2]
    p = to_px(pos)[0]
    if not (0 <= p[0] < w and 0 <= p[1] < h):
        return
    yaw = math.radians(mapframe.yaw_deg)
    nose = to_px(pos + np.array([[math.cos(yaw), math.sin(yaw)]]) * 0.8)[0]
    cv2.line(img, (int(p[0]), int(p[1])), (int(nose[0]), int(nose[1])),
             COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.circle(img, (int(p[0]), int(p[1])), 3, COLOR_TEXT, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Panneau de secours
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
