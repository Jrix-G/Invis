"""Detection d'obstacle monoculaire a partir du flux de la camera.

Principe
--------
Sans telemetre, une seule camera ne donne pas de distance. Elle donne deux
indices exploitables des que l'appareil bouge:

1. *Plan dominant*. La camera est piquee de 45 degres vers le sol: l'essentiel
   de l'image est une surface, donc le deplacement apparent des points du sol
   entre deux images se resume a une homographie. Ce qui ne suit pas cette
   homographie ne fait pas partie du plan: c'est du relief, donc un obstacle
   candidat.

2. *Expansion*. Un objet vers lequel on se dirige grandit dans l'image. Le
   rapport d'echelle entre deux images donne un temps avant collision, sans
   jamais donner de distance en metres.

Les deux indices sont calcules par cellule d'une grille 3x3, puis passes par
une hysteresis. Aucun resultat n'est transmis au controleur de vol: ce module
observe, il ne commande rien.

Limite assumee: en vol stationnaire, il n'y a pas de flux, donc pas de
profondeur. L'etat renvoye est alors NO_FLOW et la detection est inactive.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config, geometry
from .geometry import Intrinsics

STATE_NO_FLOW = "NO_FLOW"
STATE_CLEAR = "CLEAR"
STATE_OBSTACLE = "OBSTACLE"


@dataclass
class CellReport:
    row: int
    col: int
    name: str
    n_points: int = 0
    outlier_ratio: float = 0.0
    ttc: Optional[float] = None
    raw_hit: bool = False
    confirmed: bool = False
    score: float = 0.0


@dataclass
class DetectionResult:
    state: str = STATE_NO_FLOW
    reason: str = ""
    cells: List[CellReport] = field(default_factory=list)
    global_ttc: Optional[float] = None
    median_flow_px: float = 0.0
    n_tracked: int = 0
    plane_found: bool = False
    horizon_row: Optional[float] = None
    dt: float = 0.0
    timestamp: float = 0.0
    # Pour l'affichage: couples (p0, p1) a la resolution de travail et drapeau
    # "hors plan dominant".
    flow: List[Tuple[Tuple[float, float], Tuple[float, float], bool]] = field(default_factory=list)
    scale: float = 1.0  # facteur resolution travail -> resolution affichage

    # Donnees brutes a la resolution de travail, consommees par la
    # reconstruction 3D. Tableaux alignes entre eux.
    pts_prev: Optional[np.ndarray] = None      # (N, 2) image precedente
    pts_cur: Optional[np.ndarray] = None       # (N, 2) image courante
    track_ids: Optional[np.ndarray] = None     # (N,) identifiants persistants
    residuals: Optional[np.ndarray] = None     # (N,) ecart au plan dominant
    off_plane: Optional[np.ndarray] = None     # (N,) booleen "hors sol"
    homography: Optional[np.ndarray] = None    # plan dominant, pixels travail
    work_size: Tuple[int, int] = (0, 0)        # (largeur, hauteur) de travail
    residual_threshold: float = 0.0
    plane_inlier_ratio: float = 0.0

    @property
    def worst_cell(self) -> Optional[CellReport]:
        hits = [c for c in self.cells if c.confirmed]
        if not hits:
            return None
        return max(hits, key=lambda c: c.score)


class _Hysteresis:
    """Confirme une cellule apres plusieurs detections, la libere apres plusieurs vides.

    Le bruit de compression JPEG produit des points aberrants isoles. Exiger
    une repetition coute quelques images de latence et supprime l'essentiel
    des fausses alarmes.
    """

    def __init__(self) -> None:
        self._history: Dict[Tuple[int, int], deque] = {}
        self._latched: Dict[Tuple[int, int], bool] = {}

    def update(self, key: Tuple[int, int], hit: bool) -> bool:
        hist = self._history.setdefault(key, deque(maxlen=max(config.CONFIRM_WINDOW, config.RELEASE_MISSES)))
        hist.append(bool(hit))
        latched = self._latched.get(key, False)

        recent = list(hist)[-config.CONFIRM_WINDOW:]
        if not latched and sum(recent) >= config.CONFIRM_HITS:
            latched = True
        elif latched:
            tail = list(hist)[-config.RELEASE_MISSES:]
            if len(tail) >= config.RELEASE_MISSES and not any(tail):
                latched = False

        self._latched[key] = latched
        return latched

    def reset(self) -> None:
        self._history.clear()
        self._latched.clear()


def horizon_row(frame_h: int, tilt_deg: float = config.CAMERA_TILT_DEG,
                vfov_deg: float = config.VFOV_DEG) -> Optional[float]:
    """Ligne d'horizon en pixels, ou None si elle tombe hors de l'image.

    Avec un piquage de 45 degres et un champ vertical de ~41 degres, l'horizon
    sort par le haut: toute l'image regarde vers le sol. Le savoir change la
    strategie -- il n'y a pas de "ciel" a ignorer, le plan dominant est le sol.
    """
    theta = math.radians(-tilt_deg)  # angle de l'axe optique sous l'horizontale
    f_px = (frame_h / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    row = (frame_h / 2.0) - f_px * math.tan(theta)
    if row < 0 or row > frame_h:
        return None
    return row


class ObstacleDetector:
    def __init__(self, work_width: int = config.WORK_WIDTH) -> None:
        self.work_width = work_width
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._since_detect = 0
        # Identifiants persistants: un meme point garde son numero tant qu'il
        # est suivi. Sans cela, impossible de retrouver le meme point dans une
        # vue anterieure, donc impossible de trianguler.
        self._prev_ids: Optional[np.ndarray] = None
        self._next_id = 0
        self._K: Optional[Intrinsics] = None
        self._hyst = _Hysteresis()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._lk = dict(
            winSize=(config.LK_WINDOW, config.LK_WINDOW),
            maxLevel=config.LK_LEVELS,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.sensitivity = 1.0  # >1 = plus sensible, ajuste depuis l'interface

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None
        self._prev_time = None
        self._since_detect = 0
        self._prev_ids = None
        self._hyst.reset()

    # -- pretraitement -----------------------------------------------------

    def _prepare(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        scale = self.work_width / float(w)
        small = cv2.resize(bgr, (self.work_width, max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # La compression a q12 laisse des blocs 8x8. Egaliser puis lisser
        # legerement evite que le suiveur accroche les bords de bloc.
        gray = self._clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray, 1.0 / scale

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=config.MAX_FEATURES,
            qualityLevel=config.FEATURE_QUALITY,
            minDistance=config.FEATURE_MIN_DISTANCE,
            blockSize=7,
        )
        return pts

    # -- coeur -------------------------------------------------------------

    def _seed(self, gray: np.ndarray) -> None:
        """Repart de zero sur cette image: nouveaux points, nouveaux numeros."""
        pts = self._detect_features(gray)
        if pts is None:
            pts = np.zeros((0, 1, 2), dtype=np.float32)
        self._prev_pts = pts
        self._prev_ids = self._mint_ids(len(pts))
        self._since_detect = 0

    def _mint_ids(self, count: int) -> np.ndarray:
        ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)
        self._next_id += count
        return ids

    def process(self, bgr: np.ndarray, timestamp: float) -> DetectionResult:
        gray, upscale = self._prepare(bgr)
        h, w = gray.shape[:2]
        result = DetectionResult(scale=upscale)
        result.horizon_row = horizon_row(bgr.shape[0])
        result.timestamp = timestamp
        result.work_size = (w, h)
        if self._K is None or self._K.width != w or self._K.height != h:
            self._K = Intrinsics.from_fov(w, h)

        prev_gray, prev_pts, prev_time = self._prev_gray, self._prev_pts, self._prev_time
        prev_ids = self._prev_ids
        self._prev_gray = gray
        self._prev_time = timestamp

        if (prev_gray is None or prev_pts is None or prev_ids is None
                or len(prev_pts) < config.MIN_POINTS_FOR_MODEL):
            self._seed(gray)
            result.reason = "amorcage du suivi"
            return result

        dt = max(1e-3, timestamp - (prev_time or timestamp))
        result.dt = dt

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **self._lk)
        if p1 is None:
            self._seed(gray)
            result.reason = "suivi perdu"
            return result

        # Controle aller-retour: un point qui ne revient pas a sa place est un
        # faux appariement, frequent sur les zones uniformes du sol.
        p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self._lk)
        fb_err = np.linalg.norm(prev_pts.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
        good = (st.reshape(-1) == 1) & (st_b.reshape(-1) == 1) & (fb_err < config.FB_ERROR_MAX)

        a = prev_pts.reshape(-1, 2)[good]
        b = p1.reshape(-1, 2)[good]
        ids = prev_ids[good]
        result.n_tracked = int(len(a))

        if len(a) < config.MIN_POINTS_FOR_MODEL:
            self._seed(gray)
            result.reason = f"trop peu de points suivis ({len(a)})"
            return result

        flow_mag = np.linalg.norm(b - a, axis=1)
        result.median_flow_px = float(np.median(flow_mag))

        # Scene immobile: aucune information de profondeur disponible.
        if result.median_flow_px < config.STATIC_FLOW_PX:
            result.state = STATE_NO_FLOW
            result.reason = "scene immobile, pas de parallaxe exploitable"
            self._prev_pts, self._prev_ids = self._refresh(gray, b, ids)
            for r in range(config.GRID_ROWS):
                for c in range(config.GRID_COLS):
                    self._hyst.update((r, c), False)
            result.cells = self._empty_cells()
            return result

        # 1) Plan dominant (le sol) par homographie robuste.
        residuals, plane_ok, homography, inlier_ratio, plane_inliers = self._plane_residuals(a, b)
        result.plane_found = plane_ok
        result.homography = homography
        result.plane_inlier_ratio = inlier_ratio

        # 2) Expansion globale, pour un temps avant collision de reference.
        result.global_ttc = self._ttc_from_expansion(a, b, dt)

        # 3) Notation par cellule.
        thr = self._residual_threshold(residuals, plane_inliers, result.median_flow_px)
        result.cells = self._score_cells(a, b, residuals, dt, w, h, thr)

        confirmed = [c for c in result.cells if c.confirmed]
        if confirmed:
            result.state = STATE_OBSTACLE
            worst = max(confirmed, key=lambda c: c.score)
            result.reason = f"cellule {worst.name}"
        else:
            result.state = STATE_CLEAR
            result.reason = "aucun relief confirme"

        off_plane = residuals > thr
        result.residual_threshold = float(thr)
        result.pts_prev = a
        result.pts_cur = b
        result.track_ids = ids
        result.residuals = residuals
        result.off_plane = off_plane
        result.flow = [
            ((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])), bool(flag))
            for pa, pb, flag in zip(a, b, off_plane)
        ]

        self._prev_pts, self._prev_ids = self._refresh(gray, b, ids)
        return result

    # -- sous-etapes -------------------------------------------------------

    def _empty_cells(self) -> List[CellReport]:
        return [
            CellReport(row=r, col=c, name=config.CELL_NAMES[r][c])
            for r in range(config.GRID_ROWS)
            for c in range(config.GRID_COLS)
        ]

    def _refresh(self, gray: np.ndarray, tracked: np.ndarray,
                 ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reprend les points suivis et en ajoute de nouveaux regulierement.

        Se contenter des points survivants condamne le detecteur: un obstacle
        qui entre dans le champ n'aurait jamais de point a lui, et le suivi
        s'eteindrait image apres image. On recomplete donc periodiquement, en
        masquant les zones deja couvertes pour ne pas empiler des doublons.

        Les points repris gardent leur identifiant; seuls les nouveaux en
        recoivent un. C'est ce qui permet a la reconstruction de retrouver le
        meme point dans une vue prise plus tot.
        """
        pts = tracked.reshape(-1, 1, 2).astype(np.float32)
        self._since_detect += 1

        need = len(pts) < config.MIN_FEATURES or self._since_detect >= config.REDETECT_EVERY
        if not need:
            return pts, ids

        self._since_detect = 0
        room = config.MAX_FEATURES - len(pts)
        if room <= 0:
            return pts, ids

        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
        for (x, y) in pts.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), config.FEATURE_MIN_DISTANCE, 0, -1)

        fresh = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(room),
            qualityLevel=config.FEATURE_QUALITY,
            minDistance=config.FEATURE_MIN_DISTANCE,
            blockSize=7,
            mask=mask,
        )
        if fresh is None or len(fresh) == 0:
            return pts, ids
        merged = np.vstack([pts, fresh.astype(np.float32)])
        return merged, np.concatenate([ids, self._mint_ids(len(fresh))])

    def _plane_residuals(self, a: np.ndarray,
                         b: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray],
                                                 float, Optional[np.ndarray]]:
        """Residu de chaque point par rapport au plan dominant, en pixels.

        Renvoie aussi l'homographie: la reconstruction 3D en tire le
        deplacement de la camera et l'orientation reelle du sol.
        """
        src = a.reshape(-1, 1, 2).astype(np.float32)
        dst = b.reshape(-1, 1, 2).astype(np.float32)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, config.HOMOGRAPHY_RANSAC_PX)
        if H is None:
            # Repli: modele similitude. Moins fidele au sol incline, mais il
            # absorbe quand meme la rotation et la translation dominantes.
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                               ransacReprojThreshold=config.HOMOGRAPHY_RANSAC_PX)
            if M is None:
                return np.zeros(len(a), dtype=np.float32), False, None, 0.0, None
            proj = (a @ M[:, :2].T) + M[:, 2]
            return np.linalg.norm(b - proj, axis=1).astype(np.float32), False, None, 0.0, None

        mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(a), bool)
        candidates = [(H, mask)]

        # Deuxieme plan, ajuste sur les points rejetes par le premier.
        #
        # Quand un mur grandit dans le champ, il finit par rassembler plus de
        # points que le sol et devient le plan "dominant". Sans ce second
        # ajustement, le sol se retrouvait classe hors-plan et la pose etait
        # rejetee -- au moment precis ou l'obstacle est proche. Chercher les
        # deux plans, puis designer le sol par sa normale, supprime cette
        # inversion.
        rest = ~mask
        if rest.sum() >= config.MIN_POINTS_FOR_MODEL:
            H2, in2 = cv2.findHomography(src[rest], dst[rest], cv2.RANSAC,
                                         config.HOMOGRAPHY_RANSAC_PX)
            if H2 is not None:
                full = np.zeros(len(a), bool)
                idx = np.flatnonzero(rest)
                full[idx[in2.ravel().astype(bool)]] = True
                candidates.append((H2, full))

        H_ground, mask_ground = self._select_ground(candidates, a.shape[0])
        proj = cv2.perspectiveTransform(src, H_ground).reshape(-1, 2)
        ratio = float(mask_ground.mean()) if len(mask_ground) else 0.0
        return (np.linalg.norm(b - proj, axis=1).astype(np.float32), True, H_ground,
                ratio, mask_ground)

    def _select_ground(self, candidates, n_points: int):
        """Designe, parmi les plans trouves, celui qui est le sol.

        Le critere est la direction de la normale: le sol est vu sous
        l'inclinaison de la camera, un mur est vu de face. Cette direction est
        mesurable sur l'image, sans echelle et sans capteur.
        """
        if len(candidates) == 1:
            return candidates[0]

        expected = geometry.expected_ground_normal()
        best = candidates[0]
        best_score = -2.0
        for H, mask in candidates:
            decomposition = geometry.decompose_plane(H, self._K.matrix, expected)
            if decomposition is None:
                continue
            score = decomposition[3]
            # A score comparable, le plan le mieux peuple gagne.
            score += 0.05 * float(mask.mean())
            if score > best_score:
                best_score = score
                best = (H, mask)
        return best

    def _ttc_from_expansion(self, a: np.ndarray, b: np.ndarray, dt: float) -> Optional[float]:
        """Temps avant collision deduit du grossissement du nuage de points.

        On mesure la dispersion autour du centroide avant et apres. Si elle
        grandit d'un facteur s sur dt, la surface approche et le contact
        theorique est a dt / (s - 1). Cette grandeur ne depend d'aucune
        distance connue -- c'est justement ce qui la rend utilisable ici.
        """
        return _expansion_ttc(a, b, dt)

    def _residual_threshold(self, residuals: np.ndarray, inliers: Optional[np.ndarray],
                            median_flow_px: float) -> float:
        """Seuil "hors plan", cale sur le bruit d'ajustement du sol.

        La dispersion se mesure sur les seuls points retenus comme appartenant
        au sol. Prendre la mediane de *tous* les residus marchait tant que le
        sol etait majoritaire, mais s'effondrait des qu'un obstacle occupait
        une bonne part de l'image: le bruit apparent grimpait, le seuil montait
        avec lui, et plus rien n'etait signale. Le seuil se neutralisait lui
        meme au moment ou l'obstacle etait le plus proche.
        """
        sample = residuals[inliers] if inliers is not None and inliers.sum() >= 8 else None
        if sample is not None and len(sample) >= 8:
            sigma = 1.4826 * float(np.median(sample))
            base = max(config.RESIDUAL_MIN_PX, config.RESIDUAL_SIGMA_K * sigma)
        elif len(residuals) >= config.MIN_POINTS_FOR_MODEL:
            sigma = 1.4826 * float(np.median(residuals))
            base = max(config.RESIDUAL_MIN_PX, config.RESIDUAL_SIGMA_K * sigma)
        else:
            base = max(config.RESIDUAL_MIN_PX, config.RESIDUAL_FLOW_RATIO * median_flow_px)
        return base / max(0.1, self.sensitivity)

    def _score_cells(self, a: np.ndarray, b: np.ndarray, residuals: np.ndarray,
                     dt: float, w: int, h: int, thr: float) -> List[CellReport]:
        cells = self._empty_cells()
        cell_w = w / float(config.GRID_COLS)
        cell_h = h / float(config.GRID_ROWS)

        col_idx = np.clip((a[:, 0] / cell_w).astype(int), 0, config.GRID_COLS - 1)
        row_idx = np.clip((a[:, 1] / cell_h).astype(int), 0, config.GRID_ROWS - 1)
        ttc_warn = config.TTC_WARN_S * self.sensitivity

        for cell in cells:
            sel = (row_idx == cell.row) & (col_idx == cell.col)
            n = int(sel.sum())
            cell.n_points = n
            if n < config.MIN_POINTS_PER_CELL:
                cell.confirmed = self._hyst.update((cell.row, cell.col), False)
                continue

            off = sel & (residuals > thr)
            cell.outlier_ratio = float((residuals[sel] > thr).mean())

            # Le temps avant collision se mesure sur ce qui n'appartient pas au
            # sol. Melanger les points du plan avec ceux du relief tire la
            # mediane vers l'expansion du sol et masque l'objet qui approche --
            # exactement le cas ou l'alerte compte le plus.
            if int(off.sum()) >= 4:
                cell.ttc = _expansion_ttc(a[off], b[off], dt)
            else:
                cell.ttc = _expansion_ttc(a[sel], b[sel], dt)

            off_plane = cell.outlier_ratio >= config.CELL_OUTLIER_RATIO / max(0.1, self.sensitivity)
            looming = cell.ttc is not None and cell.ttc < ttc_warn
            cell.raw_hit = bool(off_plane or looming)

            # Score: sert uniquement a classer les cellules entre elles.
            score = cell.outlier_ratio
            if cell.ttc is not None and cell.ttc > 0:
                score += min(2.0, ttc_warn / cell.ttc)
            cell.score = float(score)

            cell.confirmed = self._hyst.update((cell.row, cell.col), cell.raw_hit)

        return cells


def _expansion_ttc(a: np.ndarray, b: np.ndarray, dt: float) -> Optional[float]:
    """Temps avant collision a partir du rapport d'echelle entre deux nuages."""
    if len(a) < 4:
        return None
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ra = np.linalg.norm(a - ca, axis=1)
    rb = np.linalg.norm(b - cb, axis=1)
    keep = ra > 1e-3
    if keep.sum() < 4:
        return None
    # Mediane des rapports: insensible aux quelques points mal apparies.
    ratio = float(np.median(rb[keep] / ra[keep]))
    rate = (ratio - 1.0) / dt
    if rate < config.MIN_SCALE_RATE:
        return None
    return 1.0 / rate
