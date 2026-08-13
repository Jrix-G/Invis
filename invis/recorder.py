"""Enregistrement de session: CSV des detections et images brutes.

Un vol dure quelques minutes; rejouer la session au sol vaut mieux que de
regler des seuils en direct pendant que le drone est en l'air.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

from . import config
from .detector import DetectionResult

# t_rel  : instant d'ecriture, pratique pour lire un vol.
# t_frame: instant de reception de l'image, celui qu'a utilise le detecteur.
#          C'est lui qui doit servir au rejeu -- le temps avant collision
#          depend de l'intervalle entre images, pas de la vitesse d'ecriture
#          sur le disque.
CSV_HEADER = [
    "t_rel", "t_frame", "frame", "state", "reason", "n_tracked", "median_flow_px",
    "global_ttc", "plane_found", "dt",
]


class SessionRecorder:
    def __init__(self, base_dir: Optional[str] = None, save_frames: bool = True) -> None:
        root = base_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), config.SESSION_DIR)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, stamp)
        self.frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frames_dir if save_frames else self.dir, exist_ok=True)

        self.save_frames = save_frames
        self._t0 = time.time()
        self._frame_t0: Optional[float] = None
        self._csv_file = open(os.path.join(self.dir, "detections.csv"), "w",
                              newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        header = list(CSV_HEADER)
        for row in config.CELL_NAMES:
            for name in row:
                header += [f"{name}_pts", f"{name}_out", f"{name}_ttc", f"{name}_conf"]
        self._writer.writerow(header)
        self._n = 0

    def write(self, result: DetectionResult, jpeg: Optional[bytes] = None,
              frame_time: Optional[float] = None) -> None:
        self._n += 1
        if frame_time is not None and self._frame_t0 is None:
            self._frame_t0 = frame_time
        # Six decimales: l'intervalle entre images pilote le temps avant
        # collision, et l'arrondi ne doit pas peser plus que la mesure.
        t_frame = "" if frame_time is None else f"{frame_time - (self._frame_t0 or frame_time):.6f}"
        row = [
            f"{time.time() - self._t0:.3f}",
            t_frame,
            self._n,
            result.state,
            result.reason,
            result.n_tracked,
            f"{result.median_flow_px:.3f}",
            f"{result.global_ttc:.3f}" if result.global_ttc else "",
            int(result.plane_found),
            f"{result.dt:.3f}",
        ]
        by_key = {(c.row, c.col): c for c in result.cells}
        for r, names in enumerate(config.CELL_NAMES):
            for c, _name in enumerate(names):
                cell = by_key.get((r, c))
                if cell is None:
                    row += ["", "", "", ""]
                    continue
                row += [
                    cell.n_points,
                    f"{cell.outlier_ratio:.3f}",
                    f"{cell.ttc:.3f}" if cell.ttc else "",
                    int(cell.confirmed),
                ]
        self._writer.writerow(row)

        if self.save_frames and jpeg:
            path = os.path.join(self.frames_dir, f"{self._n:06d}.jpg")
            with open(path, "wb") as fh:
                fh.write(jpeg)

    def close(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:  # noqa: BLE001
            pass
