"""Rejoue une session enregistree a travers le detecteur.

Regler des seuils pendant que le drone vole n'a pas de sens. Une session
enregistree par gcs_vision (images + horodatage CSV) se rejoue ici autant de
fois que necessaire, avec d'autres seuils, sans remettre l'appareil en l'air.

Usage:
    python -m invis.replay invis/sessions/20260813_101500
    python -m invis.replay <dossier> --sensitivity 1.4 --no-window
    python -m invis.replay <dossier> --height 2.4

La hauteur peut etre corrigee apres coup: toutes les distances lui sont
proportionnelles, donc rejouer avec la bonne valeur les remet a l'echelle sans
rien recalculer d'autre.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis import config, geometry, overlay
    from invis.detector import STATE_OBSTACLE, ObstacleDetector
    from invis.mapper import Mapper
else:
    from . import config, geometry, overlay
    from .detector import STATE_OBSTACLE, ObstacleDetector
    from .mapper import Mapper


def load_timestamps(session_dir: str) -> List[float]:
    """Horodatage des images tel que le detecteur l'a vu.

    On prend t_frame (reception de l'image) et non t_rel (ecriture sur
    disque): le temps avant collision se deduit de l'intervalle entre images.
    Rejouer avec les mauvais intervalles fabrique des alertes qui n'ont jamais
    existe en vol.
    """
    path = os.path.join(session_dir, "detections.csv")
    stamps: List[float] = []
    if not os.path.exists(path):
        return stamps
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("t_frame") or "").strip()
            if not value:
                value = (row.get("t_rel") or "").strip()
            try:
                stamps.append(float(value))
            except ValueError:
                break
    return stamps


def main() -> int:
    parser = argparse.ArgumentParser(description="Rejeu d'une session vision")
    parser.add_argument("session", help="dossier de session (contient frames/ et detections.csv)")
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0, help="1 = temps reel, 0 = aussi vite que possible")
    parser.add_argument("--no-window", action="store_true", help="console seulement")
    parser.add_argument("--height", type=float, default=config.DEFAULT_HEIGHT_M,
                        help="hauteur de vol supposee, en metres")
    parser.add_argument("--sigma", type=float, default=config.DEFAULT_SIGMA_H_M,
                        help="incertitude sur cette hauteur")
    parser.add_argument("--flip", default=None, choices=("none", "h", "v", "hv"),
                        help="redressement de l'image (defaut: reglage de config.py)")
    args = parser.parse_args()

    frames = sorted(glob.glob(os.path.join(args.session, "frames", "*.jpg")))
    if not frames:
        print(f"aucune image dans {args.session}/frames")
        return 1

    if args.flip is None:
        flip_h, flip_v = config.CAMERA_FLIP_H, config.CAMERA_FLIP_V
    else:
        flip_h, flip_v = "h" in args.flip, "v" in args.flip
    print(f"redressement: miroir H={flip_h} V={flip_v}")

    stamps = load_timestamps(args.session)
    detector = ObstacleDetector()
    detector.sensitivity = args.sensitivity
    mapper = Mapper(height_m=args.height, sigma_h_m=args.sigma)

    print(f"{len(frames)} images, sensibilite {args.sensitivity}")
    last_state: Optional[str] = None
    obstacle_frames = 0

    for i, path in enumerate(frames):
        with open(path, "rb") as fh:
            jpeg = fh.read()
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"{i:6d}  image illisible")
            continue
        bgr = geometry.orient_frame(bgr, flip_h, flip_v)

        # Horodatage d'origine quand il existe: le temps avant collision depend
        # de l'intervalle reel entre images, pas de la vitesse de rejeu.
        t = stamps[i] if i < len(stamps) else i * 0.15
        result = detector.process(bgr, t)
        mapframe = mapper.update(result)

        if result.state == STATE_OBSTACLE:
            obstacle_frames += 1
        if result.state != last_state:
            measure = mapframe.nearest or mapframe.contact
            distance = f"  {measure.range_m:.2f} m" if measure else ""
            print(f"{t:7.2f}s  {result.state:<9} {result.reason}{distance}")
            last_state = result.state

        if not args.no_window:
            view = overlay.draw(bgr, result, show_flow=True, mapframe=mapframe)
            cv2.imshow("replay", cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            delay = 1
            if args.speed > 0 and i + 1 < len(stamps):
                delay = max(1, int((stamps[i + 1] - t) * 1000 / args.speed))
            if cv2.waitKey(delay) & 0xFF == 27:
                break
        elif args.speed > 0 and i + 1 < len(stamps):
            time.sleep(max(0.0, (stamps[i + 1] - t) / args.speed))

    if not args.no_window:
        cv2.destroyAllWindows()

    pct = 100.0 * obstacle_frames / len(frames)
    print(f"\nobstacle sur {obstacle_frames}/{len(frames)} images ({pct:.1f}%)")
    print(f"nuage reconstruit: {len(mapper.cloud)} points, "
          f"hauteur supposee {args.height:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
