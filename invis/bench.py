"""Mesure de la cadence reelle du lien video (etape P4).

Pourquoi une mesure et pas un reglage
-------------------------------------
Le premier reflexe pour gagner des images par seconde serait de remonter
CAM_PIN_XCLK_FREQ_HZ. L'historique inscrit dans
main/cam_mavlink_config_camera.h dit l'inverse: 20 MHz et 10 MHz donnaient des
images corrompues *y compris sur /jpg*, donc hors de toute charge Wi-Fi, et
5 MHz est le reglage qui a rendu l'image propre. Remonter l'horloge sans
mesure reviendrait a echanger une cadence contre une image inutilisable pour
la detection.

Cet outil mesure donc ce qu'on a reellement, et ce que les seuls leviers sans
risque -- qualite JPEG, taille d'image, plafond de cadence -- changent
vraiment. Il ne modifie jamais l'horloge du capteur et ne touche aucun
endpoint de pilotage.

Usage:
    python -m invis.bench --host 192.168.4.50
    python -m invis.bench --qualities 10,12,16,20 --seconds 8
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis import config
    from invis.mjpeg_client import VideoLink, http_get
else:
    from . import config
    from .mjpeg_client import VideoLink, http_get


@dataclass
class BenchRow:
    label: str
    fps: float
    kbps: float
    jpeg_avg_kb: float
    jitter_ms: float
    frames: int
    decoded_ok: int
    errors: str = ""


def _decode_check(jpeg: bytes) -> bool:
    """Verifie que l'image est decodable. Sans OpenCV, on teste les marqueurs."""
    try:
        import cv2  # import local: bench doit rester utilisable sans OpenCV
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img is not None
    except ImportError:
        return jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")


def measure(host: str, seconds: float, mode: str, label: str) -> BenchRow:
    link = VideoLink(host=host, mode=mode, on_log=lambda m: None)
    link.start()
    stamps: List[float] = []
    sizes: List[int] = []
    decoded = 0
    deadline = time.time() + seconds
    # Laisser le lien s'etablir avant de compter.
    settle = time.time() + 1.5
    while time.time() < deadline:
        frame = link.take_latest()
        if frame is None:
            time.sleep(0.003)
            continue
        if time.time() < settle:
            continue
        stamps.append(frame.recv_time)
        sizes.append(len(frame.jpeg))
        if _decode_check(frame.jpeg):
            decoded += 1
    err = link.stats.last_error
    link.stop()

    if len(stamps) < 2:
        return BenchRow(label, 0.0, 0.0, 0.0, 0.0, len(stamps), decoded, err or "aucune image")

    span = stamps[-1] - stamps[0]
    gaps = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
    return BenchRow(
        label=label,
        fps=(len(stamps) - 1) / span if span > 0 else 0.0,
        kbps=sum(sizes) / span / 1024.0 if span > 0 else 0.0,
        jpeg_avg_kb=statistics.mean(sizes) / 1024.0,
        jitter_ms=statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
        frames=len(stamps),
        decoded_ok=decoded,
        errors=err,
    )


def apply_control(host: str, quality: Optional[int] = None, framesize: Optional[str] = None,
                  fps: Optional[int] = None) -> str:
    params = []
    if quality is not None:
        params.append(f"quality={quality}")
    if framesize is not None:
        params.append(f"framesize={framesize}")
    if fps is not None:
        params.append(f"fps={fps}")
    if not params:
        return ""
    return http_get(host, f"{config.CONTROL_PATH}?" + "&".join(params))


def print_table(rows: List[BenchRow]) -> None:
    head = f"{'profil':<22}{'fps':>7}{'kB/s':>9}{'img kB':>9}{'gigue ms':>10}{'images':>8}{'decodees':>10}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r.label:<22}{r.fps:>7.2f}{r.kbps:>9.1f}{r.jpeg_avg_kb:>9.2f}"
              f"{r.jitter_ms:>10.1f}{r.frames:>8}{r.decoded_ok:>10}")
        if r.errors:
            print(f"{'':<22}erreur: {r.errors}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Banc de mesure du lien video ESP32-CAM")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--seconds", type=float, default=8.0, help="duree par profil")
    parser.add_argument("--qualities", default="10,12,16,20",
                        help="qualites JPEG a tester, separees par des virgules")
    parser.add_argument("--framesizes", default="qvga",
                        help="tailles a tester (qqvga,qvga,cif,vga...)")
    parser.add_argument("--mode", default="stream", choices=("stream", "snapshot", "both"))
    parser.add_argument("--restore-quality", type=int, default=12,
                        help="qualite remise en place a la fin (12 = defaut firmware)")
    args = parser.parse_args()

    print(f"cible {args.host}  --  horloge capteur inchangee (5 MHz, cf. bench.py)")
    try:
        print(f"/status initial: {http_get(args.host, config.STATUS_PATH).strip()[:300]}")
    except Exception as exc:  # noqa: BLE001
        print(f"/status indisponible: {exc}")

    modes = ["stream", "snapshot"] if args.mode == "both" else [args.mode]
    qualities = [int(q) for q in args.qualities.split(",") if q.strip()]
    framesizes = [f.strip() for f in args.framesizes.split(",") if f.strip()]

    rows: List[BenchRow] = []
    try:
        for mode in modes:
            for fs in framesizes:
                for q in qualities:
                    try:
                        apply_control(args.host, quality=q, framesize=fs)
                    except Exception as exc:  # noqa: BLE001
                        print(f"reglage {fs}/q{q} refuse: {exc}")
                        continue
                    # Le capteur a besoin de quelques images pour se stabiliser
                    # apres un changement de taille.
                    time.sleep(1.0)
                    label = f"{mode} {fs} q{q}"
                    print(f"... mesure {label}")
                    rows.append(measure(args.host, args.seconds, mode, label))
    finally:
        try:
            apply_control(args.host, quality=args.restore_quality, framesize="qvga")
            print(f"reglages restaures: qvga q{args.restore_quality}")
        except Exception as exc:  # noqa: BLE001
            print(f"restauration impossible: {exc}")

    print()
    print_table(rows)
    print()
    print("Lecture: si la cadence ne bouge presque pas quand la qualite baisse,")
    print("le goulot n'est pas le Wi-Fi mais le capteur (XCLK 5 MHz) ou la recopie")
    print("PSRAM. Dans ce cas seul un essai d'horloge tranche -- et l'historique")
    print("du projet dit que 10 et 20 MHz corrompaient l'image sur cette carte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
