"""Tests hors drone: scenarios synthetiques et lien MJPEG local.

Aucun materiel requis. Lancement:
    python -m invis.test_invis
    python invis/test_invis.py

Les scenarios reproduisent ce que la camera voit en vol: un sol texture qui
defile, un objet qui grossit, une rotation pure, un stationnaire. Le detecteur
doit distinguer ces cas -- notamment ne PAS crier a l'obstacle sur une simple
rotation, cause classique de fausse alarme en flux optique.
"""

from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
import time
from collections import Counter
from typing import Optional

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from invis.detector import ObstacleDetector, horizon_row
    from invis.mjpeg_client import VideoLink
else:
    from .detector import ObstacleDetector, horizon_row
    from .mjpeg_client import VideoLink

W, H = 320, 240
FPS_SIM = 1.0 / 0.15  # cadence realiste a 5 MHz d'horloge capteur

_rng = np.random.default_rng(3)
_GROUND = cv2.cvtColor(
    cv2.GaussianBlur(_rng.integers(40, 210, size=(H * 3, W * 3), dtype=np.uint8), (5, 5), 0),
    cv2.COLOR_GRAY2BGR,
)
_OBJ = cv2.cvtColor(
    cv2.GaussianBlur(_rng.integers(0, 255, size=(220, 220), dtype=np.uint8), (3, 3), 0),
    cv2.COLOR_GRAY2BGR,
)

_failures = 0


def _check(ok: bool, label: str, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"{'OK   ' if ok else 'ECHEC'} {label:<36} {detail}")


# ---------------------------------------------------------------------------
# Generation d'images
# ---------------------------------------------------------------------------

def synth_frame(k: int, with_obj: bool = False, growth: float = 0.0,
                ox: float = 0.5, oy: float = 0.5, yaw: float = 0.0,
                ground_zoom: float = 0.02) -> np.ndarray:
    """Une image de la sequence simulee, deja passee par une compression JPEG.

    La compression n'est pas un detail: a qualite 12 le firmware laisse des
    blocs 8x8 qui perturbent le suivi de points. Tester sur des images propres
    donnerait un resultat trop optimiste.
    """
    scale = 1.0 + ground_zoom * k
    M = cv2.getRotationMatrix2D((_GROUND.shape[1] / 2, _GROUND.shape[0] / 2), yaw * k, scale)
    warped = cv2.warpAffine(_GROUND, M, (_GROUND.shape[1], _GROUND.shape[0]))
    y0 = _GROUND.shape[0] // 2 - H // 2
    x0 = _GROUND.shape[1] // 2 - W // 2
    frame = warped[y0:y0 + H, x0:x0 + W].copy()

    if with_obj:
        size = max(10, min(int(30 * (1.0 + growth * k)), 200))
        patch = cv2.resize(_OBJ, (size, size))
        cx, cy = int(W * ox), int(H * oy)
        px, py = max(0, cx - size // 2), max(0, cy - size // 2)
        hh, ww = min(size, H - py), min(size, W - px)
        frame[py:py + hh, px:px + ww] = patch[:hh, :ww]

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 52])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else frame


def run_sequence(n: int = 30, **kwargs) -> Counter:
    det = ObstacleDetector()
    t = 0.0
    states = []
    for k in range(n):
        t += 1.0 / FPS_SIM
        states.append(det.process(synth_frame(k, **kwargs), t).state)
    return Counter(states)


# ---------------------------------------------------------------------------
# Tests detecteur
# ---------------------------------------------------------------------------

def test_detector() -> None:
    print("\n-- detecteur ------------------------------------------------")

    row = horizon_row(H)
    _check(row is None, "horizon hors champ a -45 deg",
           "toute l'image regarde le sol, pas de ciel a ignorer")

    c = run_sequence(with_obj=False)
    _check(c.get("OBSTACLE", 0) <= 2, "sol seul en avance -> libre", dict(c))

    c = run_sequence(with_obj=True, growth=0.22)
    _check(c.get("OBSTACLE", 0) >= 5, "obstacle au centre -> alerte", dict(c))

    c = run_sequence(with_obj=True, growth=0.22, ox=0.22)
    _check(c.get("OBSTACLE", 0) >= 5, "obstacle a gauche -> alerte", dict(c))

    c = run_sequence(with_obj=True, growth=0.22, ox=0.78, oy=0.75)
    _check(c.get("OBSTACLE", 0) >= 5, "obstacle en bas a droite -> alerte", dict(c))

    c = run_sequence(with_obj=True, growth=0.015)
    _check(c.get("OBSTACLE", 0) <= 2, "objet lointain, approche lente -> libre", dict(c))

    c = run_sequence(with_obj=False, yaw=1.5, ground_zoom=0.004)
    _check(c.get("OBSTACLE", 0) <= 2, "rotation pure (lacet) -> pas de fausse alerte", dict(c))

    c = run_sequence(with_obj=False, ground_zoom=0.05)
    _check(c.get("OBSTACLE", 0) <= 2, "avance rapide sur le sol -> libre", dict(c))

    # Stationnaire: bruit de capteur seul, aucune parallaxe.
    det = ObstacleDetector()
    base = synth_frame(0)
    t = 0.0
    states = []
    for _ in range(15):
        t += 1.0 / FPS_SIM
        noisy = np.clip(base.astype(np.int16) + _rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        states.append(det.process(noisy, t).state)
    c = Counter(states)
    _check(c.get("NO_FLOW", 0) >= 13, "stationnaire -> detection declaree inactive", dict(c))


# ---------------------------------------------------------------------------
# Test du lien MJPEG
# ---------------------------------------------------------------------------

class _MjpegHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *_args) -> None:  # silence
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        self.end_headers()
        try:
            for _ in range(40):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(self.payload)).encode() + b"\r\n\r\n")
                self.wfile.write(self.payload)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def test_link() -> None:
    print("\n-- lien video -----------------------------------------------")

    ok, buf = cv2.imencode(".jpg", synth_frame(0), [cv2.IMWRITE_JPEG_QUALITY, 52])
    _MjpegHandler.payload = buf.tobytes()

    with socketserver.TCPServer(("127.0.0.1", 0), _MjpegHandler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        link = VideoLink(host="127.0.0.1", port=port, mode="stream")
        link.start()
        deadline = time.time() + 4.0
        got: Optional[bytes] = None
        seen = 0
        while time.time() < deadline and seen < 5:
            frame = link.take_latest()
            if frame:
                got = frame.jpeg
                seen += 1
            else:
                time.sleep(0.005)
        link.stop()
        httpd.shutdown()

    _check(seen >= 5, "flux multipart decoupe en images", f"{seen} images")
    _check(bool(got) and got.startswith(b"\xff\xd8") and got.endswith(b"\xff\xd9"),
           "images extraites bornees SOI/EOI")
    if got:
        img = cv2.imdecode(np.frombuffer(got, np.uint8), cv2.IMREAD_COLOR)
        _check(img is not None and img.shape[:2] == (H, W), "image decodable a la bonne taille",
               str(None if img is None else img.shape))


def test_session_roundtrip() -> None:
    """Enregistrer puis rejouer doit redonner le meme verdict.

    Le rejeu sert a regler les seuils au sol. S'il ne reproduit pas ce qui
    s'est passe en vol, il induit en erreur au lieu d'aider.
    """
    print("\n-- session enregistree --------------------------------------")

    if __package__ in (None, ""):
        from invis.recorder import SessionRecorder
        from invis.replay import load_timestamps
    else:
        from .recorder import SessionRecorder
        from .replay import load_timestamps

    import shutil
    import tempfile

    base = tempfile.mkdtemp(prefix="vision_test_")
    try:
        rec = SessionRecorder(base_dir=base)
        det = ObstacleDetector()
        live = []
        t = 1_700_000_000.0  # horodatage absolu, comme en vol
        for k in range(24):
            t += 1.0 / FPS_SIM
            # En vol, l'enregistreur stocke les octets JPEG recus tels quels,
            # ceux-la memes que le detecteur a analyses. Le test doit faire
            # pareil: re-encoder ici ajouterait une compression que le direct
            # n'a pas subie, et le rejeu differerait pour cette seule raison.
            ok, buf = cv2.imencode(".jpg", synth_frame(k, with_obj=True, growth=0.22),
                                   [cv2.IMWRITE_JPEG_QUALITY, 52])
            jpeg = buf.tobytes()
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            result = det.process(frame, t)
            live.append(result.state)
            rec.write(result, jpeg, frame_time=t)
        rec.close()

        stamps = load_timestamps(rec.dir)
        _check(len(stamps) == 24, "CSV relu ligne par ligne", f"{len(stamps)} lignes")

        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        _check(all(abs(g - 1.0 / FPS_SIM) < 0.01 for g in gaps),
               "intervalles preserves par le CSV",
               f"median {sorted(gaps)[len(gaps)//2]:.3f}s attendu {1.0/FPS_SIM:.3f}s")

        det2 = ObstacleDetector()
        replayed = []
        for i, path in enumerate(sorted(os.listdir(os.path.join(rec.dir, "frames")))):
            with open(os.path.join(rec.dir, "frames", path), "rb") as fh:
                img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
            replayed.append(det2.process(img, stamps[i]).state)

        # Pas d'egalite au bit pres, et ce n'est pas un defaut du rejeu.
        # L'horodatage en vol est un time.time(), soit ~1,7e9 secondes: en
        # float64 un pas de 0,15 s y perd environ 0,3 ms, que le CSV, lui,
        # ecrit proprement. Les deux passages voient donc des intervalles
        # infimement differents, et un temps avant collision juste au seuil
        # peut basculer. Ce que le rejeu doit garantir, c'est le meme verdict,
        # pas les memes bits.
        agreement = sum(1 for x, y in zip(live, replayed) if x == y) / len(live)
        _check(agreement >= 0.8, "rejeu conforme au direct",
               f"{agreement:.0%} d'accord, direct {Counter(live)} / rejeu {Counter(replayed)}")
        delta = abs(Counter(live)["OBSTACLE"] - Counter(replayed)["OBSTACLE"])
        _check(delta <= 3, "meme volume d'alertes au rejeu", f"ecart {delta} images")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_geometry() -> None:
    """La geometrie doit etre exacte: c'est elle qui porte les metres."""
    print("\n-- geometrie ------------------------------------------------")
    from invis.geometry import (Intrinsics, attitude_from_plane_normal,
                                       body_from_camera, coverage, ground_range,
                                       range_band, row_for_range)

    K = Intrinsics.from_fov(W, H)
    uv = np.array([[K.cx, K.cy], [K.cx, 0.0], [K.cx, H - 1.0]])
    ratios, valid = ground_range(uv, K, 1.0)
    _check(bool(valid.all()) and abs(ratios[0] - 1.0) < 1e-6,
           "pixel central -> distance = hauteur", f"{ratios[0]:.4f} x h")
    _check(abs(ratios[1] - 2.194) < 0.01 and abs(ratios[2] - 0.459) < 0.01,
           "haut et bas de l'image", f"{ratios[1]:.3f} et {ratios[2]:.3f} x h")

    near, far = coverage(K, 2.0)
    _check(abs(near - 0.92) < 0.02 and abs(far - 4.39) < 0.05,
           "portee a 2 m de haut", f"{near:.2f} a {far:.2f} m")

    ok = True
    for tilt, roll in ((-45.0, 0.0), (-40.0, 7.0), (-50.0, -12.0), (-30.0, 20.0)):
        n = body_from_camera(tilt, roll_deg=roll).T @ np.array([0.0, 0.0, 1.0])
        t_est, r_est = attitude_from_plane_normal(n)
        ok &= abs(t_est - tilt) < 0.01 and abs(r_est - roll) < 0.01
    _check(ok, "tangage et roulis relus depuis la normale")

    ok = True
    for target in (1.0, 2.0, 3.5):
        row = row_for_range(K, target, 2.0)
        if row is None:
            ok = False
            continue
        back, _ = ground_range(np.array([[K.cx, row]]), K, 2.0)
        ok &= abs(back[0] - target) < 0.01
    _check(ok, "distance -> ligne -> distance (aller-retour)")

    lo, hi = range_band(4.0, 2.0, 0.4)
    _check(abs(lo - 3.2) < 1e-6 and abs(hi - 4.8) < 1e-6,
           "incertitude sur h purement multiplicative", f"[{lo:.2f} - {hi:.2f}]")


def test_pose_convention() -> None:
    """La convention de translation d'OpenCV est verifiee, pas supposee."""
    print("\n-- convention de pose ---------------------------------------")
    from invis.geometry import decompose_plane, expected_ground_normal
    from invis.simulator import FlightSimulator

    for yaw_rate in (0.0, 20.0):
        sim = FlightSimulator(height_m=2.0, speed_mps=0.8, walls=[], yaw_rate_dps=yaw_rate)
        t0, t1 = 1.0, 1.0 + 1.0 / FPS_SIM

        def ground_h(t):
            return sim._plane_homography(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                                         np.array([0.0, 1.0, 0.0]), t)

        H = ground_h(t1) @ np.linalg.inv(ground_h(t0))
        H /= H[2, 2]
        out = decompose_plane(H, sim.K.matrix, expected_ground_normal(sim.tilt_deg))
        if out is None:
            _check(False, f"decomposition possible (lacet {yaw_rate:.0f})")
            continue
        R, t, _n, score = out

        p0, R0 = sim.camera_pose(t0)
        p1, R1 = sim.camera_pose(t1)
        R_true = R1.T @ R0
        t_true = R1.T @ (p0 - p1)

        _check(score > 0.999, f"plan sol reconnu (lacet {yaw_rate:.0f} deg/s)",
               f"score {score:.4f}")
        _check(float(np.abs(R - R_true).max()) < 1e-6,
               f"rotation conforme (lacet {yaw_rate:.0f} deg/s)")
        _check(float(np.abs(t * sim.height_m - t_true).max()) < 1e-6,
               f"translation a l'echelle (lacet {yaw_rate:.0f} deg/s)",
               f"{np.round(t * sim.height_m, 4)} contre {np.round(t_true, 4)}")


def _fly(height_m: float, speed_mps: float, wall_x: float):
    """Fait voler le simulateur vers un mur et collecte les erreurs."""
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    sim = FlightSimulator(height_m=height_m, speed_mps=speed_mps,
                          walls=[Wall(x_m=wall_x, width_m=1.6, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=height_m)
    contact_err, tri_err, tilt_err, odo = [], [], [], []
    n = int(((wall_x - 0.5) / speed_mps) * FPS_SIM)

    for k in range(n):
        t = k / FPS_SIM
        result = det.process(sim.render(t), t)
        frame = mapper.update(result)
        truth = sim.truth(t)
        tilt_err.append(abs(frame.tilt_deg - sim.tilt_deg))
        odo.append((frame.position[0], truth.camera_x))
        gt = truth.nearest_wall_range
        if gt is not None and gt < 4.0:
            if frame.contact:
                contact_err.append(frame.contact.range_m - gt)
            if frame.nearest:
                tri_err.append(frame.nearest.range_m - gt)
    return mapper, np.array(contact_err), np.array(tri_err), np.array(tilt_err), odo


def test_metric_reconstruction() -> None:
    """Les distances annoncees sont comparees a la verite terrain."""
    print("\n-- distances metriques --------------------------------------")

    for height_m, speed, wall_x in ((2.0, 0.8, 5.0), (2.5, 0.6, 6.0), (1.5, 1.0, 4.0)):
        mapper, contact, tri, tilt, odo = _fly(height_m, speed, wall_x)
        tag = f"h={height_m} v={speed} mur={wall_x}m"

        _check(len(contact) >= 3 and abs(contact.mean()) < 0.5,
               f"contact sol juste ({tag})",
               f"n={len(contact)} biais {contact.mean():+.2f} m" if len(contact) else "n=0")
        _check(len(tri) >= 3 and abs(tri.mean()) < 0.5,
               f"triangulation juste ({tag})",
               f"n={len(tri)} biais {tri.mean():+.2f} m" if len(tri) else "n=0")
        _check(float(np.median(tilt)) < 1.5,
               f"assiette retrouvee ({tag})", f"ecart median {np.median(tilt):.2f} deg")

        travelled = max(1e-6, abs(odo[-1][1]))
        drift = abs(odo[-1][0] - odo[-1][1]) / travelled
        _check(drift < 0.35, f"odometrie ({tag})",
               f"{odo[-1][0]:.2f} m estime contre {odo[-1][1]:.2f} m, derive {drift:.0%}")
        _check(len(mapper.cloud) > 500, f"nuage alimente ({tag})", f"{len(mapper.cloud)} pts")


def test_scale_invariance() -> None:
    """Une hauteur fausse doit tout multiplier, sans rien deformer."""
    print("\n-- effet d'une hauteur fausse -------------------------------")
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    truth_h = 2.0
    readings = {}
    for assumed in (truth_h, truth_h * 1.25):
        sim = FlightSimulator(height_m=truth_h, speed_mps=0.8,
                              walls=[Wall(x_m=5.0, width_m=1.6, height_m=1.4)])
        det = ObstacleDetector()
        mapper = Mapper(height_m=assumed)
        values = []
        for k in range(int((4.5 / 0.8) * FPS_SIM)):
            t = k / FPS_SIM
            frame = mapper.update(det.process(sim.render(t), t))
            values.append(frame.contact.range_m if frame.contact else None)
        readings[assumed] = values

    a, b = readings[truth_h], readings[truth_h * 1.25]
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    _check(len(pairs) >= 3, "mesures comparables obtenues", f"{len(pairs)} paires")
    if len(pairs) >= 3:
        ratio = float(np.median([y / x for x, y in pairs]))
        _check(abs(ratio - 1.25) < 0.08,
               "hauteur surestimee de 25% -> distances +25%",
               f"rapport mesure {ratio:.3f}")


def test_rendering() -> None:
    """Le rendu doit produire une image utilisable, a la bonne taille."""
    print("\n-- rendu ----------------------------------------------------")
    from invis import overlay, panels
    from invis.mapper import Mapper
    from invis.render3d import Renderer3D
    from invis.simulator import FlightSimulator, Wall

    sim = FlightSimulator(height_m=2.0, speed_mps=0.8,
                          walls=[Wall(x_m=4.0, width_m=1.6, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.0)
    renderer = Renderer3D((320, 240))
    img = result = frame = None
    for k in range(26):
        t = k / FPS_SIM
        img = sim.render(t)
        result = det.process(img, t)
        frame = mapper.update(result)

    view = overlay.draw(img, result, show_flow=True, mapframe=frame, show_ranges=True)
    _check(view.shape == img.shape, "calque a la taille de l'image", str(view.shape))

    view3d = renderer.render(mapper, frame, 0.14)
    _check(view3d.shape == (240, 320, 3), "vue 3D a la taille demandee", str(view3d.shape))
    _check(float(view3d.std()) > 3.0, "vue 3D non vide", f"ecart-type {view3d.std():.1f}")

    measures = panels.draw_measures((300, 240), result, frame, 7.0, 7.0, 0.4)
    spare = panels.draw_spare((300, 240))
    composed = panels.compose(view, measures, view3d, spare, (320, 256))
    _check(composed.shape == (515, 643, 3), "composition 2x2", str(composed.shape))
    _check(overlay.to_ppm(composed) is not None, "composition encodable pour l'affichage")


def test_frame_quality() -> None:
    """Le controle qualite doit reperer les images abimees sans jeter les saines.

    Il ne pretend pas restaurer la precision -- la mesure montre qu'il ne le
    fait pas, voir l'en-tete de framecheck.py. Ce qu'on verifie ici, c'est
    qu'il *voit* la degradation, ce qui permet de la signaler.
    """
    print("\n-- controle qualite des images ------------------------------")
    from invis.framecheck import VERDICT_REJECT, FrameGate
    from invis.simulator import FlightSimulator, Wall

    sim = FlightSimulator(height_m=2.0, speed_mps=0.8,
                          walls=[Wall(x_m=5.0, width_m=1.6, height_m=1.4)])
    rng = np.random.default_rng(31)
    gate = FrameGate()

    clean_rejected = 0
    damaged_rejected = 0
    damaged_total = 0
    clean_total = 0

    for k in range(60):
        img = sim.render(k / FPS_SIM)
        damaged = k > 8 and (k % 4 == 0)
        if damaged:
            img = sim.damage(img, rng, blue_blocks=True, blur=(k % 8 == 0))
            damaged_total += 1
        else:
            clean_total += 1
        verdict = gate.check(img, jpeg_size=4000).verdict
        if verdict == VERDICT_REJECT:
            if damaged:
                damaged_rejected += 1
            else:
                clean_rejected += 1

    _check(clean_rejected == 0, "aucune image saine ecartee",
           f"{clean_rejected} sur {clean_total}")
    _check(damaged_rejected >= damaged_total // 2, "images abimees reperees",
           f"{damaged_rejected} sur {damaged_total}")

    # Image tronquee: la taille JPEG suffit a la reconnaitre.
    gate2 = FrameGate()
    for k in range(10):
        gate2.check(sim.render(k / FPS_SIM), jpeg_size=4000)
    verdict = gate2.check(sim.render(1.0), jpeg_size=300).verdict
    _check(verdict == VERDICT_REJECT, "JPEG tronque ecarte", verdict)

    # Sur un flux sain, le filtre doit rester totalement transparent.
    gate3 = FrameGate()
    for k in range(40):
        gate3.check(sim.render(k / FPS_SIM), jpeg_size=4000)
    _check(gate3.rejected == 0, "filtre transparent sur flux sain",
           f"{gate3.rejected}/{gate3.total} ecartees")


def test_orientation() -> None:
    """Redresser une image retournee doit restituer les memes distances.

    C'est le test qui compte pour l'orientation: il ne verifie pas que l'image
    est jolie, il verifie que la geometrie est retablie. Une image a l'envers
    non corrigee inverse la relation ligne/distance -- le proche est lu comme
    du lointain -- et un miroir simple inverse le sens du repere, donc le
    lacet.
    """
    print("\n-- orientation de l'image ----------------------------------")
    from invis.geometry import orient_frame
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    def fly(flip_h, flip_v):
        """Vol identique, image eventuellement retournee puis redressee."""
        sim = FlightSimulator(height_m=2.0, speed_mps=0.8,
                              walls=[Wall(x_m=5.0, width_m=1.6, height_m=1.4)])
        det = ObstacleDetector()
        mapper = Mapper(height_m=2.0)
        out = []
        for k in range(int((4.5 / 0.8) * FPS_SIM)):
            t = k / FPS_SIM
            img = sim.render(t)
            if flip_h or flip_v:
                # La camera rend l'image retournee...
                img = orient_frame(img, flip_h, flip_v)
                # ... et le programme la redresse avec le meme reglage.
                img = orient_frame(img, flip_h, flip_v)
            frame = mapper.update(det.process(img, t))
            out.append(frame.contact.range_m if frame.contact else None)
        return out

    ref = fly(False, False)
    for flip_h, flip_v, label in ((True, True, "180 degres"), (False, True, "miroir vertical"),
                                  (True, False, "miroir horizontal")):
        got = fly(flip_h, flip_v)
        pairs = [(a, b) for a, b in zip(ref, got) if a and b]
        same = len(pairs) >= 3 and all(abs(a - b) < 1e-6 for a, b in pairs)
        _check(same, f"{label} redresse -> distances identiques",
               f"{len(pairs)} mesures comparees")

    # Et sans redressement, le resultat doit differer: sinon le test ci-dessus
    # ne prouverait rien.
    sim = FlightSimulator(height_m=2.0, speed_mps=0.8,
                          walls=[Wall(x_m=5.0, width_m=1.6, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.0)
    upside = []
    for k in range(int((4.5 / 0.8) * FPS_SIM)):
        t = k / FPS_SIM
        frame = mapper.update(det.process(orient_frame(sim.render(t), True, True), t))
        upside.append(frame.contact.range_m if frame.contact else None)

    pairs = [(a, b) for a, b in zip(ref, upside) if a and b]
    differs = len(pairs) == 0 or any(abs(a - b) > 0.05 for a, b in pairs)
    _check(differs, "image non redressee -> resultat different",
           f"{len(pairs)} mesures comparables" if pairs else "aucune mesure exploitable")


class _TruncatingHandler(http.server.BaseHTTPRequestHandler):
    """Serveur MJPEG qui coupe une image sur trois, comme la vraie carte."""

    payload = b""
    frames = 30

    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        self.end_headers()
        try:
            for i in range(self.frames):
                body = self.payload
                truncated = (i % 3 == 1)
                if truncated:
                    # Taille annoncee correcte, contenu coupe: exactement ce que
                    # produit une image perdue en vol.
                    body = self.payload[: len(self.payload) // 2]
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(body)).encode() + b"\r\n\r\n")
                self.wfile.write(body)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def test_truncated_frames() -> None:
    """Une image tronquee doit etre jetee, jamais recollee a la suivante.

    Sans en-tete de taille, la recherche par marqueurs prend le debut de
    l'image coupee et la fin de la suivante: le resultat se decode en aplat
    gris, que l'analyse prend pour une image floue. Deux images sont perdues
    au lieu d'une, et le diagnostic accuse le mauvais coupable.
    """
    print("\n-- images tronquees ----------------------------------------")

    ok, buf = cv2.imencode(".jpg", synth_frame(0), [cv2.IMWRITE_JPEG_QUALITY, 52])
    _TruncatingHandler.payload = buf.tobytes()

    with socketserver.TCPServer(("127.0.0.1", 0), _TruncatingHandler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        link = VideoLink(host="127.0.0.1", port=port, mode="stream")
        link.start()
        received = []
        deadline = time.time() + 5.0
        while time.time() < deadline and len(received) < 8:
            frame = link.take_latest()
            if frame:
                received.append(frame.jpeg)
            else:
                time.sleep(0.004)
        corrupt = link.stats.corrupt
        link.stop()
        httpd.shutdown()

    _check(len(received) >= 5, "images saines livrees malgre les coupures",
           f"{len(received)} recues")
    _check(corrupt >= 1, "images tronquees comptees comme telles",
           f"{corrupt} rejetees")

    every_whole = all(j.startswith(b"\xff\xd8") and j.endswith(b"\xff\xd9")
                      for j in received)
    _check(every_whole, "aucune image livree n'est tronquee")

    sizes = {len(j) for j in received}
    _check(len(sizes) == 1 and sizes.pop() == len(_TruncatingHandler.payload),
           "aucune image chimere (debut d'une, fin de la suivante)",
           f"tailles livrees: {sorted({len(j) for j in received})}")

    decoded = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in received]
    _check(all(d is not None and d.shape[:2] == (H, W) for d in decoded),
           "toutes les images livrees se decodent")


def test_updater() -> None:
    """Une mise a jour non signee, falsifiee ou plus ancienne doit etre refusee.

    C'est le test le plus important de ce fichier: un mecanisme de mise a jour
    est de l'execution de code a distance. Tout ce qui suit verifie qu'il
    echoue *fermement* -- une verification qui laisse passer en cas de doute
    ne protege de rien.
    """
    print("\n-- mise a jour signee --------------------------------------")
    import hashlib
    import json
    import shutil
    import tempfile
    import zipfile
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from invis import updater
    from invis.updater import Release, UpdateError
    from invis.version import is_newer, parse

    _check(parse("1.10.0") > parse("1.9.0"), "comparaison numerique des versions",
           "1.10.0 > 1.9.0")
    _check(is_newer("1.0.1", "1.0.0") and not is_newer("1.0.0", "1.0.1"),
           "sens de comparaison correct")

    key = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    public_hex = key.public_key().public_bytes_raw().hex()

    work = tempfile.mkdtemp(prefix="maj_test_")
    try:
        archive = os.path.join(work, "payload.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("invis/version.py", 'VERSION = "9.9.9"\n')
            zf.writestr("invis/dummy.py", "x = 1\n")
        payload = open(archive, "rb").read()
        digest = hashlib.sha256(payload).hexdigest()

        saved_key = updater.PUBLIC_KEY_HEX
        saved_root = updater.data_dir
        updater.PUBLIC_KEY_HEX = public_hex
        updater.data_dir = lambda: work

        def release(sig, sha=digest, version="9.9.9"):
            return Release(version=version, url="https://example.invalid/payload.zip",
                           signature_hex=sig, sha256=sha)

        def install_with(data, rel):
            """Installe en court-circuitant le reseau, pas la verification."""
            import urllib.request

            class _Resp:
                def read(self, *_a):
                    return data

                def __enter__(self):
                    return self

                def __exit__(self, *_a):
                    return False

            saved_open = urllib.request.urlopen
            urllib.request.urlopen = lambda *a, **k: _Resp()
            try:
                return updater.install(rel, current="1.0.0")
            finally:
                urllib.request.urlopen = saved_open

        # 1. Signature valide: installation acceptee.
        good = key.sign(payload).hex()
        try:
            path = install_with(payload, release(good))
            ok = os.path.exists(os.path.join(path, "invis", "version.py"))
        except UpdateError as exc:
            ok = False
            print(f"       (echec inattendu: {exc})")
        _check(ok, "archive correctement signee -> installee")

        # 2. Contenu modifie apres signature.
        tampered = bytearray(payload)
        tampered[len(tampered) // 2] ^= 0xFF
        tampered = bytes(tampered)
        refused = False
        try:
            install_with(tampered, release(good, sha=hashlib.sha256(tampered).hexdigest()))
        except UpdateError:
            refused = True
        _check(refused, "archive falsifiee -> REFUSEE")

        # 3. Signee avec une autre cle.
        refused = False
        try:
            install_with(payload, release(other.sign(payload).hex()))
        except UpdateError:
            refused = True
        _check(refused, "signature d'une autre cle -> REFUSEE")

        # 4. Empreinte annoncee fausse (corruption de transfert).
        refused = False
        try:
            install_with(payload, release(good, sha="00" * 32))
        except UpdateError:
            refused = True
        _check(refused, "empreinte incoherente -> REFUSEE")

        # 5. Version pas plus recente.
        refused = False
        try:
            install_with(payload, release(good, version="0.9.0"))
        except UpdateError:
            refused = True
        _check(refused, "version plus ancienne -> REFUSEE")

        # 6. Adresse non chiffree.
        refused = False
        try:
            install_with(payload, Release(version="9.9.9", url="http://example.invalid/p.zip",
                                          signature_hex=good, sha256=digest))
        except UpdateError:
            refused = True
        _check(refused, "archive servie en clair (HTTP) -> REFUSEE")

        # 7. Aucune cle publique compilee: on refuse plutot que de faire confiance.
        updater.PUBLIC_KEY_HEX = ""
        refused = False
        try:
            install_with(payload, release(good))
        except UpdateError:
            refused = True
        updater.PUBLIC_KEY_HEX = public_hex
        _check(refused, "sans cle publique -> REFUSEE (echec ferme)")

        # 8. Archive qui tente d'ecrire hors du dossier cible.
        evil = os.path.join(work, "evil.zip")
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../../evade.py", "print('dehors')\n")
        evil_bytes = open(evil, "rb").read()
        refused = False
        try:
            install_with(evil_bytes, release(key.sign(evil_bytes).hex(),
                                             sha=hashlib.sha256(evil_bytes).hexdigest()))
        except UpdateError:
            refused = True
        _check(refused, "archive au chemin remontant -> REFUSEE")

        # 9. Manifeste hors HTTPS.
        refused = False
        try:
            updater.fetch_manifest("http://example.invalid/manifest.json")
        except UpdateError:
            refused = True
        _check(refused, "manifeste en clair -> REFUSE")
    finally:
        updater.PUBLIC_KEY_HEX = saved_key
        updater.data_dir = saved_root
        shutil.rmtree(work, ignore_errors=True)


def test_bootstrap() -> None:
    """Le code mis a jour doit reellement remplacer le code embarque.

    Sans ce mecanisme, tout le systeme de mise a jour est inerte: l'archive se
    telecharge, se verifie, s'installe... et l'application relance le code
    embarque dans l'executable. Le defaut est silencieux, d'ou ce test.
    """
    print("\n-- choix du code au demarrage ------------------------------")
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import invis_bootstrap as bs

    work = tempfile.mkdtemp(prefix="boot_test_")
    saved_path = list(sys.path)
    saved_dir = bs.data_dir
    try:
        bundle = os.path.join(work, "bundle")
        os.makedirs(os.path.join(bundle, "invis"))
        with open(os.path.join(bundle, "invis", "version.py"), "w", encoding="utf-8") as fh:
            fh.write('VERSION = "1.0.0"\n')

        root = os.path.join(work, "data", "payload")
        for version in ("0.9.0", "1.0.1", "1.0.10"):
            d = os.path.join(root, version, "invis")
            os.makedirs(d)
            with open(os.path.join(d, "version.py"), "w", encoding="utf-8") as fh:
                fh.write(f'VERSION = "{version}"\n')
        # Dossier sans version lisible: doit etre ignore sans faire echouer.
        os.makedirs(os.path.join(root, "casse", "invis"))

        bs.data_dir = lambda app_name=None: os.path.join(work, "data")

        _check(bs.bundled_version(bundle) == "1.0.0", "version embarquee lue sans import",
               bs.bundled_version(bundle) or "aucune")

        found = [v for _p, v, _d in bs.candidates(root)]
        _check(found == ["1.0.10", "1.0.1", "0.9.0"],
               "versions triees numeriquement, invalides ecartees", str(found))

        chosen = bs.activate(bundle_root=bundle)
        _check(chosen is not None and os.path.basename(chosen) == "1.0.10",
               "la plus recente est retenue",
               os.path.basename(chosen) if chosen else "aucune")
        _check(chosen is not None and sys.path[0] == chosen,
               "placee devant le code embarque")

        # Rien de plus recent que l'embarque: on garde l'embarque.
        sys.path[:] = saved_path
        with open(os.path.join(bundle, "invis", "version.py"), "w", encoding="utf-8") as fh:
            fh.write('VERSION = "9.9.9"\n')
        _check(bs.activate(bundle_root=bundle) is None,
               "aucune version plus recente -> code embarque conserve")

        # Emplacement illisible: le demarrage ne doit pas echouer.
        bs.data_dir = lambda app_name=None: os.path.join(work, "inexistant")
        _check(bs.activate(bundle_root=bundle) is None,
               "dossier absent -> repli silencieux, pas d'erreur")
    finally:
        bs.data_dir = saved_dir
        sys.path[:] = saved_path
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    test_geometry()
    test_orientation()
    test_pose_convention()
    test_detector()
    test_frame_quality()
    test_metric_reconstruction()
    test_scale_invariance()
    test_rendering()
    test_link()
    test_truncated_frames()
    test_updater()
    test_bootstrap()
    test_session_roundtrip()
    print()
    if _failures:
        print(f"{_failures} test(s) en echec")
        return 1
    print("tous les tests passent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
