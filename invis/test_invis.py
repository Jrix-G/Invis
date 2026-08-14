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
import json
import math
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


class _CountingServer(socketserver.ThreadingTCPServer):
    """Serveur qui compte les sockets acceptees et les acces simultanes.

    C'est la seule mesure qui compte pour ce test: pas le nombre de requetes
    HTTP, mais le nombre de connexions TCP reellement ouvertes.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opened = 0
        self.live = 0
        self.peak = 0
        self._count_lock = threading.Lock()

    def get_request(self):
        sock, addr = super().get_request()
        with self._count_lock:
            self.opened += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
        return sock, addr

    def shutdown_request(self, request) -> None:
        with self._count_lock:
            self.live -= 1
        super().shutdown_request(request)

    def handle_error(self, request, client_address) -> None:
        # Fermer une socket persistante fait crier le serveur de test. C'est
        # le comportement attendu ici, pas une erreur a afficher.
        pass


class _BoardHandler(http.server.BaseHTTPRequestHandler):
    """Carte simulee: /status annonce l'etat camera, /stream sert le flux."""

    protocol_version = "HTTP/1.1"      # sans quoi keep-alive est impossible
    payload = b""
    camera_enabled = True
    stream_requests = 0

    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/status"):
            body = json.dumps({"camera_enabled": type(self).camera_enabled}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        type(self).stream_requests += 1
        if not type(self).camera_enabled:
            body = b"camera off"
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(self.payload)).encode() + b"\r\n\r\n"
                                 + self.payload + b"\r\n")
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def test_socket_budget() -> None:
    """Invis ne doit jamais deborder du budget de sockets de la carte.

    Ce test ne parle pas de video: il parle de securite de vol. La carte sert
    le flux et le lien de pilotage sur le meme serveur HTTP, avec un pool de
    sept sockets purge en LRU. Le WebSocket /pilot est persistant mais
    silencieux entre deux commandes: c'est donc lui que la carte sacrifie
    quand le pool sature. La page de pilotage retombe alors sur le repli HTTP,
    l'ecart entre deux commandes depasse la seconde, et le firmware latche
    LAND. Autrement dit, une station sol trop bavarde rend le drone
    impilotable -- ce qui s'est produit.

    Deux comportements sont donc verifies, et un troisieme par omission:
    les sockets sont reutilisees, jamais plus de MAX_ESP_SOCKETS a la fois,
    et une camera annoncee eteinte fait cesser les demandes de flux.
    """
    print("\n-- budget de sockets vers la carte --------------------------")
    from invis import config
    from invis.mjpeg_client import http_get

    ok, buf = cv2.imencode(".jpg", synth_frame(0), [cv2.IMWRITE_JPEG_QUALITY, 52])
    _BoardHandler.payload = buf.tobytes()
    _BoardHandler.camera_enabled = True
    _BoardHandler.stream_requests = 0

    with _CountingServer(("127.0.0.1", 0), _BoardHandler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # 1. Flux video et interrogations de /status en parallele, depuis
        #    plusieurs fils: le cas ou une implementation naive explose.
        link = VideoLink(host="127.0.0.1", port=port, mode="stream")
        link.start()
        stop = threading.Event()

        def hammer() -> None:
            while not stop.is_set():
                try:
                    http_get("127.0.0.1", config.STATUS_PATH, port=port)
                except Exception:  # noqa: BLE001
                    pass

        workers = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        for w in workers:
            w.start()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            link.take_latest()
            time.sleep(0.01)
        stop.set()
        for w in workers:
            w.join(timeout=2.0)
        frames = link.stats.frames
        busy_peak = httpd.peak
        busy_opened = httpd.opened
        link.stop()

        _check(busy_peak <= config.MAX_ESP_SOCKETS,
               f"jamais plus de {config.MAX_ESP_SOCKETS} sockets simultanees",
               f"maximum observe {busy_peak}")
        # Sans connexions persistantes, quelques secondes de ce regime ouvrent
        # des milliers de sockets: c'est exactement ce qui purgeait /pilot.
        _check(busy_opened <= 4 * config.MAX_ESP_SOCKETS,
               "sockets reutilisees et non rouvertes a chaque appel",
               f"{busy_opened} ouvertes pour {frames} images et un martelage de /status")
        _check(frames > 0, "le flux passe malgre le plafond de sockets",
               f"{frames} images")

        # 2. Camera annoncee eteinte: plus aucune demande de flux.
        _BoardHandler.camera_enabled = False
        _BoardHandler.stream_requests = 0
        link = VideoLink(host="127.0.0.1", port=port, mode="stream")
        link.start()
        time.sleep(max(3.0, config.CAMERA_OFF_POLL_S + 1.0))
        asked = _BoardHandler.stream_requests
        link.stop()
        httpd.shutdown()

    _check(asked == 0, "camera eteinte -> plus aucune demande de /stream",
           f"{asked} demandes")
    _check(min(config.RECONNECT_BACKOFF_S) >= 1.0,
           "plancher de reprise d'au moins 1 s",
           f"{config.RECONNECT_BACKOFF_S}")
    _check(config.STATUS_POLL_S >= 2.0, "pas de sondage /status plus rapide que 2 s",
           f"{config.STATUS_POLL_S}s")

    # 3. Contrepartie des connexions persistantes: la carte peut fermer une
    #    socket inactive sans l'annoncer -- c'est meme precisement ce que fait
    #    sa purge LRU. On ne l'apprend qu'en ecrivant dedans. Sans reprise,
    #    reutiliser une socket serait donc *moins* fiable que d'en rouvrir
    #    une, et le remede serait pire que le mal.
    class _SilentCloser(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.recv(4096)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            time.sleep(0.05)
            self.request.close()      # keep-alive annonce, socket fermee quand meme

    class _Threaded(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

        def handle_error(self, request, client_address) -> None:
            pass

    with _Threaded(("127.0.0.1", 0), _SilentCloser) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        replies = []
        for _ in range(4):
            try:
                replies.append(http_get("127.0.0.1", config.STATUS_PATH, port=port))
            except Exception as exc:  # noqa: BLE001
                replies.append(f"echec: {exc}")
            time.sleep(0.2)
        httpd.shutdown()

    _check(all(r == "ok" for r in replies),
           "socket fermee en douce par la carte -> appel suivant rejoue",
           str(replies))


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


def test_fusion() -> None:
    """Le filtre doit lisser le bruit, rejeter l'aberrant, arreter la derive."""
    print("\n-- filtrage de la pose --------------------------------------")
    from invis.fusion import ConstantVelocity, HeadingFilter, wrap_angle

    rng = np.random.default_rng(11)
    dt, truth_v = 0.15, np.array([0.8, 0.0])

    # 1) Bruit: la vitesse filtree doit etre plus proche du vrai que la mesure.
    kf = ConstantVelocity(dim=2, accel_sigma=1.5, meas_sigma=0.30)
    raw_err, filt_err = [], []
    for _ in range(120):
        z = truth_v + rng.normal(0.0, 0.30, size=2)
        kf.predict(dt)
        kf.update_velocity(z)
        raw_err.append(np.linalg.norm(z - truth_v))
        filt_err.append(np.linalg.norm(kf.velocity - truth_v))
    gain = float(np.mean(raw_err) / max(1e-9, np.mean(filt_err)))
    # Le gain est borne par le bruit de modele, et c'est voulu: avec une
    # acceleration possible de 1,5 m/s^2 sur 0,15 s, le modele lui-meme laisse
    # filer 0,22 m/s entre deux images, contre 0,30 m/s de bruit de mesure. Le
    # filtre moyenne donc environ trois mesures, soit un facteur racine de
    # trois. Exiger davantage reviendrait a pretendre que le drone
    # n'accelere pas -- le filtre suivrait alors mal les vraies variations.
    _check(gain > 1.5, "bruit de vitesse reduit par le filtre",
           f"erreur divisee par {gain:.1f} (borne modele ~1.7)")

    # 2) Aberrant: une mesure incompatible est ecartee, pas moyennee.
    accepted = kf.update_velocity(np.array([12.0, -9.0]))
    _check(not accepted, "mesure aberrante rejetee par le test de compatibilite")
    _check(np.linalg.norm(kf.velocity - truth_v) < 0.2,
           "l'etat survit intact au rejet", f"v={np.round(kf.velocity, 3)}")

    # 3) Stationnaire: la vitesse nulle observee doit stopper l'etat.
    for _ in range(12):
        kf.predict(dt)
        kf.update_velocity(np.zeros(2), sigma=0.10)
    _check(kf.speed < 0.08, "vitesse nulle observee -> etat immobilise",
           f"{kf.speed:.3f} m/s")

    # 4) L'incertitude de position doit croitre: rien ne l'observe.
    free = ConstantVelocity(dim=2)
    before = free.position_sigma
    for _ in range(80):
        free.predict(dt)
    _check(free.position_sigma > before + 0.5,
           "incertitude de position croissante sans recalage",
           f"{before:.3f} -> {free.position_sigma:.2f} m")

    # 5) Cap: le repliement doit etre correct au passage de +/-180 degres.
    heading = HeadingFilter()
    heading.set_yaw(math.radians(179.0))
    for _ in range(20):
        heading.predict(dt)
        heading.update_delta(math.radians(1.0), dt)
    _check(abs(heading.yaw_rad) > math.radians(150.0),
           "cap replie correctement au passage de 180 degres",
           f"{math.degrees(heading.yaw_rad):+.1f} deg")
    _check(abs(wrap_angle(math.radians(370.0)) - math.radians(10.0)) < 1e-9,
           "repliement d'angle exact")


def test_structure() -> None:
    """Intersection de visees: exacte sans bruit, incertitude coherente."""
    print("\n-- structure par visees multiples ---------------------------")
    from invis import geometry
    from invis.geometry import Intrinsics
    from invis.structure import RayBundle

    K = Intrinsics.from_fov(W, H)
    tilt, height = -45.0, 2.5
    R_bc = geometry.body_from_camera(tilt, 0.0)
    truth = np.array([[6.0, 0.0, 1.4], [6.0, 0.5, 0.9], [5.0, -0.3, 0.6]])
    ids = np.arange(len(truth))

    def fly(bundle, steps, noise_px=0.0, rng=None):
        out = None
        for k in range(steps):
            pos = np.array([0.09 * k, 0.0, height])
            cam = (truth - pos) @ R_bc
            uv = np.column_stack([cam[:, 0] / cam[:, 2] * K.fx + K.cx,
                                  cam[:, 1] / cam[:, 2] * K.fy + K.cy])
            if noise_px and rng is not None:
                uv = uv + rng.normal(0.0, noise_px, size=uv.shape)
            rays = geometry.rays_body(uv, K, R_bc)
            rows = bundle.observe(ids, pos, rays)
            out = bundle.solve(rows, min_views=3, max_sigma_m=1e9, focal_px=K.fx)
        return out

    X, sigma, _ = fly(RayBundle(capacity=32, sigma_px=0.6), 10)
    err = np.linalg.norm(X - truth, axis=1)
    _check(float(err.max()) < 1e-6, "visees exactes -> point exact",
           f"erreur max {err.max():.2e} m")

    # L'incertitude doit decroitre quand la base s'allonge, et le faire de
    # facon monotone: c'est ce qui la rend utilisable comme critere.
    short = fly(RayBundle(capacity=32, sigma_px=0.6), 4)[1]
    long_ = fly(RayBundle(capacity=32, sigma_px=0.6), 14)[1]
    _check(bool(np.all(long_ < short)), "incertitude decroissante avec la base",
           f"{np.round(short, 3)} -> {np.round(long_, 3)}")

    # Avec du bruit de pointage, l'erreur reelle doit rester du meme ordre que
    # l'incertitude annoncee. Une incertitude qui ne predit pas l'erreur ne
    # sert a rien -- pire, elle donne une confiance imméritee.
    rng = np.random.default_rng(5)
    Xn, sn, _ = fly(RayBundle(capacity=32, sigma_px=0.6), 14, noise_px=0.6, rng=rng)
    real = np.linalg.norm(Xn - truth, axis=1)
    _check(bool(np.all(real < 3.0 * sn)), "erreur reelle compatible avec l'incertitude",
           f"erreur {np.round(real, 3)} contre sigma {np.round(sn, 3)}")

    # Deux visees quasi confondues ne doivent rien produire de credible.
    still = RayBundle(capacity=32, sigma_px=0.6)
    for _ in range(6):
        pos = np.array([0.0, 0.0, height])
        cam = (truth - pos) @ R_bc
        uv = np.column_stack([cam[:, 0] / cam[:, 2] * K.fx + K.cx,
                              cam[:, 1] / cam[:, 2] * K.fy + K.cy])
        rows = still.observe(ids, pos, geometry.rays_body(uv, K, R_bc))
    _, _, mask = still.solve(rows, min_views=3, max_sigma_m=0.6, focal_px=K.fx)
    _check(not mask.any(), "aucune base -> aucun point accepte",
           f"{int(mask.sum())} points retenus")


def test_imu_fusion() -> None:
    """Une centrale inertielle, si elle existe, doit freiner la derive de cap.

    Le cap est la seule grandeur que rien n'observe dans ce systeme: il
    s'integre, donc il derive, et aucune image ne le recale. C'est en virage
    que cela se voit -- le lacet visuel accumule alors son erreur a chaque
    image. Le test compare le meme vol avec et sans gyrometre.
    """
    print("\n-- fusion inertielle ----------------------------------------")
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    def fly(with_imu: bool):
        sim = FlightSimulator(height_m=2.0, speed_mps=0.8, yaw_rate_dps=12.0,
                              walls=[Wall(x_m=6.0, width_m=1.6, height_m=1.4)])
        det = ObstacleDetector()
        mapper = Mapper(height_m=2.0)
        rng = np.random.default_rng(17)
        errors = []
        for k in range(int(6.0 * FPS_SIM)):
            t = k / FPS_SIM
            img = sim.render(t)
            if with_imu:
                # Gyrometre realiste: bruite et biaise, pas parfait.
                mapper.push_imu(sim.imu(t, rng=rng, gyro_sigma_dps=0.8,
                                        gyro_bias_dps=0.3, attitude=False))
            frame = mapper.update(det.process(img, t), img)
            errors.append(abs(frame.yaw_deg - math.degrees(sim.yaw_rad(t))))
        return float(np.mean(errors[-10:]))

    without = fly(False)
    with_ = fly(True)
    _check(with_ < without, "le gyrometre reduit l'erreur de cap",
           f"{without:.2f} deg sans -> {with_:.2f} deg avec")
    _check(with_ < 3.0, "cap tenu en virage avec gyrometre", f"{with_:.2f} deg")

    # L'assiette inertielle doit rendre la reconstruction insensible a
    # l'obstacle qui remplit l'image -- la faiblesse corrigee par ailleurs a
    # coups de seuils devient ici sans objet.
    sim = FlightSimulator(height_m=2.5, speed_mps=0.6,
                          walls=[Wall(x_m=6.0, width_m=1.6, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.5)
    drift = 0.0
    for k in range(int((5.5 / 0.6) * FPS_SIM)):
        t = k / FPS_SIM
        img = sim.render(t)
        mapper.push_imu(sim.imu(t, attitude=True))
        frame = mapper.update(det.process(img, t), img)
        drift = max(drift, abs(frame.tilt_deg - sim.tilt_deg))
    _check(drift < 0.01, "assiette inertielle: aucune derive de tangage",
           f"ecart max {drift:.4f} deg")


def test_loop_closure() -> None:
    """Repasser au meme endroit doit recaler la position, pas la degrader.

    Le vol est un cercle complet: le drone revient exactement a son point de
    depart, ce que l'odometrie seule ignore. C'est la seule situation ou une
    information nouvelle apparait sans nouveau capteur.
    """
    print("\n-- fermeture de boucle --------------------------------------")
    from invis import config as cfg
    from invis.loop import LoopCloser, descriptor, ground_patch
    from invis.geometry import Intrinsics
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator

    # 1) La vignette de sol doit etre la meme quel que soit le cap: c'est la
    #    propriete sur laquelle repose toute la reconnaissance.
    K = Intrinsics.from_fov(W, H)
    rng = np.random.default_rng(4)
    scene = cv2.GaussianBlur(rng.integers(0, 255, size=(H, W), dtype=np.uint8), (5, 5), 0)
    p0 = ground_patch(scene, K, 2.0, -45.0, 0.0, 0.0)
    _check(p0 is not None and float(p0.std()) > 5.0,
           "vignette de sol construite", f"ecart-type {p0.std():.1f}" if p0 is not None else "None")
    _check(descriptor(p0) is not None, "descripteur extrait de la vignette")

    # 2) Recalage sur decalage connu, dans les deux axes.
    closer = LoopCloser()
    res = cfg.LOOP_PATCH_SPAN_M / cfg.LOOP_PATCH_PX
    big = cv2.GaussianBlur(rng.integers(0, 255, size=(400, 400), dtype=np.uint8), (5, 5), 0)
    N = cfg.LOOP_PATCH_PX

    def crop(px, py):
        x0 = int(round(200 + px / res - N / 2))
        y0 = int(round(200 + py / res - N / 2))
        return big[y0:y0 + N, x0:x0 + N].copy()

    errs = []
    for dx, dy in ((0.5, 0.0), (0.0, 0.5), (-0.75, 0.25), (1.0, -1.0)):
        out = closer._align(crop(0.0, 0.0), crop(dx, dy))
        if out is None:
            errs.append(float("inf"))
            continue
        errs.append(float(np.linalg.norm(out[0] - np.array([dx, dy]))))
    _check(max(errs) < 0.10, "decalage retrouve dans les deux axes",
           f"erreur max {max(errs):.3f} m sur {len(errs)} essais")

    # 3) Vol circulaire complet: le drone revient sur ses traces.
    def fly(enabled: bool) -> float:
        saved, cfg.LOOP_ENABLED = cfg.LOOP_ENABLED, enabled
        try:
            sim = FlightSimulator(height_m=2.0, speed_mps=0.8, yaw_rate_dps=30.0, walls=[])
            det = ObstacleDetector()
            mapper = Mapper(height_m=2.0)
            turn_s = 360.0 / 30.0
            for k in range(int(turn_s * FPS_SIM * 1.05)):
                t = k / FPS_SIM
                mapper.update(det.process(sim.render(t), t))
            truth = sim.truth(t)
            error = float(np.hypot(mapper._pos[0] - truth.camera_x,
                                   mapper._pos[1] - truth.camera_y))
            return error, mapper.loop_matches, len(mapper._closer)
        finally:
            cfg.LOOP_ENABLED = saved

    without, _, _ = fly(False)
    with_, matches, places = fly(True)
    _check(places > 10, "lieux memorises le long du parcours", f"{places} cles")
    _check(matches > 0, "boucle effectivement reconnue au retour",
           f"{matches} fermetures acceptees par le filtre")
    _check(with_ <= without, "la fermeture ne degrade jamais la position",
           f"{without:.2f} m sans -> {with_:.2f} m avec")


def test_attitude_robustness() -> None:
    """L'assiette ne doit pas suivre l'obstacle qui grandit dans l'image.

    Un mur encore lointain se plie a l'homographie du sol -- sa parallaxe est
    trop faible pour l'en distinguer -- tout en tirant la normale ajustee vers
    lui. Le tangage derivait ainsi de plusieurs degres pendant l'approche, ce
    qui se payait en dizaines de centimetres sur la distance annoncee.
    """
    print("\n-- stabilite de l'assiette ----------------------------------")
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    sim = FlightSimulator(height_m=2.5, speed_mps=0.6,
                          walls=[Wall(x_m=6.0, width_m=1.6, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.5)
    tilts = []
    for k in range(int((5.5 / 0.6) * FPS_SIM)):
        t = k / FPS_SIM
        img = sim.render(t)
        tilts.append(mapper.update(det.process(img, t), img).tilt_deg)

    drift = max(abs(v - sim.tilt_deg) for v in tilts)
    _check(drift < 1.5, "tangage stable pendant toute l'approche",
           f"ecart max {drift:.2f} deg (mesure a 4.5 deg avant durcissement)")


def test_clusters_and_alert() -> None:
    """Deux obstacles separes doivent se compter comme deux, avec leur distance.

    La grille 3x3 disait dans quelle case de l'image il y avait du relief. Ce
    n'est ni un objet ni une distance: deux poteaux de part et d'autre du
    couloir donnaient la meme lecture qu'un mur en travers.
    """
    print("\n-- objets distincts et alerte -------------------------------")
    from invis.mapper import Mapper
    from invis.simulator import FlightSimulator, Wall

    # Les deux poteaux sont places dans la portee reelle du montage: a 45
    # degres de piquage la camera ne voit le sol que jusqu'a environ 2,2 fois
    # la hauteur de vol, soit 4,4 m ici. Plus loin, il n'y a rien a detecter.
    sim = FlightSimulator(height_m=2.0, speed_mps=0.8, walls=[
        Wall(x_m=4.0, y_m=-1.2, width_m=0.6, height_m=1.3),
        Wall(x_m=4.0, y_m=+1.2, width_m=0.6, height_m=1.3),
    ])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.0)
    seen_two = 0
    frames = 0
    forward_always = True
    for k in range(int((3.4 / 0.8) * FPS_SIM)):
        t = k / FPS_SIM
        img = sim.render(t)
        frame = mapper.update(det.process(img, t), img)
        if frame.clusters:
            frames += 1
            if len(frame.clusters) >= 2:
                seen_two += 1
        if frame.ok and not math.isfinite(frame.forward_m):
            forward_always = False

    _check(frames > 0, "des objets sont detectes", f"{frames} images avec objets")
    _check(seen_two >= max(1, frames // 4), "les deux poteaux comptent pour deux",
           f"{seen_two}/{frames} images voient au moins deux objets")
    _check(forward_always, "la distance droit devant est toujours renseignee",
           "sol a defaut d'obstacle")

    # Un mur en travers doit, lui, declencher l'alerte de couloir.
    sim = FlightSimulator(height_m=2.0, speed_mps=0.8,
                          walls=[Wall(x_m=4.0, width_m=2.4, height_m=1.4)])
    det = ObstacleDetector()
    mapper = Mapper(height_m=2.0)
    levels, corridor = [], 0
    for k in range(int((3.4 / 0.8) * FPS_SIM)):
        t = k / FPS_SIM
        frame = mapper.update(det.process(sim.render(t), t))
        levels.append(frame.alert_level)
        corridor += any(c.in_corridor for c in frame.clusters)
    _check(corridor > 0, "mur en travers reconnu comme coupant le couloir",
           f"{corridor} images")
    _check(max(levels) >= 2, "alerte de danger levee a l'approche",
           f"niveau max {max(levels)}")
    _check(levels[0] == 0, "aucune alerte au depart, loin du mur")


def test_ui_threading() -> None:
    """Le fil d'analyse ne doit jamais appeler Tcl.

    L'interpreteur Tcl derriere Tkinter n'est pas concu pour etre appele depuis
    plusieurs fils. Lire une variable Tk depuis le fil d'analyse ressemble a une
    lecture anodine mais c'est un appel dans Tcl: selon le moment il rend la
    bonne valeur, il leve "main thread is not in main loop", ou il corrompt
    l'etat de l'interpreteur. Cela fonctionnait par chance, l'appel tombant
    presque toujours pendant que le fil principal attendait.

    Le test rend la faute systematique: une boucle `update()` sans `mainloop()`
    garde le fil principal hors de la boucle Tcl, et toute lecture depuis le
    fil d'analyse echoue alors a coup sur. Avant correction, le fil mourait en
    quelques dixiemes de seconde.
    """
    print("\n-- cloisonnement des fils de l'interface --------------------")
    try:
        import tkinter
        tkinter.Tk().destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"IGNORE cloisonnement des fils              (Tk indisponible: {exc})")
        return

    import threading
    from invis.gcs_vision import VisionApp

    faults = []
    previous = threading.excepthook
    threading.excepthook = lambda args: faults.append(args.exc_value) or previous(args)
    try:
        app = VisionApp(host="127.0.0.1", source="sim")
        app.withdraw()
        app.update()
        app._connect()
        deadline = time.time() + 4.0
        while time.time() < deadline:
            app.update()
            time.sleep(0.02)
        alive = bool(app._worker and app._worker.is_alive())
        mapframe = app._last_mapframe
        app._disconnect()
        app.destroy()
    finally:
        threading.excepthook = previous

    _check(not faults, "aucune exception dans le fil d'analyse",
           repr(faults[0]) if faults else "")
    _check(alive, "le fil d'analyse survit a la boucle d'affichage")
    _check(mapframe is not None and mapframe.n_cells > 0,
           "la reconstruction avance malgre le cloisonnement",
           f"{mapframe.n_cells} cases" if mapframe else "aucune image analysee")


def test_elevation_grid() -> None:
    """La carte d'elevation doit moyenner, se recentrer et rester bornee."""
    print("\n-- carte d'elevation ----------------------------------------")
    from invis import config as cfg
    from invis.grid import ElevationGrid

    grid = ElevationGrid(cells=64, resolution_m=0.10)
    grid.recentre(np.zeros(2))

    # 1) Hauteur et couleur retrouvees, bruit moyenne.
    rng = np.random.default_rng(21)
    target_z, target_c = 0.7, np.array([40.0, 90.0, 200.0])
    for _ in range(20):
        pts = np.zeros((50, 3))
        pts[:, 0] = 0.15
        pts[:, 1] = -0.25
        pts[:, 2] = target_z + rng.normal(0.0, 0.05, size=50)
        cols = np.clip(target_c + rng.normal(0.0, 20.0, size=(50, 3)), 0, 255)
        grid.add(pts, 1.0, cols)

    got = grid.height_at(np.array([[0.15, -0.25]]))[0]
    _check(abs(got - target_z) < 0.02, "hauteur moyennee sur les mesures repetees",
           f"{got:.3f} m pour {target_z} m")
    centres, colours, shade = grid.surface(min_count=1)
    _check(len(centres) == 1, "une seule case renseignee", f"{len(centres)} cases")
    _check(float(np.abs(colours[0] - target_c).max()) < 12.0,
           "couleur moyennee sur les mesures repetees", f"{np.round(colours[0], 1)}")

    # 2) Une case jamais vue ne repond pas: l'ignorance doit se dire.
    unseen = grid.height_at(np.array([[2.0, 2.0]]))[0]
    _check(np.isnan(unseen), "case jamais observee -> pas de hauteur inventee")

    # 3) Recentrage: le contenu doit designer le meme terrain apres decalage.
    moved = grid.recentre(np.array([2.5, 0.0]))
    still = grid.height_at(np.array([[0.15, -0.25]]))[0]
    _check(moved, "recentrage declenche par l'eloignement")
    _check(abs(still - target_z) < 0.02,
           "le terrain reste au meme endroit apres recentrage",
           f"{still:.3f} m" if np.isfinite(still) else "perdu")

    # 4) Ce qui entre par un bord doit etre vide, pas recopie de l'autre bord.
    behind = grid.height_at(np.array([[5.4, 0.0]]))[0]
    _check(np.isnan(behind), "le terrain ne reapparait pas par le bord oppose",
           "NaN" if np.isnan(behind) else f"{behind:.3f}")

    # 5) Memoire bornee: repasser cent fois n'ajoute pas de case.
    before = len(grid)
    for _ in range(100):
        grid.add(np.array([[2.5, 0.05, 0.3]]), 2.0)
    _check(len(grid) == before + 1, "memoire bornee par le terrain, pas par la duree",
           f"{before} -> {len(grid)} cases")

    # 6) Ombrage: un plan doit s'eclairer uniformement, une pente non.
    flat = ElevationGrid(cells=32, resolution_m=0.10)
    flat.recentre(np.zeros(2))
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    flat.add(plane, 1.0)
    _, _, sh_flat = flat.surface(min_count=1)

    slope = ElevationGrid(cells=32, resolution_m=0.10)
    slope.recentre(np.zeros(2))
    ramp = plane.copy()
    ramp[:, 2] = 0.5 * ramp[:, 0]
    slope.add(ramp, 1.0)
    _, _, sh_slope = slope.surface(min_count=1)
    _check(float(sh_flat.std()) < 1e-3 < float(np.abs(sh_slope.mean() - sh_flat.mean())),
           "l'ombrage ne reagit qu'au relief",
           f"plat sigma={sh_flat.std():.4f}, pente ecart={abs(sh_slope.mean()-sh_flat.mean()):.3f}")


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
    test_fusion()
    test_structure()
    test_imu_fusion()
    test_loop_closure()
    test_elevation_grid()
    test_clusters_and_alert()
    test_ui_threading()
    test_attitude_robustness()
    test_metric_reconstruction()
    test_scale_invariance()
    test_rendering()
    test_link()
    test_truncated_frames()
    test_socket_budget()
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
