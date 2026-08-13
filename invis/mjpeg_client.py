"""Client MJPEG / snapshot pour le pont ESP32-CAM.

Le firmware ne sert qu'un flux video a la fois et n'a aucune marge pour
accumuler du retard. Le client suit donc une regle simple: il ne conserve
jamais qu'une seule image, la plus recente. Une image qui n'a pas ete
consommee a temps est perdue volontairement, pas mise en file.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def _parse_boundary(content_type: str) -> Optional[bytes]:
    """Extrait la borne annoncee par l'en-tete multipart."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            value = part[len("boundary="):].strip().strip('"')
            if value:
                return b"--" + value.encode("ascii", "replace")
    return None


def _content_length(headers: bytes) -> Optional[int]:
    """Taille annoncee d'une partie, ou None si l'en-tete ne la donne pas."""
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                return int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    return None


@dataclass
class Frame:
    """Une image JPEG brute et l'instant ou elle a fini d'arriver."""

    jpeg: bytes
    recv_time: float
    index: int


@dataclass
class LinkStats:
    """Compteurs du lien video, lus par l'interface."""

    frames: int = 0
    dropped: int = 0
    bytes_total: int = 0
    fps: float = 0.0
    kbps: float = 0.0
    last_error: str = ""
    connected: bool = False
    reconnects: int = 0
    corrupt: int = 0
    _window: list = field(default_factory=list, repr=False)


class VideoLink:
    """Thread reseau: telecharge le flux et expose la derniere image.

    Deux modes:
      - "stream": multipart/x-mixed-replace sur /stream, le mode nominal;
      - "snapshot": interrogation repetee de /jpg, repli quand le flux
        multipart se coupe sans arret (client fantome, Wi-Fi qui decroche).
    """

    def __init__(
        self,
        host: str = config.DEFAULT_HOST,
        port: int = config.DEFAULT_PORT,
        mode: str = "stream",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.mode = mode
        self._on_log = on_log or (lambda msg: None)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._counter = 0
        self.stats = LinkStats()

    # -- cycle de vie ------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats = LinkStats()
        self._thread = threading.Thread(target=self._run, name="video-link", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        self.stats.connected = False
        with self._lock:
            self._latest = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- acces aux images --------------------------------------------------

    def take_latest(self) -> Optional[Frame]:
        """Retire et renvoie la derniere image, ou None si rien de neuf."""
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def _publish(self, jpeg: bytes) -> None:
        now = time.time()
        self._counter += 1
        with self._lock:
            if self._latest is not None:
                self.stats.dropped += 1
            self._latest = Frame(jpeg=jpeg, recv_time=now, index=self._counter)
        self.stats.frames += 1
        self.stats.bytes_total += len(jpeg)
        self._tick_rate(now, len(jpeg))

    def _tick_rate(self, now: float, nbytes: int) -> None:
        window = self.stats._window
        window.append((now, nbytes))
        cutoff = now - 2.0
        while window and window[0][0] < cutoff:
            window.pop(0)
        if len(window) >= 2:
            span = window[-1][0] - window[0][0]
            if span > 0:
                self.stats.fps = (len(window) - 1) / span
                self.stats.kbps = sum(b for _, b in window[1:]) / span / 1024.0

    # -- URLs --------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    # -- boucle ------------------------------------------------------------

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            started = time.time()
            delivered = self.stats.frames
            try:
                if self.mode == "snapshot":
                    self._pump_snapshots()
                else:
                    self._pump_stream()
                attempt = 0
            except Exception as exc:  # noqa: BLE001 - on veut vraiment tout rattraper
                if self._stop.is_set():
                    break
                self.stats.connected = False
                self.stats.last_error = f"{type(exc).__name__}: {exc}"

                # Un flux qui a reellement fonctionne remet le compteur a zero.
                #
                # Sans cela le decompte ne redescendait jamais: un flux qui
                # marche ne rend pas la main, il ne fait que lever une
                # exception en mourant. Trois coupures suffisaient donc a
                # coller l'attente au maximum, definitivement -- et la carte
                # coupant le flux toutes les quelques secondes, l'attente
                # devenait plus longue que le flux lui-meme. C'etait la cause
                # des saccades observees, cote client et non cote carte.
                lived = time.time() - started
                if lived >= config.RECONNECT_STABLE_S and self.stats.frames > delivered:
                    attempt = 0

                delay = config.RECONNECT_BACKOFF_S[min(attempt, len(config.RECONNECT_BACKOFF_S) - 1)]
                self.stats.reconnects += 1
                self._on_log(f"lien perdu ({self.stats.last_error}), "
                             f"reprise dans {delay:.2f}s (flux tenu {lived:.1f}s)")
                attempt += 1
                self._stop.wait(delay)
        self.stats.connected = False

    def _pump_stream(self) -> None:
        url = self._url(config.STREAM_PATH)
        self._on_log(f"connexion {url}")
        req = urllib.request.Request(url, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=config.CONNECT_TIMEOUT_S) as resp:
            self.stats.connected = True
            self.stats.last_error = ""
            boundary = _parse_boundary(resp.headers.get("Content-Type", ""))
            self._on_log(f"flux ouvert (bornes {boundary.decode(errors='replace')})"
                         if boundary else "flux ouvert (sans bornes annoncees)")
            buffer = bytearray()
            while not self._stop.is_set():
                chunk = resp.read(4096)
                if not chunk:
                    raise ConnectionError("flux ferme par la carte")
                buffer.extend(chunk)
                if boundary:
                    self._drain_multipart(buffer, boundary)
                else:
                    self._drain_jpegs(buffer)

    def _drain_multipart(self, buffer: bytearray, boundary: bytes) -> None:
        """Decoupe le flux en s'appuyant sur les en-tetes de chaque partie.

        Pourquoi ne pas se contenter de chercher les marqueurs JPEG: si une
        image arrive tronquee -- ce qui se produit reellement sur cette carte,
        le decodeur signalant "premature end of data segment" -- la recherche
        par marqueurs attrape le debut de l'image coupee puis la *fin de la
        suivante*, et fabrique une image chimere. Celle-ci se decode en un
        aplat gris, que l'analyse prend pour une image floue et non pour une
        image perdue. Deux images sont ainsi gachees au lieu d'une, et le
        diagnostic est fausse.

        L'en-tete Content-Length envoye par le firmware donne la taille exacte:
        on sait alors si l'image est complete, et on jette proprement celles
        qui ne le sont pas.
        """
        while True:
            start = buffer.find(boundary)
            if start < 0:
                if len(buffer) > config.MAX_JPEG_BYTES:
                    del buffer[:-len(boundary)]
                return

            head_end = buffer.find(b"\r\n\r\n", start)
            if head_end < 0:
                if start > 0:
                    del buffer[:start]
                return
            head_end += 4

            headers = bytes(buffer[start:head_end])
            length = _content_length(headers)
            if length is None:
                # Partie sans taille annoncee: repli sur les marqueurs, en se
                # limitant a ce qui suit l'en-tete.
                del buffer[:head_end]
                self._drain_jpegs(buffer)
                return
            if length <= 0 or length > config.MAX_JPEG_BYTES:
                self.stats.corrupt += 1
                del buffer[:head_end]
                continue
            if len(buffer) < head_end + length:
                if start > 0:
                    del buffer[:start]
                return

            payload = bytes(buffer[head_end:head_end + length])
            del buffer[:head_end + length]

            if payload.startswith(SOI) and payload.endswith(EOI):
                self._publish(payload)
            else:
                # Taille annoncee tenue mais contenu non conforme: image
                # abimee en vol, on la jette au lieu de la faire analyser.
                self.stats.corrupt += 1

    def _drain_jpegs(self, buffer: bytearray) -> None:
        """Extrait toutes les images completes presentes dans le tampon.

        On ignore volontairement les en-tetes de partie multipart: chercher
        SOI/EOI marche quel que soit le format exact des bornes, et resiste a
        une bordure tronquee.
        """
        while True:
            start = buffer.find(SOI)
            if start < 0:
                if len(buffer) > config.MAX_JPEG_BYTES:
                    del buffer[:-2]
                return
            end = buffer.find(EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                if len(buffer) > config.MAX_JPEG_BYTES:
                    # Tampon incoherent: on jette et on repart au prochain SOI.
                    buffer.clear()
                return
            end += 2
            self._publish(bytes(buffer[start:end]))
            del buffer[:end]

    def _pump_snapshots(self) -> None:
        url = self._url(config.SNAPSHOT_PATH)
        self._on_log(f"mode snapshot sur {url}")
        self.stats.connected = True
        while not self._stop.is_set():
            req = urllib.request.Request(url, headers={"Connection": "close"})
            with urllib.request.urlopen(req, timeout=config.READ_TIMEOUT_S) as resp:
                data = resp.read()
            if not data.startswith(SOI):
                raise ValueError("reponse /jpg non JPEG")
            self._publish(data)


def http_get(host: str, path: str, port: int = config.DEFAULT_PORT, timeout: float = 3.0) -> str:
    """Appel utilitaire pour /status et /control. Renvoie le corps en texte."""
    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")
